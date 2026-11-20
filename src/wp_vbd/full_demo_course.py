"""Whole-timeline supervision plans with disjoint temporal development blocks.

The plans reference real teacher endpoints. A whole trajectory supplies the
windows; backpropagation remains bounded and never resets inside a window.
"""
import numpy as np

from .multirate_window import long_step_window


def authored_frame_horizon(path,default=100):
    """Read numeric authored frame defaults without importing/running a demo."""
    import ast
    import operator
    tree=ast.parse(path.read_text());environment={}
    binary={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,
            ast.Div:operator.truediv,ast.FloorDiv:operator.floordiv}
    def value(node,scope):
        if isinstance(node,ast.Constant):return node.value
        if isinstance(node,ast.Name):return scope[node.id]
        if isinstance(node,ast.Attribute):return value(node.value,scope)[node.attr]
        if isinstance(node,ast.Subscript):return value(node.value,scope)[value(node.slice,scope)]
        if isinstance(node,ast.BinOp) and type(node.op) in binary:
            return binary[type(node.op)](value(node.left,scope),value(node.right,scope))
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id in ('int','float') and len(node.args)==1:
            return (int if node.func.id=='int' else float)(value(node.args[0],scope))
        if isinstance(node,ast.Dict):
            result={}
            for key,item in zip(node.keys,node.values):
                try:result[value(key,scope)]=value(item,scope)
                except (KeyError,TypeError,ValueError):continue
            return result
        raise ValueError('Nonconstant expression')
    def assignments(body,scope):
        for node in body:
            targets=node.targets if isinstance(node,ast.Assign) else [node.target] if isinstance(node,ast.AnnAssign) else []
            for target in targets:
                if isinstance(target,ast.Name):
                    try:scope[target.id]=value(node.value,scope)
                    except (KeyError,TypeError,ValueError):pass
    assignments(tree.body,environment)
    for node in tree.body:
        if isinstance(node,ast.ClassDef):
            local=dict(environment);assignments(node.body,local);environment[node.name]=local
    result=int(default)
    for node in ast.walk(tree):
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='set_defaults':
            for keyword in node.keywords:
                if keyword.arg=='num_frames':result=int(value(keyword.value,environment))
    if result<=0:raise ValueError('Positive authored frame count required')
    return result


def timeline_catalog(grid, *, events=(), bins=6, hops=2,
                     target_dts=(1/180, 1/90, 1/60)):
    if bins < 1 or hops < 1:
        raise ValueError('Positive timeline and autonomous-horizon budgets required')
    events=sorted(set(events))
    training=[];development=[];blocks=[];missing=[]
    for segment,(first,last) in enumerate(grid.segments):
        edges=np.linspace(first,last,bins+1,dtype=np.int64)
        for phase,(lo,hi) in enumerate(zip(edges[:-1],edges[1:])):
            lo=int(lo);hi=int(hi)
            # Each part of the motion has a development block. The final fifth
            # of each part is never a training state, target or control span.
            split=lo+int((hi-lo)*.8)
            blocks.append({'segment':segment,'phase_bin':phase,'first':split,'last':hi})
            train_first=lo if phase==0 else lo+1
            for label,start,end,destination in [('training',train_first,split-1,training),
                                                 ('development',split,hi,development)]:
                before=len(destination)
                if end<=start:
                    missing.append({'segment':segment,'phase_bin':phase,'split':label});continue
                for duration in target_dts:
                    # Small/coarse native grids use their exact real endpoint;
                    # no interpolated target or invented fractional step.
                    native=float(np.median(np.diff(grid.times[start:end+1])))
                    duration=max(float(duration),native)
                    for candidate in np.linspace(start,end-1,8,dtype=np.int64):
                        candidate=int(candidate)
                        try:
                            plan=long_step_window(grid,candidate,[duration]*hops,events=events)
                        except ValueError:
                            continue
                        if plan['endpoint_indices'][-1]>end or len(plan['strides'])>hops+2:
                            continue
                        # Removal / feedback boundaries are split by the common
                        # planner; no call averages a discrete topology change.
                        plan.update(segment=segment,phase_bin=phase)
                        signature=(candidate,tuple(plan['strides']))
                        if not any((p['start'],tuple(p['strides']))==signature for p in destination[before:]):
                            destination.append(plan)
                if len(destination)==before:
                    missing.append({'segment':segment,'phase_bin':phase,'split':label})
    return {'training':{'windows':training},'development':{'windows':development},
            'reserved_development_blocks':blocks,'missing_phase_splits':missing,
            'phase_bins':bins,'horizon':hops,
            'split_scope':'Disjoint intervals of the same authored trajectory; not held-out assets or materials'}


def select_timeline_plan(catalog, occurrence, *, development=False,complete_motion_sampling=False):
    rows=catalog['development' if development else 'training']['windows']
    groups={}
    for plan in rows:
        groups.setdefault((plan['segment'],plan['phase_bin']),[]).append(plan)
    if not groups:
        raise ValueError('No admitted timeline windows')
    keys=sorted(groups)
    key=keys[occurrence%len(keys)]
    options=groups[key]
    durations={}
    for plan in options:
        signature=round(sum(plan['durations']),12)
        durations.setdefault(signature,[]).append(plan)
    cycle=occurrence//len(keys)
    duration_keys=sorted(durations)
    selected=durations[duration_keys[cycle%len(duration_keys)]]
    if complete_motion_sampling and not development:
        # Always include the beginning and the latest training interval.
        # The latter reaches the stop/settle phase of motions such as the
        # ten-second twist. Interior bins use representative middle windows;
        # they must not always start at each bin's first recorded frame.
        if key==keys[0]:return selected[(cycle//len(duration_keys))%len(selected)]
        if key==keys[-1]:return selected[-1-((cycle//len(duration_keys))%len(selected))]
        return selected[(len(selected)//2+cycle//len(duration_keys))%len(selected)]
    return selected[(cycle//len(duration_keys))%len(selected)]


def device_window(states, start, span, device):
    """Upload one complete autonomous window, including its accepted history.

    Scene geometry stays resident. Targets stay separate from exogenous input
    channels. The student carries its own history between network calls.
    """
    result={}
    for key,value in states.items():
        result[key]=value[start:start+span+1].to(device)
    return result
