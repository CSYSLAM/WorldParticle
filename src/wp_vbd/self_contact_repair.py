"""Bounded local contact repair with an explicit finite-overlap tolerance.

Projection gaps use the last accepted feature frames. They are diagnostics of
oriented feature overlap, not an exact solid penetration-depth certificate.
The auxiliary straight-through target teaches proposals to need less repair.
"""
import time

import torch


def repair_positions(snapshot,positions,mass,movable,*,rest_positions,length,sweeps=8,local_fraction=.01,maximum_fraction=.001):
    if positions.dtype!=torch.float64 or positions.shape!=snapshot.reference.shape:
        raise ValueError('Matched F64 particle positions required')
    if type(sweeps) is not int or not 1<=sweeps<=16 or not 0<local_fraction<.1 or not 0<maximum_fraction<.01:
        raise ValueError('Explicit small overlap tolerances and a bounded repair budget required')
    n=len(positions)
    if rest_positions.shape!=positions.shape:raise ValueError('Fixed rest-geometry feature scales required')
    if mass.shape!=(n,) or movable.shape!=(n,) or movable.dtype!=torch.bool:
        raise ValueError('Matched physical mass and movable-particle mask required')
    if not bool(torch.isfinite(positions).all()) or bool((movable&(mass<=0)).any()):raise ValueError('Finite physical repair inputs required')
    began=time.perf_counter();constraints=[]
    with torch.no_grad():
        inverse=torch.where(movable,mass.clamp_min(1e-30).reciprocal(),0.)
        for owners,geometry in zip(snapshot.owners,snapshot.geometry):
            if not len(owners):continue
            ref=rest_positions[owners];split=3 if geometry.kind==9 else 2
            edge_lengths=[(ref[:,1]-ref[:,0]).norm(dim=-1)]
            if geometry.kind==9:edge_lengths.extend([(ref[:,2]-ref[:,1]).norm(dim=-1),(ref[:,0]-ref[:,2]).norm(dim=-1)])
            else:edge_lengths.append((ref[:,3]-ref[:,2]).norm(dim=-1))
            feature=torch.stack(edge_lengths).amin(0)
            tolerance=torch.minimum(feature*local_fraction,feature.new_full(feature.shape,length*maximum_fraction)).clamp_min(length*1e-10)
            for a in range(split):
                for b in range(split,4):constraints.append((owners[:,a],owners[:,b],geometry.separating_axis,tolerance))
        work=positions.detach().clone();before=0.;fixed_bad=0
        for a,b,axis,tol in constraints:
            depth=-((work[b]-work[a])*axis).sum(-1)
            before=max(before,float(depth.clamp_min(0).max()))
            fixed_bad+=int(((~movable[a])&(~movable[b])&(depth>tol)).sum())
        for _ in range(sweeps):
            delta=torch.zeros_like(work);degree=mass.new_zeros(n)
            for a,b,axis,tol in constraints:
                gap=((work[b]-work[a])*axis).sum(-1)
                error=(-.5*tol-gap).clamp_min(0.)
                denominator=inverse[a]+inverse[b]
                value=torch.where(denominator>0,error/denominator.clamp_min(1e-30),0.)
                impulse=value[:,None]*axis
                delta.index_add_(0,a,-inverse[a,None]*impulse);delta.index_add_(0,b,inverse[b,None]*impulse)
                active=(value>0).to(degree.dtype)
                degree.index_add_(0,a,active);degree.index_add_(0,b,active)
            relaxation=.8/degree.max().clamp_min(1.)
            trial=work+relaxation*delta
            offset=trial-snapshot.reference;travel=offset.norm(dim=-1)
            cap=(.99*snapshot.motion_budget/travel.clamp_min(1e-30)).clamp_max(1.)
            limited=snapshot.reference+offset*cap[:,None]
            work=torch.where(movable[:,None],limited,positions.detach())
        after=0.;ratio=0.;excess=0
        for a,b,axis,tol in constraints:
            depth=-((work[b]-work[a])*axis).sum(-1)
            after=max(after,float(depth.clamp_min(0).max()));ratio=max(ratio,float((depth/tol).clamp_min(0).max()))
            excess+=int((depth>tol).sum())
        correction=work-positions.detach()
        if not torch.equal(work[~movable],positions.detach()[~movable]):raise AssertionError('Contact repair changed a prescribed boundary')
    corrected=positions+correction
    auxiliary=((positions-work.detach()).square().sum(-1)*movable).sum()/movable.sum().clamp_min(1)/length**2
    detail={'sweeps':sweeps,'local_feature_overlap_fraction':local_fraction,'maximum_overlap_length_fraction':maximum_fraction,
        'maximum_oriented_overlap_before_m':before,'maximum_oriented_overlap_after_m':after,
        'maximum_oriented_overlap_to_tolerance_ratio':ratio,'remaining_excess_constraints':excess,
        'prescribed_prescribed_excess_constraints':fixed_bad,'maximum_position_repair_m':float(correction.norm(dim=-1).max()),
        'seconds':time.perf_counter()-began,'prescribed_positions_unchanged':True,
        'scope':'Finite-budget mass-weighted contact corrections, with fixed boundaries and unchanged query coverage. Oriented overlap tolerance is explicit. Not a zero-intersection or exact solid penetration certificate; material/orientation validation remains the caller responsibility. Auxiliary loss uses a detached repair target.'}
    return corrected,auxiliary,detail


def configure(scene,configuration):
    scene.learned_contact_repair=bool(configuration.get('learned_contact_repair',False))


def tolerances(snapshot,rest_positions,length):
    values=[]
    for owners,geometry in zip(snapshot.owners,snapshot.geometry):
        ref=rest_positions[owners]
        edges=[(ref[:,1]-ref[:,0]).norm(dim=-1)]
        if geometry.kind==9:edges.extend([(ref[:,2]-ref[:,1]).norm(dim=-1),(ref[:,0]-ref[:,2]).norm(dim=-1)])
        else:edges.append((ref[:,3]-ref[:,2]).norm(dim=-1))
        feature=torch.stack(edges).amin(0)
        values.append(torch.minimum(.01*feature,feature.new_full(feature.shape,.001*length)).clamp_min(length*1e-10))
    return values


def bounded_path(snapshot,positions,rest_positions,length):
    """Explicit oriented-overlap budget, including the prescribed motion."""
    from .self_contact_rows import safe_norm
    valid=torch.isfinite(positions).all()&(safe_norm(positions-snapshot.reference)<=snapshot.motion_budget).all()
    for owners,geometry,tol in zip(snapshot.owners,snapshot.geometry,tolerances(snapshot,rest_positions,length)):
        if not len(owners):continue
        p=positions[owners];axis=geometry.separating_axis;split=3 if geometry.kind==9 else 2
        projection=(p*axis[:,None]).sum(-1)
        gap=projection[:,split:].amin(-1)-projection[:,:split].amax(-1)
        valid=valid&(gap>=-tol).all()
    return valid


def candidate(scene,state,context):
    from .data import State
    if not getattr(scene,'learned_contact_repair',False) or not hasattr(context.contacts,'self_snapshot'):return state
    snapshot=context.contacts.self_snapshot;n=len(snapshot.reference);positions=state.x[0,:n]
    if bool(bounded_path(snapshot,positions,scene.rest_x[:n],scene.length)):return state
    repaired,auxiliary,detail=repair_positions(snapshot,positions,scene.mass[:n],scene.movable[:n],
        rest_positions=scene.rest_x[:n],length=scene.length)
    context.contact_repair_loss=getattr(context,'contact_repair_loss',positions.new_zeros(()))+auxiliary
    context.contact_repairs=getattr(context,'contact_repairs',[])+[detail]
    x=torch.cat((repaired[None],state.x[:,n:]),1)
    return State(x,state.q,state.v,state.omega,state.history)
