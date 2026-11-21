"""Bounded CPU call statistics and named CUDA-stream intervals, without traces."""
import cProfile
import functools
from contextlib import contextmanager
import json
from pathlib import Path
import pstats
import time

import torch


class ExecutionProfile:
    def __init__(self, response, device, output, *, python_calls=True, nvtx=False, prefix=''):
        self.output = Path(output)
        self.response = response
        self.device = torch.device(device)
        self.handles = []
        self.stack = []
        self.intervals = []
        self.patches=[]
        self.python_calls=python_calls
        self.nvtx=nvtx and self.device.type=='cuda'
        self.prefix=prefix

    def start(self):
        self.profiler = cProfile.Profile() if self.python_calls else None
        if self.profiler is not None:self.profiler.enable()
        if self.response is not None:
            def begin(module, unused):
                context = self.measure('wp.learned_response_including_physical_projection')
                context.__enter__()
                self.stack.append(context)
            def finish(module, unused, result):
                self.stack.pop().__exit__(None, None, None)
            self.handles = [self.response.register_forward_pre_hook(begin),
                            self.response.register_forward_hook(finish, always_call=True)]
            backbone=getattr(self.response,'hierarchy',None)
            if backbone is not None:
                frames=[]
                def start_backbone(module,unused):
                    context=self.measure('wp.shared_backbone');context.__enter__();frames.append(context)
                def end_backbone(module,unused,result):frames.pop().__exit__(None,None,None)
                self.handles.extend([backbone.register_forward_pre_hook(start_backbone),
                    backbone.register_forward_hook(end_backbone,always_call=True)])
        # Instrument coarse stages, not every tensor operator. Nested times
        # are explicitly inclusive and must not be added together.
        from . import surface_joint_system,protected_response,articulation_coarse,analytic_material_response
        from . import rigid_contact_solver,mixed_contact_runtime
        for owner,name,label in [
            (surface_joint_system,'system','wp.physical_assembly'),
            (surface_joint_system,'relation_features','wp.relation_features'),
            (surface_joint_system,'merit','wp.merit'),
            (surface_joint_system,'retract','wp.constraint_retraction'),
            (surface_joint_system,'diagnostics','wp.force_diagnostics'),
            (protected_response.ProtectedCoarse,'__init__','wp.protected_basis'),
            (articulation_coarse,'augment','wp.articulation_modes'),
            (analytic_material_response,'correct','wp.analytic_material_correction'),
            (rigid_contact_solver,'choose','wp.line_search'),
            (mixed_contact_runtime.MixedCollisionOracle,'query','wp.collision_query'),
            (mixed_contact_runtime.MixedContactExampleSolver,'read','wp.native_state_read'),
            (mixed_contact_runtime.MixedContactExampleSolver,'_write_native_output','wp.native_state_write')]:
            original=getattr(owner,name)
            def wrap(function,category):
                @functools.wraps(function)
                def measured(*args,**kwargs):
                    with self.measure(category):return function(*args,**kwargs)
                return measured
            setattr(owner,name,wrap(original,label));self.patches.append((owner,name,original))

    @contextmanager
    def measure(self, name):
        name=self.prefix+name
        if self.nvtx:torch.cuda.nvtx.range_push(name)
        start = end = None
        if self.device.type == 'cuda':
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        wall = time.perf_counter()
        try:
            yield
        finally:
            if end is not None:
                end.record()
            self.intervals.append((name, time.perf_counter() - wall, start, end))
            if self.nvtx:torch.cuda.nvtx.range_pop()

    def finish(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        for owner,name,original in reversed(self.patches):setattr(owner,name,original)
        self.patches=[]
        if self.profiler is not None:self.profiler.disable()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        rows = []
        if self.profiler is not None:
            for (file, line, name), (primitive, calls, own, total, callers) in pstats.Stats(self.profiler).stats.items():
                rows.append({'name': f'{file}:{line}:{name}', 'calls': calls,
                             'self_cpu_seconds': own, 'inclusive_cpu_seconds': total})
        selected = {row['name']: row for field in ('self_cpu_seconds', 'inclusive_cpu_seconds')
                    for row in sorted(rows, key=lambda row: row[field], reverse=True)[:40]}
        ranges = {}
        for name, wall, start, end in self.intervals:
            row = ranges.setdefault(name, {'calls': 0, 'wall_seconds': 0., 'cuda_stream_seconds': 0.})
            row['calls'] += 1
            row['wall_seconds'] += wall
            if start is not None:
                row['cuda_stream_seconds'] += start.elapsed_time(end) / 1000
        result = {'device': str(self.device), 'python_calls_enabled':self.python_calls,
                  'nvtx_enabled':self.nvtx, 'rows': list(selected.values()),
                  'ranges': ranges, 'retained_intervals': len(self.intervals),
                  'scope': 'Instrumented simulation only, excluding initialization. CUDA-stream intervals include '
                           'CPU submission gaps and stream idle time; these are not sums of kernel busy time. '
                           'Ranges are nested/inclusive: backbone is inside learned response; material/basis/line-search '
                           'and collision are inside solver. Do not add inclusive ranges. The response includes physical '
                           'projection, so it is not pure network latency. No per-operator event trace is retained; '
                           'profiler overhead invalidates speedup claims.'}
        self.output.write_text(json.dumps(result, indent=2) + '\n')
        return result
