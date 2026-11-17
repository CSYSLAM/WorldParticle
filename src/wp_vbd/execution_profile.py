"""Bounded CPU call statistics and named CUDA-stream intervals, without traces."""
import cProfile
from contextlib import contextmanager
import json
from pathlib import Path
import pstats
import time

import torch


class ExecutionProfile:
    def __init__(self, response, device, output):
        self.output = Path(output)
        self.response = response
        self.device = torch.device(device)
        self.handles = []
        self.stack = []
        self.intervals = []

    def start(self):
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        if self.response is not None:
            def begin(module, unused):
                context = self.measure('wp.neural_response')
                context.__enter__()
                self.stack.append(context)
            def finish(module, unused, result):
                self.stack.pop().__exit__(None, None, None)
            self.handles = [self.response.register_forward_pre_hook(begin),
                            self.response.register_forward_hook(finish, always_call=True)]

    @contextmanager
    def measure(self, name):
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

    def finish(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self.profiler.disable()
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        rows = []
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
        result = {'device': str(self.device), 'rows': list(selected.values()),
                  'ranges': ranges, 'retained_intervals': len(self.intervals),
                  'scope': 'Instrumented simulation only, excluding initialization. CUDA-stream intervals include '
                           'CPU submission gaps and stream idle time; these are not sums of kernel busy time. '
                           'Neural intervals are nested in solver intervals. No per-operator event trace is retained; '
                           'profiler overhead invalidates speedup claims.'}
        self.output.write_text(json.dumps(result, indent=2) + '\n')
        return result
