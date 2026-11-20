"""Bound Torch allocations and observe native CUDA process memory."""
import math
import os
import subprocess
import time

import torch


class GpuBudget:
    def __init__(self, device, gib=12.):
        if not math.isfinite(gib) or gib<=2:
            raise ValueError('GPU budget must be finite and exceed the two GiB native reserve')
        self.device=torch.device(device)
        self.gib=gib
        self.observed_mib=0
        self.samples=0
        self.last_observation=-float('inf')
        self.trim_unused_cache=False
        self.cache_trim_count=0
        self.cache_trim_released_bytes=0
        if self.device.type=='cuda':
            total=torch.cuda.get_device_properties(self.device).total_memory
            torch.cuda.set_per_process_memory_fraction(min((gib-2)*2**30/total,1.),self.device)
            torch.cuda.reset_peak_memory_stats(self.device)

    def observe(self, force=False):
        if self.device.type!='cuda' or not force and time.perf_counter()-self.last_observation<2:
            return
        if self.trim_unused_cache:
            reserved=torch.cuda.memory_reserved(self.device)
            allocated=torch.cuda.memory_allocated(self.device)
            if reserved>max(2.,self.gib-4)*2**30 and reserved-allocated>1024**3:
                # Release idle allocator blocks, never live state, autograd
                # activations or geometry. Native contact queries and Torch
                # use separate pools; idle Torch blocks otherwise inflate
                # the process budget after a large variable-size scene.
                torch.cuda.empty_cache()
                self.cache_trim_count+=1
                self.cache_trim_released_bytes+=max(0,reserved-torch.cuda.memory_reserved(self.device))
        result=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory',
                               '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        values=[]
        for line in result.stdout.splitlines():
            fields=line.split(',')
            if len(fields)==2 and fields[0].strip()==str(os.getpid()):
                values.append(float(fields[1].strip()))
        if not values:raise RuntimeError('Cannot observe this CUDA process memory budget')
        self.observed_mib=max(self.observed_mib,max(values))
        self.samples+=1
        self.last_observation=time.perf_counter()
        if max(values)>self.gib*1024:
            raise RuntimeError(f'GPU process budget exceeded: {max(values):g} MiB > {self.gib*1024:g} MiB')
        return max(values)

    def report(self):
        result={'enabled':self.device.type=='cuda','process_budget_gib':self.gib,
                'torch_allocator_cap_gib':self.gib-2,'native_reserve_gib':2,
                'observed_process_max_mib':self.observed_mib,'process_observations':self.samples,
                'trim_idle_torch_cache_before_observation':self.trim_unused_cache,
                'idle_cache_trims':self.cache_trim_count,'idle_cache_bytes_released':self.cache_trim_released_bytes,
                'scope':'Torch allocator cap; native process memory sampled between frames at most every two seconds. Native allocations have no hard allocator cap; transient process peaks may be missed. Monitoring is included in inference wall time.'}
        if self.device.type=='cuda':
            result.update(peak_torch_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                          peak_torch_reserved_bytes=torch.cuda.max_memory_reserved(self.device))
        return result
