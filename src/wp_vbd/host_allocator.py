"""Explicit reclamation of unused glibc CPU pages for offline training jobs."""
import ctypes
import gc
from pathlib import Path


def _rss():
    for line in Path('/proc/self/status').read_text().splitlines():
        if line.startswith('VmRSS:'):return int(line.split()[1])*1024
    raise RuntimeError('Cannot observe this process host RSS')


def release_unused_host_memory():
    """Collect unreachable objects and release allocator pages, never live tensors.

    This Linux/glibc resource policy is optional. Its observed RSS change is
    diagnostic; neither a positive release nor an absolute memory bound is
    guaranteed. It does not empty CUDA caches or alter any reference/state.
    """
    library=ctypes.CDLL(None)
    try:trim=library.malloc_trim
    except AttributeError as error:raise RuntimeError('Host trimming requires glibc malloc_trim') from error
    trim.argtypes=[ctypes.c_size_t];trim.restype=ctypes.c_int
    before=_rss();collected=gc.collect();result=int(trim(0));after=_rss()
    return {'rss_before_bytes':before,'rss_after_bytes':after,
            'observed_rss_decrease_bytes':before-after,'unreachable_objects_collected':collected,
            'glibc_trim_return':result}
