"""Exact storage sharing along the time axis of read-only reference arrays.

No states, inputs or frames are removed. Only an array whose complete byte
representation is identical at every time is represented by an expanded view.
Consumers must keep reference tensors read-only and copy working states.
"""
import numpy as np
import torch


def temporal_tensor(value, device='cpu', *, compact_constants=False, scan_chunk_bytes=8*1024**2):
    if not isinstance(value, np.ndarray) or value.ndim < 1 or value.dtype.hasobject:
        raise ValueError('A numeric time-indexed reference array is required')
    if type(scan_chunk_bytes) is not int or scan_chunk_bytes <= 0:
        raise ValueError('A positive byte scan budget is required')
    if compact_constants and torch.device(device).type != 'cpu':
        raise ValueError('Exact constant reference storage is an explicit host-trajectory policy')
    shared = False
    if compact_constants and len(value) > 1:
        first = np.ascontiguousarray(value[:1]).copy()
        frame_bytes = first.nbytes
        expected = first.view(np.uint8).reshape(1, frame_bytes)
        chunk_rows = max(1, scan_chunk_bytes//max(frame_bytes, 1))
        for start in range(1, len(value), chunk_rows):
            block = np.ascontiguousarray(value[start:start+chunk_rows])
            observed = block.view(np.uint8).reshape(len(block), frame_bytes)
            if not bool((observed == expected).all()):
                break
        else:
            shared = True
    tensor = (torch.as_tensor(first, device=device).expand(value.shape) if shared
              else torch.as_tensor(value, device=device))
    return tensor, {'representation': 'constant_time_axis' if shared else 'dense',
                    'shape': list(value.shape), 'dtype': str(value.dtype),
                    'logical_bytes': tensor.numel()*tensor.element_size(),
                    'storage_bytes': tensor.untyped_storage().nbytes(),
                    'all_time_rows_verified_bitexact': shared}


def storage_summary(states, fields):
    unique = {}
    for value in states.values():
        storage = value.untyped_storage()
        if storage.nbytes():
            unique[(str(value.device), storage.data_ptr())] = storage.nbytes()
    return {'constant_time_storage_enabled': any(row['representation'] == 'constant_time_axis'
                                                for row in fields.values()),
            'logical_bytes': sum(value.numel()*value.element_size() for value in states.values()),
            'unique_tensor_storage_bytes': sum(unique.values()), 'fields': fields,
            'scope': 'Read-only reference tensor storage, excluding native scene geometry, working states, '
                     'temporary archive loading allocations, neural activations, allocator overhead and Python objects.'}
