"""Bounded, dtype-preserving tensor copies when a diagnostic is serialized."""
from collections import defaultdict

import torch


def to_plain(value,*,maximum_pack_bytes=8*1024**2):
    """Convert nested tensors to Python values, batching small CUDA copies.

    Do not call inside a physics iteration to validate a state. This is only
    the recorder boundary: accepted state, history and decisions are untouched.
    Integer and boolean tensors retain their own dtype, never a float carrier.
    """
    if maximum_pack_bytes<1:raise ValueError('Positive diagnostic packing budget required')
    tensors={};plain={};groups=defaultdict(list)
    def collect(item):
        if isinstance(item,torch.Tensor):tensors.setdefault(id(item),item)
        elif isinstance(item,dict):
            for child in item.values():collect(child)
        elif isinstance(item,(list,tuple)):
            for child in item:collect(child)
    collect(value)
    for key,item in tensors.items():
        source=item.detach()
        if source.device.type!='cuda' or not source.numel() or source.numel()*source.element_size()>maximum_pack_bytes:
            plain[key]=source.cpu().tolist()
        else:groups[(source.device,source.dtype)].append((key,source))
    with torch.no_grad():
        for entries in groups.values():
            chunk=[];size=0
            def flush():
                if not chunk:return
                host=torch.cat([item.reshape(-1) for _,item in chunk]).cpu()
                offset=0
                for key,item in chunk:
                    count=item.numel();plain[key]=host[offset:offset+count].reshape(item.shape).tolist();offset+=count
            for key,item in entries:
                item_bytes=item.numel()*item.element_size()
                if chunk and size+item_bytes>maximum_pack_bytes:flush();chunk=[];size=0
                chunk.append((key,item));size+=item_bytes
            flush()
    def rebuild(item):
        if isinstance(item,torch.Tensor):return plain[id(item)]
        if isinstance(item,dict):return {key:rebuild(child) for key,child in item.items()}
        if isinstance(item,list):return [rebuild(child) for child in item]
        if isinstance(item,tuple):return tuple(rebuild(child) for child in item)
        return item
    return rebuild(value)
