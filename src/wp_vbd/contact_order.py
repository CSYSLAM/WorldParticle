"""Canonical complete contact rows, preserving native export slot mappings."""
from dataclasses import fields,replace

import numpy as np
import torch


def canonical_contacts(packet):
    count=len(packet.owners)
    if not count:return packet
    tensors={field.name:getattr(packet,field.name) for field in fields(packet) if isinstance(getattr(packet,field.name),torch.Tensor)}
    if any(value.ndim<1 or len(value)!=count for value in tensors.values()):
        raise ValueError('Every contact tensor must have one leading row per candidate')
    primary=[name for name in ('relation_kind','source_kind','owners','shapes','point_ids') if name in tensors]
    def columns(names):
        values=[]
        for name in names:
            array=tensors[name].detach().cpu().numpy().reshape(count,-1)
            values.extend(array[:,column] for column in range(array.shape[1]))
        return values
    keys=columns(primary)
    order=np.lexsort(tuple(reversed(keys)))
    # Particle/shape/feature identities normally identify a unique row.
    # Full geometry/material tie-breaks keep duplicate features complete.
    duplicate=np.ones(count-1,dtype=bool)
    for key in keys:duplicate&=key[order[1:]]==key[order[:-1]]
    if duplicate.any():
        secondary=[name for name in tensors if name not in (*primary,'source_slots')]
        keys+=columns(secondary);order=np.lexsort(tuple(reversed(keys)))
    if np.array_equal(order,np.arange(count)):return packet
    index=torch.as_tensor(order.copy(),device=packet.owners.device,dtype=torch.long)
    # source_slots is permuted along with its contact, so force readout still
    # targets the original native row; it never participates in ordering.
    return replace(packet,**{name:value.index_select(0,index) for name,value in tensors.items()})
