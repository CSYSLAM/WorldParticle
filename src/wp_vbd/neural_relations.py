"""Pack active neural messages without changing physical/geometric rows."""
from types import SimpleNamespace

import torch


def active_layout(layout, active):
    """Keep ordered relations active in any batch member.

    This is an eager neural optimization. The physical topology, collision
    candidates, forces, material rows and feature-path guards are untouched.
    Active-set changes require a fresh pack on every neural evaluation.
    """
    if active.ndim!=2 or active.shape[1]!=len(layout.incidence):
        raise ValueError('Matched batched relation activity required')
    if active.is_cuda and torch.cuda.is_current_stream_capturing():
        raise ValueError('Dynamic neural relation packing is not CUDA-graph captured')
    selected=torch.where(active.detach().bool().any(0))[0]
    inverse=torch.full((len(layout.incidence),),-1,device=selected.device,dtype=torch.long)
    inverse[selected]=torch.arange(len(selected),device=selected.device)
    slots=[]
    for rows,ids in layout.slots:
        keep=inverse[rows]>=0
        slots.append((inverse[rows[keep]],ids[keep]))
    packed=SimpleNamespace(incidence=layout.incidence[selected],
                           relation_kind=layout.relation_kind[selected],slots=tuple(slots))
    if hasattr(layout,'legacy_kind'):packed.legacy_kind=layout.legacy_kind[selected]
    return packed,selected
