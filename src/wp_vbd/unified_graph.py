"""DOF/hyperedge topology shared by all physical backends.

Build outside CUDA capture after potential-contact generation. A packed batch
owns disjoint node ranges. Static world nodes never connect dynamic systems.
This module represents relations; it does not supply their physical laws.
"""
from dataclasses import dataclass
from types import SimpleNamespace

import torch


@dataclass(frozen=True)
class HeterogeneousTopology:
    n: int
    kind: torch.Tensor
    movable: torch.Tensor
    object_id: torch.Tensor
    batch_id: torch.Tensor
    interaction_id: torch.Tensor
    layout: object
    potential_active: torch.Tensor
    topology_generation: int

    def validate_active(self, active):
        """Call before replay; a new pair requires a new potential graph."""
        if active.shape[-1]!=len(self.potential_active):raise ValueError('Relation capacity changed')
        if bool((active.bool()&~self.potential_active).any()):
            raise ValueError('An active relation is outside the declared potential graph')


def build_topology(kind,movable,object_id,incidence,relation_kind,*,batch_id=None,potential_active=None,generation=0):
    n=len(kind);device=kind.device
    if batch_id is None:batch_id=torch.zeros(n,device=device,dtype=torch.long)
    if potential_active is None:potential_active=torch.ones(len(incidence),device=device,dtype=torch.bool)
    if kind.shape!=(n,) or movable.shape!=(n,) or object_id.shape!=(n,) or batch_id.shape!=(n,):raise ValueError('Invalid node topology shapes')
    if incidence.ndim!=2 or incidence.shape[1]<1 or relation_kind.shape!=(len(incidence),):raise ValueError('Invalid hyperedge shapes')
    if potential_active.shape!=(len(incidence),) or potential_active.dtype!=torch.bool:raise ValueError('Explicit Boolean potential mask required')
    if bool(((kind<0)|(kind>3)).any()) or movable.dtype!=torch.bool:raise ValueError('Unknown DOF type or mobility')
    if bool(((incidence < -1)|(incidence>=n)).any()):raise ValueError('Invalid hyperedge owner')
    if bool(((relation_kind<0)|(relation_kind>=16)).any()):raise ValueError('Unknown relation type')
    parent=list(range(n));mobile=movable.tolist();batches=batch_id.tolist()
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for row,potential in zip(incidence.tolist(),potential_active.tolist()):
        if not potential:continue
        members=[i for i in row if i>=0]
        if not members:raise ValueError('A potential relation has no DOF participants')
        if len({batches[i] for i in members})!=1:raise ValueError('Cross-batch physical relation')
        ids=[i for i in members if mobile[i]]
        for i in ids[1:]:parent[root(i)]=root(ids[0])
    component=torch.tensor([root(i) for i in range(n)],device=device)
    slots=tuple((torch.where(incidence[:,slot]>=0)[0],incidence[incidence[:,slot]>=0,slot]) for slot in range(incidence.shape[1]))
    layout=SimpleNamespace(incidence=incidence,relation_kind=relation_kind,slots=slots)
    return HeterogeneousTopology(n,kind,movable,object_id,batch_id,component,layout,potential_active,int(generation))
