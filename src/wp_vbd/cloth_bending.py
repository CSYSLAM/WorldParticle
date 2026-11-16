"""Intrinsic low-frequency bending modes for a structured, edge-pinned sheet.

Detect the geometry, not a demo ID. Independent section rotations describe
isometric cylindrical bending, including the constant-angle hinge mode.
All other geometries keep the general response and geometric retraction.
"""
from dataclasses import dataclass
import hashlib

import numpy as np
import torch

from . import math3d as so3


@dataclass
class BendingPlan:
    entities: torch.Tensor
    mobile: torch.Tensor
    sections: torch.Tensor
    mobile_section: torch.Tensor
    endpoints: tuple
    functions: torch.Tensor
    signature: str

    def geometry(self,state):
        centers=state.x[:,self.sections].mean(2)
        axis=so3.normalize(state.x[:,self.endpoints[1]]-state.x[:,self.endpoints[0]])
        spans=centers[:,1:]-centers[:,:-1]
        derivative=torch.linalg.cross(axis[:,None].expand_as(spans),spans)
        increments=derivative[...,None]*self.functions[None,:,None,:]
        zero=increments.new_zeros(increments.shape[0],1,3,increments.shape[-1])
        basis=torch.cat((zero,increments.cumsum(1)),1)[:,self.mobile_section]
        return axis,spans,basis

    def modes(self,state):
        basis=self.geometry(state)[2]
        return torch.cat((basis,torch.zeros_like(basis)),2)

    def decompose(self,scene,state,update):
        axis,spans,basis=self.geometry(state)
        b=basis.double();mass=scene.mass[self.mobile].double()
        value=update[:,self.mobile,:3].double()
        gram=torch.einsum('bnip,n,bniq->bpq',b,mass,b)
        rhs=torch.einsum('bnip,n,bni->bp',b,mass,value)
        coefficients=torch.linalg.solve_ex(gram,rhs[...,None],check_errors=False).result[...,0].to(basis.dtype)
        angles=coefficients@self.functions.T
        remainder=update[:,self.mobile,:3]-torch.einsum('bnip,bp->bni',basis,coefficients)
        return axis,spans,angles,remainder

    def retract(self,scene,state,update):
        axis,spans,angles,remainder=self.decompose(scene,state,update)
        rotated=so3.rotate(so3.exp(axis[:,None]*angles[...,None]),spans)-spans
        shift=torch.cat((rotated.new_zeros(rotated.shape[0],1,3),rotated.cumsum(1)),1)[:,self.mobile_section]
        return state.x[:,self.mobile]+shift+remainder

    def plane_bound(self,scene,state,update):
        _,spans,angles,remainder=self.decompose(scene,state,update)
        arc=spans.norm(dim=-1)*angles.abs()
        travel=torch.cat((arc.new_zeros(arc.shape[0],1),arc.cumsum(1)),1)[:,self.mobile_section]
        return travel+(-remainder[...,2]).clamp_min(0)


def detect_bending_plan(scene,ids):
    """Accept only a planar regular grid with one entire edge prescribed."""
    rest=scene.rest_x[ids].double().cpu().numpy();movable=scene.movable[ids].cpu().numpy()
    pinned=np.where(~movable)[0]
    if len(pinned)<2 or movable.sum()<4:return None
    span=max(np.ptp(rest,axis=0).max(),1e-8);tolerance=span*1e-5
    _,singular,vt=np.linalg.svd(rest-rest.mean(0),full_matrices=False)
    if singular[1]<tolerance or singular[-1]>tolerance:return None
    fixed=rest[pinned];_,sv,axes=np.linalg.svd(fixed-fixed.mean(0),full_matrices=False)
    if len(sv)>1 and sv[1]>tolerance:return None
    along=axes[0];transverse=np.cross(along,vt[-1]);transverse/=np.linalg.norm(transverse)
    coordinate=rest@transverse;anchor=float(coordinate[pinned].mean())
    if abs(anchor-coordinate.min())<tolerance:coordinate=coordinate-anchor
    elif abs(anchor-coordinate.max())<tolerance:coordinate=anchor-coordinate
    else:return None
    keys=np.rint(coordinate/tolerance).astype(np.int64);unique=np.unique(keys)
    if len(unique)<3:return None
    groups=[np.where(keys==key)[0] for key in unique]
    if len({len(g) for g in groups})!=1:return None
    if set(groups[0])!=set(pinned):return None
    group_index=np.empty(len(ids),np.int64)
    for j,g in enumerate(groups):group_index[g]=j
    lookup=np.full(scene.n,-1,np.int64);lookup[ids.cpu().numpy()]=group_index
    for incidence in (scene.triangles,scene.bend_edges[scene.bending_indices]):
        rows=incidence.cpu().numpy();valid=(rows>=0).all(-1)
        rows=rows[valid];inside=(lookup[rows]>=0).all(-1);rows=rows[inside]
        # Faces span at most one section; bend hinges may span two.
        limit=1 if incidence.shape[1]==3 else 2
        if len(rows) and np.ptp(lookup[rows],axis=1).max()>limit:return None
    order=np.argsort(fixed@along);endpoints=tuple(ids[pinned[order[[0,-1]]]].tolist())
    section_coordinate=np.array([coordinate[g].mean() for g in groups])
    middle=.5*(section_coordinate[1:]+section_coordinate[:-1])/section_coordinate[-1]
    count=min(6,len(groups)-1)
    functions=np.cos(np.pi*middle[:,None]*np.arange(count)[None])
    sections=ids[torch.as_tensor(np.stack(groups),device=scene.device)]
    mobile=ids[torch.as_tensor(np.where(movable)[0],device=scene.device)]
    signature=hashlib.sha256(sections.cpu().numpy().tobytes()+functions.tobytes()+np.asarray(endpoints).tobytes()).hexdigest()
    return BendingPlan(ids,mobile,sections,torch.as_tensor(group_index[movable],device=scene.device),endpoints,torch.as_tensor(functions,device=scene.device,dtype=scene.rest_x.dtype),signature)
