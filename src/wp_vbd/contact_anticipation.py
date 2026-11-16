"""Positive inverse-secant metric for a newly encountered finite contact.

The physical force and merit do not change. An inactive contact can have zero
local curvature while every fixed trial overshoots a stiff activation edge.
A metric based on the finite one-dimensional constitutive response resolves
that gap without adding primal iterations or shrinking the declared alpha set.
"""
import torch
from .mixed_metric import frozen_contact_term
from .data import State
from . import math3d as so3


def secant_weight(stiffness, local_support, gap, closing):
    # Twice the pair support bounds the independent-endpoint Jacobi scale.
    # Weight stays in [0,K]; gap=0 continuously recovers active stiffness.
    support=2*local_support
    penetration=(closing-gap).clamp_min(0)
    denominator=support*closing+stiffness*gap
    return torch.where((closing>0)&(support>0),support*stiffness*penetration/denominator.clamp_min(1e-300),0.).clamp_min(0)


@torch.no_grad()
def metric(scene,state,forecast,elastic,contacts,result,residual):
    if not len(contacts.owners):return [],result['active'],torch.ones(len(state.x),device=scene.device,dtype=torch.bool)
    local=elastic.diagonal.double()
    ghost=torch.diag_embed((~scene.dof).double())[None]
    inverse=torch.linalg.inv_ex(local+ghost,check_errors=False)
    inverse_value=inverse.inverse*scene.dof[None,:,:,None]*scene.dof[None,:,None,:]
    jacobian=torch.einsum('ri,brsij->brsj',contacts.normal,result['relative_skeleton_jacobian']).double()
    compliance=torch.einsum('brsi,brsij,brsj->br',jacobian,inverse_value[:,contacts.owners],jacobian)
    support=torch.where(compliance>0,compliance.clamp_min(1e-300).reciprocal(),0.)
    _,_,_,_,target_depth=contacts.geometry(scene,forecast)
    # Material forcing, especially a moving pin, need not point toward the
    # inertial target. Use the already assembled local inverse to forecast
    # that correction too; no candidate state or extra primal step commits.
    local_delta=-(inverse_value@residual.double()[...,None])[...,0]
    travel=local_delta[...,:3].norm(dim=-1)+scene.body_radius*local_delta[...,3:].norm(dim=-1)
    bound=scene.geometry_oracle.motion_budget
    local_delta=local_delta*(bound/travel.clamp_min(bound)).clamp_max(1)[...,None]
    local_q=so3.normalize(so3.multiply(so3.exp(local_delta[...,3:]),state.q))
    local=State(state.x+local_delta[...,:3],local_q,state.v,state.omega)
    _,_,_,_,local_depth=contacts.geometry(scene,local)
    # Transmit the direction of a mesh's net material drive to its mobile
    # vertices. This catches nonlocal motion caused by a prescribed boundary
    # while avoiding a directionless near-contact attraction on withdrawal.
    labels=scene.material_object_index;count=len(scene.material_object_keys)
    maximum=travel.new_zeros(len(travel),count).scatter_reduce(1,labels[None].expand_as(travel),local_delta[...,:3].norm(dim=-1),reduce='amax',include_self=True)
    drive=local_delta.new_zeros(len(travel),count,3).index_add(1,labels,-residual[...,:3])
    shift=drive/drive.norm(dim=-1,keepdim=True).clamp_min(1e-30)*maximum[...,None]
    object_x=state.x+shift[:,labels]*scene.movable[None,:,None]
    object_x=torch.where((scene.kind<2)[None,:,None],object_x,local.x)
    _,_,_,_,object_depth=contacts.geometry(scene,State(object_x,local_q,state.v,state.omega))
    target_depth=torch.maximum(target_depth,torch.maximum(local_depth,object_depth))
    current=result['raw_depth'];gap=(-current).clamp_min(0);closing=(target_depth-current).clamp_min(0)
    weight=secant_weight(contacts.material[None],support,gap,closing)
    weight=torch.where((current<0)&(target_depth>0),weight,0.)
    valid=(inverse.info==0).all(-1)&torch.isfinite(weight).all(-1)&torch.isfinite(inverse_value).flatten(1).all(-1)
    topology=result['active']|(weight>0)
    term=frozen_contact_term(contacts,jacobian[:,:,None].detach(),weight[...,None].detach(),
                             compact=bool(getattr(scene,'compact_frozen_contact_metric',False)),
                             cache_name='_normal_anticipation_metric_selection')
    return [term],topology,valid
