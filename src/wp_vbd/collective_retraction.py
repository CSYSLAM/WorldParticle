"""Finite SE(3) transport of collective node proposals, plus local strain.

This is a geometric retraction of the proposed update, not a dynamics solve.
Only isolated all-cloth components are currently admitted: tet path guards
retain their exact polynomial linear path, and attachments retain their
jointly guarded original path. Every fitted system is 3 by 3.
"""
import torch
from . import math3d as so3


def decompose(scene,state,update,ids):
    # Static movable incidence is prepared outside CUDA capture. Boolean
    # selection here would require a device synchronization and dynamic size.
    mobile=ids
    mass=scene.mass[mobile];position=state.x[:,mobile]
    center=(position*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
    lever=position-center;velocity=update[:,mobile,:3]
    translation=(velocity*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
    eye=torch.eye(3,device=position.device,dtype=position.dtype)
    inertia=(mass[None,:,None,None]*(lever.square().sum(-1)[...,None,None]*eye-lever[...,None]*lever[...,None,:])).sum(1)
    angular_momentum=(mass[None,:,None]*torch.linalg.cross(lever,velocity-translation)).sum(1)
    rotation=torch.linalg.solve_ex(inertia.double(),angular_momentum.double()[...,None],check_errors=False).result[...,0].to(position.dtype)
    remainder=velocity-translation-torch.linalg.cross(rotation[:,None].expand_as(lever),lever)
    return mobile,center,lever,translation,rotation,remainder


def twist_translation(rotation,translation):
    squared=rotation.square().sum(-1,keepdim=True)
    angle=squared.clamp_min(1e-20).sqrt()
    a=torch.where(squared<1e-4,.5-squared/24+squared.square()/720,(1-angle.cos())/squared.clamp_min(1e-20))
    b=torch.where(squared<1e-4,1/6-squared/120+squared.square()/5040,(angle-angle.sin())/(squared*angle).clamp_min(1e-20))
    first=torch.linalg.cross(rotation,translation)
    return translation+a*first+b*torch.linalg.cross(rotation,first)


def retract(scene,state,delta):
    x=state.x+delta[...,:3]
    bending=getattr(scene,'use_bending_retraction',False)
    for ids in (scene.non_bending_cloth_components if bending else scene.cloth_retraction_components):
        mobile,_,lever,t,w,remainder=decompose(scene,state,delta,ids)
        rotated=so3.rotate(so3.exp(w)[:,None],lever)-lever
        shift=twist_translation(w,t[:,0])[:,None]
        x=x.index_copy(1,mobile,state.x[:,mobile]+rotated+shift+remainder)
    if bending:
        for plan in scene.bending_plans:x=x.index_copy(1,plan.mobile,plan.retract(scene,state,delta))
    return x


def plane_motion_bound(scene,state,update):
    result=update[...,:3].new_zeros(update.shape[:2])
    bending=getattr(scene,'use_bending_retraction',False)
    for ids in (scene.non_bending_cloth_components if bending else scene.cloth_retraction_components):
        mobile,_,lever,t,w,remainder=decompose(scene,state,update,ids)
        # Integral of rotated translation + rotation arc + local downward
        # strain is an upper bound on downward travel for EVERY alpha in [0,1].
        bound=t.norm(dim=-1)+w.norm(dim=-1)[:,None]*lever.norm(dim=-1)+(-remainder[...,2]).clamp_min(0)
        result=result.index_copy(1,mobile,bound)
    if bending:
        for plan in scene.bending_plans:result=result.index_copy(1,plan.mobile,plan.plane_bound(scene,state,update))
    return result
