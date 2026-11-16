"""Objective joint coordinates, maximal-body Jacobians and finite drive rows.

Independent local operators, not a complete solver. D6 follows the pinned
Newton intrinsic-Euler axis order, including the two-axis locked middle
coordinate and left-handed triples. The physical log Jacobian is exact;
source_approx=True is only an offline comparison to Newton's enabled
small-angle approximation.
"""
import torch
from . import math3d as so3
from .response_operator import skew


def log_coordinates(parent,child,parent_rest,child_rest,*,source_approx=False,jacobian=True):
    relative=so3.multiply(so3.conjugate(parent),child)
    rest=so3.multiply(so3.conjugate(parent_rest),child_rest)
    aligned=so3.normalize(so3.multiply(relative,so3.conjugate(rest)))
    value=so3.log(aligned)
    if not jacobian:return value,None
    s=skew(value)
    identity=torch.eye(3,device=value.device,dtype=value.dtype)
    rotation=so3.matrix(parent)
    if source_approx:
        covector=(identity+.5*s+(s@s)/12)@so3.matrix(aligned).transpose(-1,-2)@rotation.transpose(-1,-2)
    else:
        squared=value.square().sum(-1);angle=squared.clamp_min(1e-20).sqrt()
        coefficient=(1-.5*angle*torch.cos(angle/2)/torch.sin(angle/2).clamp_min(1e-20))/squared.clamp_min(1e-20)
        coefficient=torch.where(squared<1e-6,1/12+squared/720+squared.square()/30240,coefficient)
        covector=(identity-.5*s+coefficient[...,None,None]*(s@s))@rotation.transpose(-1,-2)
    return value,covector


def d6_angular_coordinates(axes,count,parent,child,*,jacobian=True):
    if count not in (2,3):raise ValueError('D6 Euler coordinates require two or three angular axes')
    a0,a1=axes[...,0,:],axes[...,1,:]
    if count==2:
        a2=a1;a1=torch.linalg.cross(a0,a2)
    else:a2=axes[...,2,:]
    a2_rh=torch.linalg.cross(a0,a1)
    basis=torch.stack((a0,a1,a2_rh),-1)
    rotation=so3.matrix(so3.multiply(so3.conjugate(parent),child))
    local=basis.transpose(-1,-2)@rotation@basis
    angles=torch.stack((torch.atan2(-local[...,1,2],local[...,2,2]),torch.asin(local[...,0,2].clamp(-1,1)),torch.atan2(-local[...,0,1],local[...,0,0])),-1)
    angles=torch.remainder(angles+torch.pi,2*torch.pi)-torch.pi
    handed=torch.where((a2_rh*a2).sum(-1)<0,-1.,1.)
    coords=angles*torch.stack((torch.ones_like(handed),torch.ones_like(handed),handed),-1)
    first=so3.exp(a0*coords[...,:1]);transported_1=so3.rotate(first,a1)
    second=so3.exp(transported_1*coords[...,1:2]);transported_2=so3.rotate(so3.multiply(second,first),a2)
    world0=so3.rotate(parent,a0);world1=so3.rotate(parent,transported_1);world2=so3.rotate(parent,transported_2)
    if count==2:
        coords=coords[...,[0,2,1]];world1,world2=world2,world1
    determinant=(world0*torch.linalg.cross(world1,world2)).sum(-1)
    if not jacobian:return coords,None,determinant.abs()>=1e-4
    safe=torch.where(determinant<0,-1.,1.)*determinant.abs().clamp_min(1e-4)
    covector=torch.stack((torch.linalg.cross(world1,world2),torch.linalg.cross(world2,world0),torch.linalg.cross(world0,world1)),-2)/safe[...,None,None]
    return coords,covector,determinant.abs()>=1e-4


def anchor_coordinates(position,rotation,com,frames,*,jacobian=True):
    """Parent-frame separation and its exact two-body COM-pose Jacobian.

    Differentiating the parent frame is essential for prismatic/D6 joints.
    It also makes the damping coordinate objective under common rotation.
    """
    anchors=position+so3.rotate(rotation,frames[...,:3]-com)
    parent_q=so3.multiply(rotation[...,0,:],frames[...,0,3:])
    parent_r=so3.matrix(parent_q);inverse=parent_r.transpose(-1,-2)
    delta=anchors[...,1,:]-anchors[...,0,:]
    value=(inverse@delta[...,None])[...,0]
    if not jacobian:return value,None,anchors,parent_q
    identity=torch.eye(3,device=value.device,dtype=value.dtype).expand(*value.shape[:-1],3,3)
    parent_arm=anchors[...,1,:]-position[...,0,:]
    child_arm=anchors[...,1,:]-position[...,1,:]
    jp=inverse@torch.cat((-identity,skew(parent_arm)),-1)
    jc=inverse@torch.cat((identity,-skew(child_arm)),-1)
    return value,torch.stack((jp,jc),-2),anchors,parent_q


def unwrap_near(value,previous):
    """Single angular DOF only; caller admits less than pi relative motion."""
    turns=torch.round((previous-value)/(2*torch.pi)).detach()
    return value+2*torch.pi*turns


def drive_limit_force(q,rate,target,target_rate,drive_k,drive_d,lower,upper,limit_k,limit_d,dt,*,support=None,bound_override=None):
    """Finite-material constitutive fixed point of separate drive/limit rows.

    Drives clamp their position target into existing limits. A pure drive
    damper remains active at K=0. Unilateral limits clamp the *combined*
    spring/damping force at the nearest authored bound, matching the source
    compliant law rather than its legacy mutually exclusive drive/limit path.
    """
    has_limit=(limit_k>0)&((lower>-1e10)|(upper<1e10))
    target=torch.where(has_limit,torch.minimum(torch.maximum(target,lower),upper),target)
    drive=drive_k*(q-target)+drive_d*(rate-target_rate)
    drive_weight=drive_k+drive_d/dt
    choose_lower=torch.where(lower<=-1e10,False,torch.where(upper>=1e10,True,q-lower<=upper-q))
    if bound_override is not None:choose_lower=bound_override
    bound=torch.where(choose_lower,lower,upper)
    # Infinite absent bounds must not create 0*inf or poison autograd.
    bound=torch.where(has_limit,bound,q.detach())
    raw=limit_k*(q-bound)+limit_d*rate
    limit=torch.where(choose_lower,raw.clamp_max(0),raw.clamp_min(0))
    limit=torch.where(has_limit,limit,0.)
    limit_weight=(limit_k+limit_d/dt)*has_limit*((choose_lower&(raw<0))|(~choose_lower&(raw>0)))
    active=torch.ones_like(has_limit) if support is None else support>0
    return {'drive_force':torch.where(active,drive,0.),'limit_force':torch.where(active,limit,0.),'drive_weight':torch.where(active,drive_weight,0.),'limit_weight':torch.where(active,limit_weight,0.),'bound_lower':choose_lower,'has_limit':has_limit}
