"""Objective finite-material joint rows, independent of a global solver.

COM poses use world angular increments. Structural, drive and unilateral
limit rows share exact coordinate covectors and positive GN weights. Finite
ALM multipliers are eliminated at their physical constitutive fixed point;
single-axis winding remains genuine cross-step state owned by the caller.
"""
from types import SimpleNamespace
import torch
from . import math3d as so3
from .joint_coordinates import anchor_coordinates,log_coordinates,d6_angular_coordinates,drive_limit_force,unwrap_near


def coordinates(spec,position,rotation,com,*,previous_angle=None,jacobian=True):
    linear,jlinear,anchors,parent=anchor_coordinates(position,rotation,com,spec.frames,jacobian=jacobian)
    child=so3.multiply(rotation[...,1,:],spec.frames[...,1,3:])
    angular,jangular=log_coordinates(parent,child,spec.rest_parent,spec.rest_child,jacobian=jacobian)
    valid=torch.ones_like(linear[...,0],dtype=torch.bool)
    for count in (2,3):
        ids=torch.where(spec.angular_mode==count)[0]
        if len(ids):
            value,j,good=d6_angular_coordinates(spec.angular_axes[ids],count,parent[:,ids],child[:,ids],jacobian=jacobian)
            basis=spec.angular_axes[ids].clone()
            if count==2:basis[:,2]=torch.linalg.cross(basis[:,0],basis[:,1])
            angular=angular.index_copy(1,ids,torch.einsum('jic,bji->bjc',basis,value))
            if jacobian:jangular=jangular.index_copy(1,ids,torch.einsum('jic,bjik->bjck',basis,j))
            valid=valid.index_copy(1,ids,good)
    if jacobian:
        zero=torch.zeros_like(jangular)
        ja=torch.stack((torch.cat((zero,-jangular),-1),torch.cat((zero,jangular),-1)),-2)
    structural=torch.cat((torch.einsum('jcd,bjd->bjc',spec.linear_projector,linear),torch.einsum('jcd,bjd->bjc',spec.angular_projector,angular)),-1)
    structural_j=torch.cat((torch.einsum('jcd,bjdsi->bjcsi',spec.linear_projector,jlinear),torch.einsum('jcd,bjdsi->bjcsi',spec.angular_projector,ja)),2) if jacobian else None
    linear_axis=torch.einsum('jcd,bjd->bjc',spec.linear_axes,linear)+spec.coordinate_offset[...,:3]
    angular_axis=torch.einsum('jcd,bjd->bjc',spec.angular_axes,angular)+spec.coordinate_offset[...,3:]
    if previous_angle is not None:
        winding=spec.angular_mode==1
        angle=unwrap_near(angular_axis[...,0],previous_angle)
        angular_axis=torch.cat((torch.where(winding,angle,angular_axis[...,0])[...,None],angular_axis[...,1:]),-1)
    axis=torch.cat((linear_axis,angular_axis),-1)
    axis_j=torch.cat((torch.einsum('jcd,bjdsi->bjcsi',spec.linear_axes,jlinear),torch.einsum('jcd,bjdsi->bjcsi',spec.angular_axes,ja)),2) if jacobian else None
    return SimpleNamespace(structural=structural,structural_jacobian=structural_j,axis=axis,axis_jacobian=axis_j,anchors=anchors,valid=valid)


def evaluate(spec,current,start,dt,target,target_rate,*,frozen_bound=None,derivatives=True):
    """Return residual forces, sparse GN rows and a matched frozen-row merit.

    `current` and `start` are coordinate snapshots, so line searches can keep
    the step-start damping and the selected unilateral bound fixed. Target
    and enabled values are explicitly passed/read each physical step.
    """
    enabled=spec.enabled.to(current.axis.dtype)[None,:,None]
    if derivatives:
        if current.structural_jacobian is None or current.axis_jacobian is None:raise ValueError('Force rows require coordinate Jacobians')
        structural_force=spec.stiffness*current.structural+spec.damping*(current.structural-start.structural)/dt
        structural_weight=spec.stiffness+spec.damping/dt
    rate=(current.axis-start.axis)/dt
    response=drive_limit_force(current.axis,rate,target,target_rate,spec.drive_k,spec.drive_d,spec.lower,spec.upper,spec.limit_k,spec.limit_d,dt,bound_override=frozen_bound)
    weight=jacobian=pair_residual=None
    if derivatives:
        force=torch.cat((structural_force,response['drive_force'],response['limit_force']),-1)*enabled
        weight=torch.cat((structural_weight.expand_as(current.structural),response['drive_weight'].expand_as(current.axis),response['limit_weight'].expand_as(current.axis)),-1)*enabled
        jacobian=torch.cat((current.structural_jacobian,current.axis_jacobian,current.axis_jacobian),2)
        pair_residual=torch.einsum('bjcsi,bjc->bjsi',jacobian,force)
    # Keep absent infinite bounds out of all differentiable arithmetic.
    clamped_target=torch.where(response['has_limit'],torch.minimum(torch.maximum(target,spec.lower),spec.upper),target)
    bound=torch.where(response['bound_lower'],spec.lower,spec.upper)
    bound=torch.where(response['has_limit'],bound,current.axis.detach())
    raw=spec.limit_k*(current.axis-bound)+spec.limit_d*rate
    active_force=torch.where(response['bound_lower'],raw.clamp_max(0),raw.clamp_min(0))
    limit_energy=.5*active_force.square()/(spec.limit_k+spec.limit_d/dt).clamp_min(1e-30)*response['has_limit']
    energy=(.5*spec.stiffness*current.structural.square()+.5*spec.damping/dt*(current.structural-start.structural).square()).sum(-1)
    energy=energy+(.5*spec.drive_k*(current.axis-clamped_target).square()+.5*spec.drive_d/dt*(current.axis-start.axis-dt*target_rate).square()+limit_energy).sum(-1)
    return {'pair_residual':pair_residual,'jacobian':jacobian,'weight':weight,'energy':energy*spec.enabled,'bound_lower':response['bound_lower'],'valid':(current.valid&start.valid)|~spec.enabled,'coordinates':current}
