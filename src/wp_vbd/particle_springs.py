"""Distance spring material rows for the future coupled particle adapter.

This standalone constitutive operator is not a scene/training admission.
The previous accepted endpoint distance defines the damping anchor, matching
Newton VBD's spring force. The frozen solve metric retains only positive
curvature; the physical force and incremental potential are unchanged.
"""
import math

import torch


def evaluate(position,anchor,rest_length,stiffness,damping,dt):
    if position.ndim!=4 or position.shape[-2:]!=(2,3) or anchor.shape!=position.shape:
        raise ValueError('Current and accepted-anchor spring endpoints require [batch,rows,2,3]')
    if not position.is_floating_point() or not math.isfinite(dt) or dt<=0:
        raise ValueError('Floating physical positions and a positive timestep required')
    rows=position.shape[1]
    for value in (anchor,rest_length,stiffness,damping):
        if value.device!=position.device or value.dtype!=position.dtype or not bool(torch.isfinite(value).all()):
            raise ValueError('Finite spring data on one physical dtype/device required')
    if not bool(torch.isfinite(position).all()):raise ValueError('Nonfinite spring endpoints')
    for value in (rest_length,stiffness,damping):
        if value.shape!=(rows,) or bool((value<0).any()):raise ValueError('Nonnegative per-spring material required')
    delta=position[:,:,0]-position[:,:,1];previous=anchor[:,:,0]-anchor[:,:,1]
    length=delta.norm(dim=-1);old_length=previous.norm(dim=-1)
    active=(stiffness+damping)>0
    if bool((active[None]&((length<=1e-8)|(old_length<=1e-8))).any()):
        raise ValueError('Active distance springs require noncollapsed current and accepted-anchor geometry')
    normal=delta/length.clamp_min(1e-8)[...,None]
    rate=damping/dt
    scalar=stiffness*(length-rest_length)+rate*(length-old_length)
    force0=-scalar[...,None]*normal
    force=torch.stack((force0,-force0),2)
    energy=.5*stiffness*(length-rest_length).square()+.5*rate*(length-old_length).square()
    axial=(stiffness+rate)[None].expand_as(length)
    transverse=scalar/length.clamp_min(1e-8)
    identity=torch.eye(3,device=position.device,dtype=position.dtype)
    outer=normal[:,:,:,None]*normal[:,:,None,:]
    hessian=transverse[:,:,None,None]*identity+(axial-transverse)[:,:,None,None]*outer
    # Closed-form spectral factor: positive radial and transverse channels.
    # No negative-curvature force is removed; only the solve metric is PSD.
    tangential=transverse.detach().clamp_min(0).sqrt()
    radial=axial.detach().sqrt()
    factor=tangential[:,:,None,None]*identity+(radial-tangential)[:,:,None,None]*outer.detach()
    pair=torch.stack((factor,-factor),3)
    jacobian=torch.nn.functional.pad(pair,(0,3))
    valid=torch.isfinite(force).all((-1,-2))&torch.isfinite(energy)&torch.isfinite(hessian).all((-1,-2))&torch.isfinite(jacobian).all((-1,-2,-3))
    return {'force':force,'energy':energy,'length':length,'anchor_length':old_length,
            'exact_relative_hessian':hessian,'metric_jacobian':jacobian,
            'metric_weight':torch.ones_like(length)[...,None].expand(*length.shape,3),
            'valid':valid}
