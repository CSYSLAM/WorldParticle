"""Frozen proposal bound for the remaining collision-query motion envelope."""
import math

import torch


def proposal_scale(scene,start,current,update,*,safety=.99):
    """One scalar per batch for an affine translation / Exp rotation ray.

    The oracle measures translation plus radius times shortest rotation.
    Its value at the current iterate plus t*(|dx|+r*|dtheta|) bounds every
    point along the unprojected ray. Retractions and existing motion/path
    checks still decide acceptance. This neither expands the query nor
    changes its budget.
    """
    if update.ndim!=3 or update.shape!=(*current.x.shape[:2],6):
        raise ValueError('Complete batched six-slot physical proposal required')
    if not math.isfinite(safety) or not 0<safety<1:
        raise ValueError('A strict remaining-motion safety fraction is required')
    budget=scene.geometry_oracle.motion_budget
    if not math.isfinite(budget) or budget<=0:
        raise ValueError('Finite positive collision-query motion budget required')
    radius=scene.body_radius
    if getattr(scene,'compiled_response_kernels',False) and not torch.is_grad_enabled() and update.is_cuda:
        if radius.shape!=(scene.n,):raise ValueError('Finite nonnegative physical surface radii required')
        from .compiled_contact import evaluate as compiled
        current_valid,used=scene.geometry_oracle.motion(start,current)
        cap,valid,stats,radius_valid=compiled(_cap_tensor,update,scene.dof,radius,scene.movable,
                                            current_valid,used,budget,safety)
        if not bool(radius_valid):raise ValueError('Finite nonnegative physical surface radii required')
        # Diagnostic tensors are serialized by the existing recorder. They
        # never influence acceptance; keep them on device inside iterations.
        detail={'scale':cap.detach(),'valid':valid.detach(),'motion_budget':budget,'safety':safety,
                'maximum_current_surface_travel':stats[:,0].detach(),
                'maximum_bounded_surface_travel':stats[:,1].detach()}
        return cap,valid,detail
    if radius.shape!=(scene.n,) or not bool(torch.isfinite(radius).all()) or bool((radius<0).any()):
        raise ValueError('Finite nonnegative physical surface radii required')
    with torch.no_grad():
        valid,used=scene.geometry_oracle.motion(start,current)
        finite=torch.isfinite(update).all((1,2))&torch.isfinite(used).all(-1)
        valid=valid&finite
        safe=torch.where(torch.isfinite(update),update.detach(),0.)*scene.dof
        rate=safe[...,:3].norm(dim=-1)+radius*safe[...,3:].norm(dim=-1)
        remaining=(budget-used).clamp_min(0.)
        ratio=safety*remaining/rate.clamp_min(torch.finfo(update.dtype).tiny)
        allowed=torch.where((rate>0)&scene.movable[None],ratio,1.).clamp(0.,1.)
        cap=allowed.amin(-1)
        cap=torch.where(valid,cap,0.)
        bound=used+cap[:,None]*rate
        bounded=(bound[:,scene.movable]<=budget).all(-1)
        valid=valid&bounded
        cap=torch.where(valid,cap,0.)
        detail={'scale':cap.cpu().tolist(),'valid':valid.cpu().tolist(),
                'motion_budget':budget,'safety':safety,
                'maximum_current_surface_travel':used[:,scene.movable].amax(-1).cpu().tolist() if bool(scene.movable.any()) else [0.]*len(update),
                'maximum_bounded_surface_travel':bound[:,scene.movable].amax(-1).cpu().tolist() if bool(scene.movable.any()) else [0.]*len(update)}
    return cap,valid,detail


def _cap_tensor(update,dof,radius,movable,current_valid,used,budget,safety):
    radius_valid=torch.isfinite(radius).all()&~(radius<0).any()
    finite=torch.isfinite(update).all((1,2))&torch.isfinite(used).all(-1)
    valid=current_valid&finite
    safe=torch.where(torch.isfinite(update),update.detach(),0.)*dof
    rate=safe[...,:3].norm(dim=-1)+radius*safe[...,3:].norm(dim=-1)
    remaining=(budget-used).clamp_min(0.)
    ratio=safety*remaining/rate.clamp_min(torch.finfo(update.dtype).tiny)
    allowed=torch.where((rate>0)&movable[None],ratio,1.).clamp(0.,1.)
    cap=allowed.amin(-1);cap=torch.where(valid,cap,0.)
    bound=used+cap[:,None]*rate
    bounded=((bound<=budget)|~movable[None]).all(-1)
    valid=valid&bounded;cap=torch.where(valid,cap,0.)
    stats=torch.stack((torch.where(movable[None],used,0.).amax(-1),
                       torch.where(movable[None],bound,0.).amax(-1)),-1)
    return cap,valid,stats,radius_valid
