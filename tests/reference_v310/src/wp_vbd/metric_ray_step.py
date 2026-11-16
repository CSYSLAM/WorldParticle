"""Frozen physical curvature scaling of an existing contact proposal.

Each interaction component uses min(1, -R.d / d.H.d). This changes the
proposal, not the force, metric, iteration count or acceptance predicate.
The coefficient is a frozen physical decision in backward, like the trust
radius and line-search selection. It is not an exact nonlinear line solve.
"""
import torch

from .protected_response import metric_action


def scale_proposal(scene, metric, residual, update):
    if residual.shape!=update.shape or update.ndim!=3 or update.shape[1:]!=(scene.n,6):
        raise ValueError('Matched batch of physical six-slot residuals and proposals required')
    with torch.no_grad():
        image=metric_action(metric,update)
        scales=torch.ones_like(update[...,0])
        valid=torch.ones(len(update),device=update.device,dtype=torch.bool)
        details=[]
        for ids in scene.interaction_entities:
            numerator=-(residual[:,ids]*update[:,ids]).sum((1,2))
            denominator=(update[:,ids]*image[:,ids]).sum((1,2))
            good=torch.isfinite(numerator)&torch.isfinite(denominator)&(numerator>=0)&(denominator>=0)
            ratio=numerator/denominator.clamp_min(torch.finfo(update.dtype).tiny)
            scale=torch.where(denominator>0,ratio,1.).clamp(0.,1.)
            scale=torch.where(good,scale,0.)
            scales[:,ids]=scale[:,None];valid&=good
            details.append({'owners':len(ids),'linear_descent':numerator.cpu().tolist(),
                            'metric_curvature':denominator.cpu().tolist(),'scale':scale.cpu().tolist(),
                            'valid':good.cpu().tolist()})
    return update*scales[...,None],valid,details
