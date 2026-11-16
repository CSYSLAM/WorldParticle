"""Protect finite-budget accuracy where the physical coarse space is complete."""
import torch


def complete_mask(scene,coarse,gain):
    if gain.ndim!=2 or gain.shape[1]!=scene.n:raise ValueError('Per-entity coarse gains required')
    if getattr(coarse,'packed',False):
        mask=coarse.complete[coarse.layout.labels][None].expand_as(gain)
    else:
        if len(coarse.ranks)!=len(scene.interaction_entities):raise ValueError('Matched coarse ranks and physical components required')
        mask=torch.zeros_like(gain,dtype=torch.bool)
        for ids,rank in zip(scene.interaction_entities,coarse.ranks):
            complete=rank>=scene.dof[ids].sum()
            mask=mask.index_copy(1,ids,complete[:,None].expand(-1,len(ids)))
    return mask&scene.movable[None]


def coarse_gain(module,scene,latent,relation,coarse):
    """Ask the same head for bounded complete-space and legacy coupling gains.

    With the current0.8 relaxation, a complete frozen quadratic block gets
    an effective0.784..0.8 Newton step. Four rounds leave at most0.216**4 of its
    initial linear residual. Gains stay below the metric-ray cancellation
    threshold so the head retains a real forward derivative. Incomplete spaces retain the learned
    response. This is not a nonlinear/contact convergence guarantee.
    """
    head=module.coarse_response
    if getattr(head,'protect_complete_response',False):
        return head(scene,latent,relation,complete_mask=complete_mask(scene,coarse,latent[...,0]))
    return head(scene,latent,relation)
