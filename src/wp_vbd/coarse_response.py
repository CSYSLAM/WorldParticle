"""Experimental component modes; not connected to the deployed checkpoint.

The relation-only neural branch cannot respond to residuals in ker(T). This
prototype tests a complementary, physically scaled six-mode response. Callers
must freeze one symmetric positive metric H throughout an application and
provide its matrix-free action, not the full nonlinear residual function.
"""

import torch

from .response_operator import skew


def rigid_component_modes(position, center, dof):
    """Six common translation/rotation modes in an entity's world tangent.

    position: [B,N,3], center: [B,1,3], dof: [N,6]. Particle rotation rows
    and all fixed rows are zero. These are low-frequency candidate modes, not
    a claim of exact null modes when pins, contact, or finite prestress exist.
    """
    shape = (*position.shape[:-1], 6, 6)
    modes = position.new_zeros(shape)
    eye = torch.eye(3, device=position.device, dtype=position.dtype)
    modes[..., :3, :3] = eye
    modes[..., :3, 3:] = -skew(position - center)
    modes[..., 3:, 3:] = eye
    return modes * dof[None, :, :, None]


def balanced_galerkin_response(residual, modes, local_apply, metric_apply):
    """Return P r with P=Q+(I-QH) B (I-HQ), Q=Z(Z^T H Z)^-1 Z^T.

    r is [B,N,D], Z is [B,N,D,q], 1<=q<=6 for ONE interaction component.
    B and H callbacks accept [B,N,D,columns] and return the same shape. The
    component must supply independent modes on its active DOFs: Cholesky
    rejects a singular coarse metric. No regularization silently repairs a
    physically invalid basis. With fixed symmetric H and SPD active B, P is
    symmetric positive and exact on H Z; r=0 remains exactly zero.

    Work: one H action on q modes, two more H actions on vectors, two coarse
    solves of size <=6, and one local B action. No dense global matrix or
    convergence loop is constructed. Scaling to many pools, contact history,
    learned coarse factors, and whole-step momentum remain research work.
    """
    if not 1 <= modes.shape[-1] <= 6:
        raise ValueError('A component coarse block must have one to six modes')
    h_modes = metric_apply(modes)
    gram = torch.einsum('bndp,bndq->bpq', modes, h_modes)
    cholesky = torch.linalg.cholesky(gram)

    def coarse(value):
        restricted = torch.einsum('bndp,bndq->bpq', modes, value)
        coefficient = torch.cholesky_solve(restricted, cholesky)
        return torch.einsum('bndp,bpq->bndq', modes, coefficient)

    coarse_r = coarse(residual[..., None])
    local = local_apply(residual[..., None] - metric_apply(coarse_r))
    return (coarse_r + local - coarse(metric_apply(local)))[..., 0]
