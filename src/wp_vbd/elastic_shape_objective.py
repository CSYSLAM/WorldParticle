"""Training/audit objectives for a learned one-column elastic subspace.

The full 24-column operator is used only to form frozen training labels or
diagnostics. Inference needs curvature actions only on the predicted columns.
"""

import torch

from .elastic_shape_modes import _pseudoinverse_solve
from .protected_response import metric_action


@torch.no_grad()
def projected_modal_systems(scene, metric, coarse, effective_residual, details, *, joint=False):
    """Reduce the actual remaining quadratic in the deployed projected basis.

    By default each record evaluates one object in one interaction. With
    joint=True, concatenate all interacting objects' spaces and retain the
    cross terms, matching the deployed combined ray's independent scalar.
    """
    systems = []; grouped = {}
    for item in details:
        basis = item['basis'].double()
        local = torch.nn.functional.pad(basis, (0, 0, 0, 3))
        full = local.new_zeros(len(local), scene.n, 6, basis.shape[-1])
        full = full.index_copy(1, item['ids'], local)
        full = full * scene.dof[None, :, :, None]
        projected = full - coarse(metric_action(metric, full))
        image = metric_action(metric, projected)
        for component, ids in enumerate(scene.interaction_entities):
            value, hv = projected[:, ids], image[:, ids]
            gram = torch.einsum('bnip,bniq->bpq', value, hv)
            gram = (gram + gram.transpose(-1, -2)) * .5
            if not bool((gram.diagonal(dim1=-2, dim2=-1).sum(-1) > 0).any()):
                continue
            if joint:
                grouped.setdefault(component, []).append((value, hv, item['object_id']))
                continue
            rhs = torch.einsum('bnip,bni->bp', value, effective_residual[:, ids].double())
            optimum = _pseudoinverse_solve(gram, rhs)
            work = (rhs * optimum).sum(-1).clamp_min(0)
            systems.append({'object_id': item['object_id'], 'component': component,
                            'gram': gram, 'rhs': rhs, 'optimum_coefficients': optimum,
                            'optimal_work': work})
    if joint:
        for component, candidates in grouped.items():
            ids = scene.interaction_entities[component]
            value = torch.cat([v for v, hv, _ in candidates], -1)
            image = torch.cat([hv for v, hv, _ in candidates], -1)
            gram = torch.einsum('bnip,bniq->bpq', value, image)
            gram = (gram + gram.transpose(-1, -2)) * .5
            rhs = torch.einsum('bnip,bni->bp', value, effective_residual[:, ids].double())
            optimum = _pseudoinverse_solve(gram, rhs)
            systems.append({'component': component, 'gram': gram, 'rhs': rhs,
                            'object_ids': [object_id for _, _, object_id in candidates],
                            'optimum_coefficients': optimum,
                            'optimal_work': (rhs * optimum).sum(-1).clamp_min(0)})
    return systems


def directional_energy_recovery(coefficients, system):
    """Fraction of attainable quadratic energy decrease, invariant to ±scale.

    For E(u)=r.u+u.Hu/2 and d=Bc, an optimal scalar step recovers
    (c.g)^2 / (c.G.c), where g=B^T r and G=B^T H B.
    The best column in the whole basis recovers g^T G^+ g. Both are twice
    the corresponding energy decrease. Empty/zero-load systems carry no label.
    """
    c = coefficients.double()
    gram, rhs, optimum = system['gram'], system['rhs'], system['optimal_work']
    curvature = torch.einsum('bp,bpq,bq->b', c, gram, c)
    dot = (c * rhs).sum(-1)
    # Relative rank handling must not introduce dependence on direction scale.
    scale = gram.diagonal(dim1=-2, dim2=-1).abs().sum(-1) * c.square().sum(-1)
    active = (optimum > 1e-30) & torch.isfinite(optimum)
    usable = active & (curvature > scale * 1e-12) & torch.isfinite(curvature)
    denominator = torch.where(usable, curvature * optimum, torch.ones_like(curvature))
    recovery = torch.where(usable, dot.square() / denominator, torch.zeros_like(dot))
    return recovery, active


def ray_energy_recovery(direction, image, residual, optimal_work):
    """Evaluate the actual combined deployed ray, retaining object cross terms."""
    dot = (direction * residual).sum((1, 2))
    curvature = (direction * image).sum((1, 2))
    active = (optimal_work > 1e-30) & torch.isfinite(optimal_work)
    usable = active & (curvature > 0) & torch.isfinite(curvature)
    denominator = torch.where(usable, curvature * optimal_work, torch.ones_like(curvature))
    return torch.where(usable, dot.square() / denominator, torch.zeros_like(dot)), active


def physical_direction_loss(coefficients, system):
    """Loss on the deployable ray response, with frozen physics labels."""
    recovery, active = directional_energy_recovery(coefficients, system)
    return ((1 - recovery) * active).sum() / active.sum().clamp_min(1)
