"""Positive contact anticipation for two-to-four-owner physical rows."""
import torch

from . import math3d as so3
from .contact_anticipation import secant_weight
from .data import State
from .mixed_metric import MetricTerm


def _metric_term(candidates, jacobian, weight, *, compact):
    ids = candidates.owners.clamp_min(0)
    if not compact:
        return MetricTerm(ids, jacobian[:, :, None], weight[..., None])
    if weight.requires_grad or jacobian.requires_grad:
        raise ValueError('Anticipation compaction requires frozen rows and weights')
    # Only the solve metric is compacted. All geometry, compliance, forecast
    # activation and topology tests above this call still use every candidate.
    selected = torch.where((weight != 0).any(0))[0]
    cache = getattr(candidates, '_anticipation_metric_selection', None)
    if cache is None or not torch.equal(cache[0], selected):
        cache = (selected.detach().clone(), object(), ids[selected])
        candidates._anticipation_metric_selection = cache
    selected, topology_key, ids = cache
    return MetricTerm(ids, jacobian[:, selected, None], weight[:, selected, None], topology_key)


def _normal_compliance(jacobian, inverse_blocks, ids, *, maximum_rows=16384):
    """Evaluate every sparse row without gathering all inverse blocks at once."""
    if maximum_rows <= 0:
        raise ValueError('A positive compliance chunk size is required')
    if not len(ids):
        return jacobian.new_zeros(jacobian.shape[:2])
    # The batch dimension also contributes to the temporary gather. Each
    # chunk has the original per-row contraction and complete owner slots.
    chunk = max(1, maximum_rows//max(1, len(jacobian)))
    values = []
    for first in range(0, len(ids), chunk):
        row = jacobian[:, first:first+chunk]
        blocks = inverse_blocks[:, ids[first:first+chunk]]
        values.append(torch.einsum('brsi,brsij,brsj->br', row, blocks, row))
    return torch.cat(values, 1)


@torch.no_grad()
def metric(scene, state, forecast, base, candidates, result, residual, *, force_aware_material=False, dt=None):
    if not len(candidates.owners):
        return [], result['active'], torch.ones(len(state.x), device=scene.device, dtype=torch.bool)
    local = base.diagonal.double()
    ghost = torch.diag_embed((~scene.dof).double())[None]
    inverse = torch.linalg.inv_ex(local+ghost, check_errors=False)
    value = inverse.inverse*scene.dof[None, :, :, None]*scene.dof[None, :, None, :]
    jacobian = candidates.normal_jacobian(scene, state)
    ids = candidates.owners.clamp_min(0)
    compliance = _normal_compliance(jacobian, value, ids)
    support = torch.where(compliance > 0, compliance.clamp_min(1e-300).reciprocal(), 0.)
    target = candidates.depth(scene, forecast)
    delta = -(value@residual.double()[..., None])[..., 0]
    travel = delta[..., :3].norm(dim=-1)+scene.body_radius*delta[..., 3:].norm(dim=-1)
    bound = scene.geometry_oracle.motion_budget
    delta = delta*(bound/travel.clamp_min(bound)).clamp_max(1)[..., None]
    q = so3.normalize(so3.multiply(so3.exp(delta[..., 3:]), state.q))
    local_state = State(state.x+delta[..., :3], q, state.v, state.omega)
    local_target = candidates.depth(scene, local_state)
    target = torch.maximum(target, local_target)
    labels = scene.material_object_index; count = len(scene.material_object_keys)
    maximum = travel.new_zeros(len(travel), count).scatter_reduce(1, labels[None].expand_as(travel), delta[..., :3].norm(dim=-1), reduce='amax', include_self=True)
    drive = delta.new_zeros(len(travel), count, 3).index_add(1, labels, -residual[..., :3])
    if force_aware_material:
        if dt is None or dt<=0:raise ValueError('A physical step interval is required for the material force forecast')
        mass=state.x.new_zeros(count).index_add(0,labels,scene.mass*scene.movable)
        shift=dt**2*drive/mass[None,:,None].clamp_min(1e-30)
        shift=shift*(bound/shift.norm(dim=-1,keepdim=True).clamp_min(bound)).clamp_max(1)
    else:shift = drive/drive.norm(dim=-1, keepdim=True).clamp_min(1e-30)*maximum[..., None]
    object_x = state.x+shift[:, labels]*scene.movable[None, :, None]
    object_x = torch.where((scene.kind < 2)[None, :, None], object_x, local_state.x)
    object_target=candidates.depth(scene, State(object_x, q, state.v, state.omega))
    target = torch.maximum(target, object_target)
    if force_aware_material:
        # A free-flight forecast ignores an established frictional lifting
        # force. Do not use it to keep an opening ground contact artificially
        # stiff. Actual force rows and proposal-activation checks stay intact.
        material_rows=candidates.source_kind==1
        target=torch.where(material_rows[None],torch.maximum(local_target,object_target),target)
    current = result['raw_depth']; gap = (-current).clamp_min(0); closing = (target-current).clamp_min(0)
    weight = secant_weight(candidates.material[None], support, gap, closing)
    weight = torch.where((current < 0) & (target > 0), weight, 0.)
    valid = (inverse.info == 0).all(-1) & torch.isfinite(weight).all(-1) & torch.isfinite(value).flatten(1).all(-1)
    topology = result['active'] | (weight > 0)
    term = _metric_term(candidates, jacobian, weight,
                        compact=bool(getattr(scene, 'compact_frozen_contact_metric', False)))
    return [term], topology, valid
