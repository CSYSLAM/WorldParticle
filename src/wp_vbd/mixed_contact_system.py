"""One elastic/contact residual and one protected shared neural response.

The contact loop retains its fixed K primal/dual budget. Material elasticity
and damping enter the same sparse positive metric; inertia is counted once.
"""
import torch

from .data import State
from .feasibility import tetrahedron_prefix_minimum
from .mixed_metric import FrozenMixedMetric
from .mixed_physics import incremental_objective, residual_system
from .mixed_solver import relation_inputs
from .unified_graph import build_topology
from . import rigid_contact_system as contact

supports_rigid_component_batch = False
HISTORY_KEYS = {'pin_x_prev', 'pin_initialized'}
rows = contact.rows
coarse = contact.coarse


def validate_history(scene, start):
    if start.history is None or set(start.history) != HISTORY_KEYS:
        raise ValueError('Complete accepted pin-position history is required')
    previous = start.history['pin_x_prev']; initialized = start.history['pin_initialized']
    if previous.shape != (len(start.x), len(scene.fixed_particle_ids), 3) or initialized.shape != (len(start.x),) or initialized.dtype != torch.bool or not bool(torch.isfinite(previous).all()):
        raise ValueError('Invalid pin-position history')


def begin(scene, start, candidates, dt):
    validate_history(scene, start)
    context = contact.begin(scene, start, candidates, dt)
    ids = scene.fixed_particle_ids
    previous = torch.where(start.history['pin_initialized'][:, None, None], start.history['pin_x_prev'], start.x[:, ids])
    context.material_start = State(start.x.index_copy(1, ids, previous), start.q, start.v, start.omega)
    context.pin_velocity = (start.x[:, ids] - previous) / dt
    return context


def system(scene, state, start, tx, tq, context):
    material_start = context.material_start
    assembled = residual_system(scene, state, material_start, tx, tq, context.dt, return_metric_base=True)
    residual, _, extras, base = assembled
    jacobian, _ = relation_inputs(scene.elastic_scene, state, material_start, extras, context.dt, torch.ones_like(state.x[..., 0]))
    elastic = FrozenMixedMetric(scene, state, material_start, tx, tq, context.dt, extras, jacobian, base=base, elastic_mode='objective')
    result = rows(scene, state, start, context)
    anticipation=[]
    if getattr(scene,'predictive_contact_metric',False):
        from .contact_anticipation import metric as prepare
        anticipation,topology,valid=prepare(scene,state,State(tx,tq,start.v,start.omega),elastic,context.contacts,result,residual)
        result['topology_active']=topology;result['valid']=torch.cat((result['valid'],valid[:,None]),-1)
        context.anticipated_contacts=int((topology&~result['active']).sum())
    metric = contact.ContactMetric(elastic.local, [*elastic.terms, *result['terms'],*anticipation], scene.dof)
    result['elastic_extras'] = extras
    return (residual + result['residual']) * scene.dof, metric, result


def relation_features(scene, state, start, context, result, feedback):
    _, structural = relation_inputs(scene.elastic_scene, state, context.material_start, result['elastic_extras'], context.dt, feedback)
    surface = contact.relation_features(scene, state, start, context, result, feedback)
    return torch.cat((structural, surface), 1)


def merit(scene, state, start, tx, tq, context, frozen):
    empty = state.x.new_empty(len(state.x), 0)
    elastic = incremental_objective(scene, state, context.material_start, tx, tq, context.dt, empty)
    return elastic + contact.merit(scene, state, start, tx, tq, context, frozen, include_inertia=False)


def acceptance_groups(scene, candidates, depth, travel, *, labels=False, metric_active=None):
    possible = (depth[0] + travel[0, candidates.owners].sum(-1) >= 0) & (candidates.material > 0)
    # A positive anticipatory metric couples two endpoints even before they
    # touch. Its protected response guarantees descent for that component;
    # splitting it during selection can turn a descent into a partial ascent.
    if metric_active is not None:possible = possible | metric_active[0]
    owners, kinds, enabled = scene.all_relations(candidates.owners, possible)
    graph = build_topology(scene.kind, scene.movable, scene.object_id, owners, kinds, potential_active=enabled)
    if labels: return graph.interaction_id
    return [torch.where(graph.interaction_id == label)[0] for label in torch.unique(graph.interaction_id).tolist() if bool(scene.movable[graph.interaction_id == label].any())]


def path_feasible(scene, start, end):
    """Guard the entire linear particle path, including an interior collapse.

    Tet determinant is cubic. For a triangle the cross product is quadratic;
    its projection on the initial normal has an analytic interval minimum.
    A positive projected area excludes both collapse and a within-step flip.
    This guards local orientation, not cloth self-contact or tunnelling.
    """
    valid = torch.ones(len(start.x), device=scene.device, dtype=torch.bool)
    if len(scene.tets):
        valid &= tetrahedron_prefix_minimum(scene, start.x, end.x, start.x.new_ones(len(start.x))) >= scene.tet_floor
    if len(scene.triangles):
        old = start.x[:, scene.triangles]; new = end.x[:, scene.triangles]
        a = old[:, :, 1] - old[:, :, 0]; b = old[:, :, 2] - old[:, :, 0]
        da = new[:, :, 1] - new[:, :, 0] - a; db = new[:, :, 2] - new[:, :, 0] - b
        cross = torch.linalg.cross(a, b); area = cross.norm(dim=-1)
        normal = cross / area.clamp_min(1e-30)[..., None]
        linear = ((torch.linalg.cross(da, b) + torch.linalg.cross(a, db)) * normal).sum(-1)
        quadratic = (torch.linalg.cross(da, db) * normal).sum(-1)
        critical = (-linear / torch.where(quadratic.abs() > 1e-30, 2 * quadratic, torch.ones_like(quadratic))).clamp(0, 1)
        critical = torch.where(quadratic > 0, critical, 0.)
        middle = area + linear * critical + quadratic * critical.square()
        minimum = torch.minimum(torch.minimum(area, area + linear + quadratic), middle)
        valid &= torch.isfinite(minimum).all(-1) & (minimum >= .01 * (2 * scene.tri_area)[None]).all(-1)
    return valid


def predictor_domain(scene, start, forecast, config, *, dt=None):
    validate_history(scene, start)
    ids = scene.fixed_particle_ids
    travel = (start.x[:, ids] - start.history['pin_x_prev']).norm(dim=-1)
    imposed = (travel <= scene.geometry_oracle.motion_budget).all(-1) | ~start.history['pin_initialized']
    if getattr(scene,'learned_contact_repair',False) and hasattr(scene.candidates,'self_snapshot'):
        # Check the joint material path after free vertices accommodate the
        # prescribed boundary command, rather than checking pins in isolation.
        return imposed
    return imposed & imposed_path_feasible(scene,start)


def imposed_path_feasible(scene,start):
    ids=scene.fixed_particle_ids
    previous=torch.where(start.history['pin_initialized'][:,None,None],start.history['pin_x_prev'],start.x[:,ids])
    baseline=State(start.x.index_copy(1,ids,previous),start.q,start.v,start.omega)
    return path_feasible(scene,baseline,start)


def tangent(scene, state): return None


def retract(scene, state):
    return state, torch.ones(len(state.x), device=scene.device, dtype=torch.bool)


def valid_state(scene, state): return path_feasible(scene, state, state)


def finish(scene, start, output, context, accepted):
    ids = scene.fixed_particle_ids
    history = {'pin_x_prev': torch.where(accepted[:, None, None], start.x[:, ids], start.history['pin_x_prev']), 'pin_initialized': start.history['pin_initialized'] | accepted}
    fixed_velocity = torch.where(accepted[:, None, None], context.pin_velocity, start.v[:, ids])
    velocity = torch.where(scene.movable[None, :, None], output.v, start.v).index_copy(1, ids, fixed_velocity)
    omega = torch.where((scene.movable & scene.body)[None, :, None], output.omega, start.omega)
    return State(output.x, output.q, velocity, omega, history)
