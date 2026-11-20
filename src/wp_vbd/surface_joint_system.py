"""One elastic, weighted-contact and finite-joint protected response."""
import math
from types import SimpleNamespace

import torch

from . import general_joint_system as joints, mixed_contact_system as pins
from . import rigid_contact_system as rigid, surface_contact_rows as surface
from .data import State
from .mixed_metric import FrozenMixedMetric, MetricTerm
from .mixed_physics import residual_system, incremental_objective
from .mixed_solver import relation_inputs
from .unified_graph import build_topology
from . import prescribed_motion as prescribed

supports_rigid_component_batch = False
contact_safety = 'BOUNDED_ENTITY_MOTION_AND_MATERIAL_ORIENTATION; FINITE_CONTACT_PENETRATION; NO_GENERAL_SURFACE_CCD_GUARANTEE'
HISTORY_KEYS = pins.HISTORY_KEYS | {'joint_angle_prev'}


def _tangent_history_enabled(scene):
    return bool(scene.metadata.get('surface_tangent_memory',False))
coarse = rigid.coarse
path_feasible = pins.path_feasible


def pin_state(state):
    return State(state.x, state.q, state.v, state.omega, {key: state.history[key] for key in pins.HISTORY_KEYS})


def validate_history(scene, start):
    expected=HISTORY_KEYS | (prescribed.HISTORY_KEYS if prescribed.enabled(scene) else set())
    if _tangent_history_enabled(scene):
        from .surface_tangent_memory import HISTORY_KEYS as tangent_keys
        expected |= tangent_keys
        if start.history is None:
            raise ValueError('Complete accepted pin and joint-angle history required')
        missing=tangent_keys-set(start.history)
        if missing:
            if missing != tangent_keys:
                raise ValueError('Partial surface tangent history is not admissible')
            from .surface_tangent_memory import initial_history
            start.history.update(initial_history(start,scene.metadata['surface_tangent_history_capacity']))
    if start.history is None or set(start.history) != expected:
        raise ValueError('Complete accepted pin and joint-angle history required')
    pins.validate_history(scene, pin_state(start))
    angle = start.history['joint_angle_prev']
    if angle.shape != (len(start.x), len(scene.joint_spec.owners)) or not bool(torch.isfinite(angle).all()):
        raise ValueError('Invalid joint winding history')
    if prescribed.enabled(scene):prescribed.previous(scene,start)


def inverse_mass(scene, state):
    from . import math3d as so3
    value = state.x.new_zeros(len(state.x), scene.n, 6, 6)
    mass = torch.where(scene.movable, scene.mass.clamp_min(1e-30).reciprocal(), 0.)
    value[..., :3, :3] = mass[None, :, None, None]*torch.eye(3, device=scene.device, dtype=state.x.dtype)
    rotation = so3.matrix(state.q)
    value[..., 3:, 3:] = rotation@scene.inverse_inertia@rotation.transpose(-1, -2)
    return value*scene.dof[None, :, :, None]*scene.dof[None, :, None, :]


def begin(scene, start, candidates, dt, *, material_start_override=None):
    validate_history(scene, start)
    ids = scene.fixed_particle_ids
    previous = torch.where(start.history['pin_initialized'][:, None, None], start.history['pin_x_prev'], start.x[:, ids])
    material_start = State(start.x.index_copy(1, ids, previous), start.q, start.v, start.omega)
    if prescribed.enabled(scene):
        material_start.history=start.history
        material_start=prescribed.previous(scene,material_start)
    if material_start_override is not None:
        if material_start_override.x.shape!=start.x.shape or material_start_override.q.shape!=start.q.shape:
            raise ValueError('Complete shared material-start pose layout required')
        material_start=material_start_override
    rc = rigid.begin(scene, start, candidates.rigid, dt); rc.material_start = material_start
    sc = surface.begin(candidates.soft, start, scene.body_com, inverse_mass(scene, start), dt,
                       candidates.soft_material, scene.metadata['friction_epsilon'], material_start=material_start,
                       tangent_support_scale=scene.metadata.get('soft_tangent_support_scale',1.))
    context = SimpleNamespace(contacts=candidates, rigid=rc, soft=sc, material_start=material_start,
                              pin_velocity=(start.x[:, ids]-previous)/dt, dt=dt,
                              rho_t=torch.cat((rc.rho_t, sc.rho_t), 1),
                              dual_t=torch.cat((rc.dual_t, sc.dual_t), 1),
                              previous_angle=start.history['joint_angle_prev'])
    context.joint_start = scene.joint_coordinates(material_start, context.previous_angle)
    context.targets, context.target_rates, context.efforts = scene.joint_controls
    if _tangent_history_enabled(scene):
        from .surface_tangent_memory import HISTORY_KEYS as tangent_keys, match_force
        context.surface_tangent_history={key:start.history[key] for key in tangent_keys}
        if context.surface_tangent_history['surface_contact_shape'].shape[1] < len(candidates.soft.shape):
            raise ValueError('Current surface contact rows exceed the declared tangent-history capacity')
        depth=candidates.soft.geometry(start,scene.body_com,jacobian=False)[-1]
        normal_load=candidates.soft_material.stiffness[None]*depth.clamp_min(0)
        sc.dual_t=match_force(candidates.soft,context.surface_tangent_history,
                              normal_load,candidates.soft_material.friction)
        context.dual_t=torch.cat((rc.dual_t,sc.dual_t),1)
    return context


def rows(scene, state, start, context):
    c = context.contacts; count = c.rigid_count
    context.rigid.dual_t = context.dual_t[:, :count]
    context.soft.dual_t = context.dual_t[:, count:]
    rr = rigid.rows(scene, state, start, context.rigid)
    sr = surface.rows(c.soft, state, scene.body_com, context.soft)
    jr = joints.joint_rows(scene, state, context)
    ids = scene.joint_spec.owners
    residual = rr['residual']+sr['residual']
    residual = residual.index_add(1, ids.flatten(), jr['pair_residual'].flatten(1, 2))*scene.dof
    terms = [*rr['terms'], sr['metric_term'], MetricTerm(ids, jr['jacobian'].detach(), jr['weight'].detach())]
    return {'residual': residual, 'terms': terms,
            'active': torch.cat((rr['active'], sr['active']), 1),
            'raw_depth': torch.cat((rr['raw_depth'], sr['raw_depth']), 1),
            'tangent_force': torch.cat((rr['tangent_force'], sr['tangent_force']), 1),
            'valid': torch.cat((rr['valid'], sr['valid'], jr['valid']), 1),
            'rigid_rows': rr, 'soft_rows': sr, 'joint_rows': jr}


def system(scene, state, start, tx, tq, context):
    residual, _, extras, base = residual_system(scene, state, context.material_start, tx, tq, context.dt, return_metric_base=True)
    jacobian, _ = relation_inputs(scene.elastic_scene, state, context.material_start, extras, context.dt, torch.ones_like(state.x[..., 0]))
    elastic = FrozenMixedMetric(scene, state, context.material_start, tx, tq, context.dt, extras, jacobian, base=base, elastic_mode='objective')
    result = rows(scene, state, start, context)
    result['elastic_extras'] = extras
    result['topology_active'] = result['active']
    anticipated = []
    if getattr(scene, 'predictive_contact_metric', False):
        from .surface_contact_anticipation import metric as prepare
        joint_term = result['terms'][-1]
        base_metric = rigid.ContactMetric(elastic.local, [*elastic.terms, joint_term], scene.dof)
        joint_residual = state.x.new_zeros(*state.x.shape[:2], 6).index_add(1, scene.joint_spec.owners.flatten(), result['joint_rows']['pair_residual'].flatten(1, 2))*scene.dof
        force_aware=bool(scene.metadata.get('force_aware_material_anticipation',False))
        anticipation_residual=residual+result['residual'] if force_aware else residual+joint_residual
        anticipated, topology, valid = prepare(scene, state, State(tx, tq, start.v, start.omega), base_metric, context.contacts, result, anticipation_residual,
                                               force_aware_material=force_aware,dt=context.dt)
        result['topology_active'] = topology; result['valid'] = torch.cat((result['valid'], valid[:, None]), -1)
        context.anticipated_contacts = int((topology & ~result['active']).sum())
    metric = rigid.ContactMetric(elastic.local, [*elastic.terms, *result['terms'], *anticipated], scene.dof)
    return (residual+result['residual'])*scene.dof, metric, result


def merit(scene, state, start, tx, tq, context, frozen):
    empty = state.x.new_empty(len(state.x), 0)
    energy = incremental_objective(scene, state, context.material_start, tx, tq, context.dt, empty)
    energy = energy+rigid.merit(scene, state, start, tx, tq, context.rigid, frozen['rigid_rows'], include_inertia=False)
    soft = context.contacts.soft
    pair = surface.merit(soft, state, scene.body_com, context.soft, frozen['soft_rows'])
    ids = soft.ports.ids
    mobile = scene.movable[ids] & (soft.ports.owners >= 0) & (soft.ports.weights > 0).any(1)
    count = mobile.sum(-1).clamp_min(1)
    for slot in range(ids.shape[1]):
        # Bool/integer division otherwise produces FP32 fractions: three
        # owners then allocate 1.0000000298 times the FP64 contact energy.
        share = mobile[:, slot].to(energy.dtype)/count
        energy = energy.index_add(1, ids[:, slot], pair*share[None])
    if not len(scene.joint_spec.owners):return energy
    joint = joints.joint_energy(scene, state, context, frozen_bound=frozen['joint_rows']['bound_lower'])
    ids = scene.joint_spec.owners; mobile = scene.movable[ids]; count = mobile.sum(-1).clamp_min(1)
    for side in (0, 1): energy = energy.index_add(1, ids[:, side], joint*(mobile[:, side]/count)[None])
    return energy


def relation_features(scene, state, start, context, result, feedback):
    _, structural = relation_inputs(scene.elastic_scene, state, context.material_start, result['elastic_extras'], context.dt, feedback)
    contact = rigid.relation_features(scene, state, start, context.rigid, result['rigid_rows'], feedback)
    c = context.contacts.soft; r = result['soft_rows']; ctx = context.soft
    b = len(state.x); count = len(c.shape); dt = context.dt; length = scene.length
    value = state.x.new_zeros(b, count, 9); parameter = torch.zeros_like(value); rate = torch.zeros_like(value)
    displacement = r['point'][:, :, 1]-ctx.start_point[:, :, 1]-r['point'][:, :, 0]+ctx.start_point[:, :, 0]+ctx.surface_velocity*dt
    tangent = -(displacement-(displacement*c.normal).sum(-1, keepdim=True)*c.normal)+ctx.c0_t
    mass = (scene.mass[c.ports.ids][:, None]*c.ports.weights).sum((-1, -2))
    value[..., 0] = r['raw_depth']/length; value[..., 1:4] = tangent/length
    value[..., 4] = r['normal_load']*dt**2/(mass[None].clamp_min(1e-30)*length)
    value[..., 5:8] = c.ports.weights[None, :, 1, 1:]
    value[..., 8] = c.feature_size/3
    parameter[..., 0] = torch.log1p(ctx.material.stiffness*length**2)/10
    parameter[..., 1] = torch.log1p(ctx.material.damping*length**2/dt)/10
    parameter[..., 2] = ctx.material.friction
    parameter[..., 3] = (c.radius+c.shape_margin)/length
    parameter[..., 4] = scene.metadata['friction_epsilon']*dt/length
    parameter[..., 5] = c.ports.weights[:, 1, 1:].square().sum(-1)
    velocity = torch.cat((state.v, state.omega), -1)
    speed = torch.einsum('brpsci,brsi->brpc', r['jacobian'], velocity[:, c.ports.ids])
    rate[..., 0] = ((speed[:, :, 1]-speed[:, :, 0])*c.normal).sum(-1)*dt/length
    feed = feedback[:, c.ports.ids[:, 0]]
    scalar = torch.stack((r['active'].to(value.dtype), value.new_full((b, count), math.log(dt)), torch.log1p(mass)[None].expand(b, -1), feed, torch.ones_like(feed)), -1)
    soft = torch.cat((value, parameter, rate, scalar), -1)
    proxy = SimpleNamespace(**{**vars(context), **vars(context.rigid)})
    proxy.previous_angle=context.previous_angle; proxy.joint_start=context.joint_start
    proxy.targets=context.targets; proxy.target_rates=context.target_rates; proxy.efforts=context.efforts
    joint = joints.relation_features(scene, state, start, proxy, {**result['rigid_rows'], 'joint_rows':result['joint_rows']}, feedback)[:, context.contacts.rigid_count:]
    return torch.cat((structural, contact, soft, joint), 1)


def acceptance_groups(scene, candidates, depth, travel, *, labels=False, metric_active=None):
    owners = candidates.owners
    motion = (travel[0, owners.clamp_min(0)]*(owners >= 0)).sum(-1)
    active = (depth[0]+motion >= 0) & (candidates.material > 0)
    if metric_active is not None: active = active | metric_active[0]
    incidence, kinds, enabled = scene.all_relations(owners, active)
    graph = build_topology(scene.kind, scene.movable, scene.object_id, incidence, kinds, potential_active=enabled)
    if labels: return graph.interaction_id
    return [torch.where(graph.interaction_id == key)[0] for key in torch.unique(graph.interaction_id).tolist() if bool(scene.movable[graph.interaction_id == key].any())]


def predictor_domain(scene, start, forecast, config, *, dt=None):
    validate_history(scene, start)
    return pins.predictor_domain(scene, pin_state(start), forecast, config, dt=dt) & joints.predictor_domain(scene, start, forecast, config, dt=dt)


def tangent(scene, state): return joints.tangent(scene, state)
def retract(scene, state): return joints.retract(scene, state)
def valid_state(scene, state): return pins.valid_state(scene, state) & joints.valid_state(scene, state)


def diagnostics(scene, result):
    if not scene.metadata.get('record_grasp_forces'):return {}
    contact=scene.candidates.soft;value=result['soft_rows']
    dynamic=contact.ports.owners[:,0]<scene.world
    tangent=value['tangent_force'][:,dynamic];load=value['normal_load'][:,dynamic]
    limit=scene.candidates.soft_material.friction[dynamic]*load
    ratio=tangent.norm(dim=-1)/limit.clamp_min(1e-30)
    return {'dynamic_soft_tangent_force_world':tangent.sum(1).detach().cpu().tolist(),
            'dynamic_soft_normal_load_sum':load.sum(-1).detach().cpu().tolist(),
            'dynamic_soft_coulomb_capacity_sum':limit.sum(-1).detach().cpu().tolist(),
            'maximum_dynamic_soft_coulomb_ratio':float(ratio.max()) if ratio.numel() else 0.,
            'soft_tangent_support_scale':scene.metadata.get('soft_tangent_support_scale',1.)}


def finish(scene, start, output, context, accepted):
    output = pins.finish(scene, pin_state(start), output, context, accepted)
    value = scene.joint_coordinates(output, context.previous_angle).axis[..., 3]
    angle = torch.where(scene.joint_spec.angular_mode == 1, value, 0.)
    history = {**output.history, 'joint_angle_prev':torch.where(accepted[:, None], angle, context.previous_angle)}
    if _tangent_history_enabled(scene):
        from .surface_tangent_memory import commit
        # rows() synchronizes the child context before evaluating the final
        # accepted force. The parent dual is then updated, so the child's
        # value is one iteration old here.
        first=context.contacts.rigid_count
        final_tangent=context.dual_t[:,first:first+len(context.contacts.soft.shape)]
        history.update(commit(context.contacts.soft,final_tangent,accepted,context.surface_tangent_history))
    velocity,omega=output.v,output.omega
    if prescribed.enabled(scene):
        history,velocity,omega=prescribed.commit(scene,start,output,context,accepted,history,velocity,omega)
    return State(output.x, output.q, velocity, omega, history)
