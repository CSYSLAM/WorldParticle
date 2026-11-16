"""Compliant rod material, contact and one COM inertia in a shared response."""
import math
from types import SimpleNamespace

import torch

from . import math3d as so3, joint_history as memory, rigid_contact_system as contact
from .data import State
from .mixed_metric import MetricTerm
from .rod_rows import RodContext, automatic_rho, coordinates, evaluate, rotate_pinned
from .unified_graph import build_topology

supports_component_batch = False
HISTORY_KEYS = {'rod_dual', 'rod_kappa', 'rod_increment', 'rod_stress',
                'rod_previous_x', 'rod_previous_q', 'rod_initialized'}
contact_safety = 'FRESH_EXPANDED_RIGID_CONTACTS; FINITE_PENETRATION; NONANTIPARALLEL_ROD_MATERIAL_DOMAIN'


def initial_history(scene, state):
    zeros = state.x.new_zeros(len(state.x), len(scene.rod.joints), 3)
    history={'rod_dual': zeros.new_zeros(*zeros.shape[:-1], 6), 'rod_kappa': zeros.clone(),
            'rod_increment': zeros.clone(), 'rod_stress': zeros.clone(),
            'rod_previous_x': state.x.clone(), 'rod_previous_q': state.q.clone(),
            'rod_initialized': torch.zeros(len(state.x), device=scene.device, dtype=torch.bool)}
    if scene.metadata.get('cross_step_contact_history'):
        from .rigid_contact_memory import initial_history as contact_history
        history.update(contact_history(scene,state))
    if scene.metadata.get('rod_finite_joint_contract'):
        spec=scene.joint_spec
        angle=scene.joint_coordinates(state,spec.coordinate_offset[None,:,3],jacobian=False).axis[...,3]
        history['joint_angle_prev']=torch.where(spec.angular_mode==1,angle,0.)
    return history


def history_keys(scene):
    keys=HISTORY_KEYS|({'joint_angle_prev'} if scene.metadata.get('rod_finite_joint_contract') else set())
    if scene.metadata.get('cross_step_contact_history'):
        from .rigid_contact_memory import history_keys as contact_keys
        return keys|contact_keys(scene)
    return keys


def validate_history(scene, start):
    if not isinstance(start.history, dict) or set(start.history) != history_keys(scene):
        raise ValueError('Complete accepted rod/contact history required')
    expected = initial_history(scene, start)
    for name in HISTORY_KEYS|({'joint_angle_prev'} if scene.metadata.get('rod_finite_joint_contract') else set()):
        value=start.history[name]
        if value.shape != expected[name].shape or value.dtype != expected[name].dtype or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid rod/contact history: '+name)
    if not bool(((start.history['rod_previous_q'].square().sum(-1)-1).abs() < 1e-8).all()):
        raise ValueError('Previous rod/contact orientations must be normalized')
    if scene.metadata.get('cross_step_contact_history'):
        from .rigid_contact_memory import validate
        validate(scene,start)


def strain(scene, state):
    spec = scene.rod; ids = spec.joints
    origin = state.x-rotate_pinned(state.q, scene.body_com)
    return coordinates(origin[:, ids], state.q[:, ids], spec.anchor_p, spec.anchor_c, spec.anchor_qp,
                       spec.anchor_qc, spec.rod_rest_bend, spec.rod_rest_twist, pinned=True)


def begin(scene, start, candidates, dt):
    validate_history(scene, start); spec = scene.rod; h = start.history
    initialized = h['rod_initialized'][:, None, None]
    use_previous = initialized
    if getattr(scene,'rod_prescribed_inputs',False):
        # The first external pose command is motion from the accepted native
        # initial pose too; it must contribute damping and contact velocity.
        use_previous = initialized | ~scene.movable[None,:,None]
    previous = State(torch.where(use_previous, h['rod_previous_x'], start.x),
                     torch.where(use_previous, h['rod_previous_q'], start.q), start.v, start.omega)
    context = contact.begin(scene, start, candidates, dt); context.material_start = previous
    if scene.metadata.get('cross_step_contact_history'):
        from .rigid_contact_memory import tangent_warm_start
        context.dual_t=tangent_warm_start(scene,start,candidates,context.rho_n)
        context.contact_cache=scene.geometry_oracle.contact_history_proposal
        context.matched_contact_history=True
    context.rod,context.rod_memory=begin_material(scene,previous,h,dt)
    return context


def begin_material(scene, previous, history, dt):
    """Freeze native rod material independently of the composing contact set."""
    spec=scene.rod;h=history
    previous_c = strain(scene, previous)
    enabled = torch.ones_like(previous_c[..., 0], dtype=torch.bool)
    frozen, sigma, friction = memory.begin_dahl(memory.DahlMemory(h['rod_kappa'], h['rod_increment'], h['rod_stress']),
                                               previous_c[..., 3:], spec.joint_stiffness[:, 3:],
                                               spec.rod_dahl_eps, spec.rod_dahl_tau, enabled, ~h['rod_initialized'][:, None])
    rho = automatic_rho(spec.joint_stiffness, spec.rod_inverse_mass[spec.joints], spec.rod_inverse_inertia[spec.joints], dt)
    rod = RodContext(spec.joint_stiffness, rho, h['rod_dual']*scene.metadata['joint_gamma'],
                             previous_c, previous_c, spec.rod_damping, sigma, friction, dt,
                             scene.metadata['joint_alpha'], True)
    return rod,frozen


def material_rows(scene, state, context):
    spec = scene.rod; ids = spec.joints
    origin = state.x-rotate_pinned(state.q, scene.body_com)
    return evaluate(origin[:, ids], state.q[:, ids], scene.body_com[ids], spec.anchor_p, spec.anchor_c,
                    spec.anchor_qp, spec.anchor_qc, spec.rod_rest_bend, spec.rod_rest_twist, context.rod)


def rows(scene, state, start, context):
    result = contact.rows(scene, state, start, context); rod = material_rows(scene, state, context)
    result['residual'] = result['residual'].index_add(1, scene.rod.joints.flatten(), rod['residual'].flatten(1, 2))*scene.dof
    result['terms'] = [*result['terms'], MetricTerm(scene.rod.joints, rod['jacobian'].detach(), context.rod.terms(rod['coordinates'])[1].detach())]
    result['valid'] = torch.cat((result['valid'], rod['in_domain']), -1); result['rod_rows'] = rod
    return result


def system(scene, state, start, tx, tq, context):
    # contact.system already includes inertia exactly once.
    residual, metric, result = contact.system(scene, state, start, tx, tq, context)
    rod = material_rows(scene, state, context)
    residual = residual.index_add(1, scene.rod.joints.flatten(), rod['residual'].flatten(1, 2))*scene.dof
    term = MetricTerm(scene.rod.joints, rod['jacobian'].detach(), context.rod.terms(rod['coordinates'])[1].detach())
    from .rod_material_ports import fields
    profile = None
    if getattr(scene,'rod_force_ports',False):
        stress, weight, _ = context.rod.terms(rod['coordinates'])
        profile = torch.where(weight>0,stress/weight.clamp_min(1e-30),0.).detach()
    scene._rod_material_fields, material_valid = fields(scene, rod['jacobian'].detach(), profile)
    if getattr(scene,'rod_gravity_ports',False):
        from .rod_gravity_ports import profile as gravity_profile
        _,weight,_=context.rod.terms(rod['coordinates'])
        load,load_valid=gravity_profile(scene,state,rod['jacobian'].detach(),weight.detach())
        scene._rod_gravity_fields,field_valid=fields(scene,rod['jacobian'].detach(),load)
        material_valid=material_valid&load_valid&field_valid
        result['rod_gravity_material_profile']=load
    result['rod_material_profile'] = profile
    result['valid'] = torch.cat((result['valid'], rod['in_domain'], material_valid[:, None]), -1)
    result['rod_rows'] = rod; result['terms'] = [*result['terms'], term]
    return residual, contact.ContactMetric(metric.local, result['terms'], scene.dof), result


def coarse(scene, state):
    result = contact.coarse(scene, state)
    for ids in scene.interaction_entities:
        ids = ids[scene.movable[ids]]
        if len(ids):
            result.blocks.append((SimpleNamespace(ids=ids), scene._rod_material_fields[:, ids], None, None, None))
            if getattr(scene,'rod_gravity_ports',False):
                result.blocks.append((SimpleNamespace(ids=ids),scene._rod_gravity_fields[:,ids],None,None,None))
    return result


def merit(scene, state, start, tx, tq, context, frozen):
    energy = contact.merit(scene, state, start, tx, tq, context, frozen)
    pair = context.rod.terms(strain(scene, state))[2]
    ids = scene.rod.joints; mobile = scene.movable[ids]; count = mobile.sum(-1).clamp_min(1)
    for side in (0, 1):
        energy = energy.index_add(1, ids[:, side], pair*(mobile[:, side].to(energy.dtype)/count)[None])
    return energy


def relation_features(scene, state, start, context, result, feedback):
    base = contact.relation_features(scene, state, start, context, result, feedback)
    return torch.cat((base, material_relation_features(scene,state,start,context,result,feedback)),1)


def material_relation_features(scene, state, start, context, result, feedback):
    """Native rod relation features without assuming a body-only contact codec."""
    spec = scene.rod; c = result['rod_rows']['coordinates']; dt = context.dt; b, r = c.shape[:2]
    value = c.new_zeros(b, r, 9); parameter = torch.zeros_like(value); rate = torch.zeros_like(value)
    scaling = c.new_tensor([1/scene.length]*3+[1.]*3)
    value[..., :6] = c*scaling
    parameter[..., :6] = torch.log1p(spec.joint_stiffness/scaling.square())/10
    parameter[..., 6] = torch.log1p(spec.rod_damping.sum(-1)/dt)/10
    parameter[..., 7] = spec.rod_dahl_eps; parameter[..., 8] = spec.rod_dahl_tau
    increment = torch.cat((c[..., :3]-context.rod.previous[..., :3], memory.rod_strain_increment(c[..., 3:], context.rod.previous[..., 3:])), -1)
    rate[..., :6] = increment*scaling
    if getattr(scene,'rod_memory_features',False):
        envelope=spec.joint_stiffness[:,3:]*spec.rod_dahl_eps[:,None]
        active=(spec.rod_dahl_eps>0)&(spec.rod_dahl_tau>0)
        value[...,6:9]=torch.where(active[None,:,None],context.rod.sigma/envelope.clamp_min(1e-30),0.)
        rate[...,6:9]=torch.where(active[None,:,None],start.history['rod_increment']/spec.rod_dahl_tau[None,:,None].clamp_min(1e-30),0.)
    mass = scene.mass[spec.joints].sum(-1)
    active = (spec.joint_stiffness+spec.rod_damping).sum(-1) > 0
    feed = feedback[:, spec.joints[:, 0]]
    scalar = torch.stack((active[None].expand(b, -1).to(c.dtype), c.new_full((b, r), math.log(dt)),
                          torch.log1p(mass)[None].expand(b, -1), feed, torch.ones_like(feed)), -1)
    return torch.cat((value, parameter, rate, scalar), -1)


def acceptance_groups(scene, candidates, depth, travel, *, labels=False, metric_active=None):
    active = (depth[0]+travel[0, candidates.owners].sum(-1) >= 0) & (candidates.material > 0)
    if metric_active is not None: active |= metric_active[0]
    ids, kinds, enabled = scene.all_relations(candidates.owners, active)
    graph = build_topology(scene.kind, scene.movable, scene.object_id, ids, kinds, potential_active=enabled)
    if labels: return graph.interaction_id
    return [torch.where(graph.interaction_id == key)[0] for key in torch.unique(graph.interaction_id).tolist()
            if bool(scene.movable[graph.interaction_id == key].any())]


def internal_dual_small(scene, state, context, result):
    rod = context.rod; next_dual = rod.numeric_trial(result['rod_rows']['coordinates']).dual
    change = (next_dual-rod.dual).abs()/rod.rho.clamp_min(1e-30)
    row_small = (change[..., :3].amax(-1) <= scene.length*5e-7) & (change[..., 3:].amax(-1) <= 5e-7)
    bad = torch.zeros_like(state.x[..., 0], dtype=torch.long)
    bad = bad.index_add(1, scene.rod.joints.flatten(), (~row_small)[:, :, None].expand(-1, -1, 2).flatten(1, 2).long())
    return bad == 0


def update_internal(scene, current, context, result):
    context.rod = context.rod.numeric_trial(result['rod_rows']['coordinates'])


def diagnostics(scene, result):
    # These are the already evaluated rows before the final internal dual
    # update. Re-evaluation with the advanced dual would report another force.
    return {'active_rigid_contact_rows':int(result['active'].sum()),
            'rod_rows_with_nonzero_wrench':int((result['rod_rows']['residual'].abs().amax((-1,-2)) > 1e-12).sum())}


def valid_state(scene, state):
    spec = scene.rod; ids = spec.joints
    a = so3.multiply(state.q[:, ids[:, 0]], spec.anchor_qp); b = so3.multiply(state.q[:, ids[:, 1]], spec.anchor_qc)
    return ((so3.matrix(a)[..., 2]*so3.matrix(b)[..., 2]).sum(-1) > -.9).all(-1)


def predictor_domain(scene, start, forecast, config, *, dt=None):
    validate_history(scene, start)
    old = State(start.history['rod_previous_x'], start.history['rod_previous_q'], start.v, start.omega)
    fixed = ~scene.movable
    angle = so3.log(so3.multiply(start.q, so3.conjugate(old.q))).norm(dim=-1)
    travel = (start.x-old.x).norm(dim=-1)+scene.body_radius*angle
    changed = ((travel[:, fixed] <= scene.geometry_oracle.motion_budget) & (angle[:, fixed] < torch.pi-1e-6)).all(-1)
    if getattr(scene,'rod_prescribed_inputs',False):
        return path_feasible(scene, start, forecast) & changed & path_feasible(scene, old, start)
    return path_feasible(scene, start, forecast) & (changed | ~start.history['rod_initialized'])


def path_feasible(scene, start, candidate):
    spec = scene.rod; ids = spec.joints
    a = so3.multiply(start.q[:, ids[:, 0]], spec.anchor_qp); b = so3.multiply(start.q[:, ids[:, 1]], spec.anchor_qc)
    cosine = (so3.matrix(a)[..., 2]*so3.matrix(b)[..., 2]).sum(-1).clamp(-1., 1.)
    motion = so3.log(so3.multiply(candidate.q, so3.conjugate(start.q))).norm(dim=-1)
    # Triangle inequality on S2 bounds the entire geodesic proposal path.
    maximum = torch.acos(cosine)+motion[:, ids].sum(-1)
    return (maximum < math.acos(-.9)).all(-1) & valid_state(scene, candidate)


def tangent(scene, state): return None
def retract(scene, state): return state, valid_state(scene, state)


def finish(scene, start, output, context, accepted):
    if not bool(accepted.any()):
        return State(output.x, output.q, start.v, start.omega, start.history)
    spec = scene.rod; final = strain(scene, output)
    value = memory.commit_dahl(context.rod_memory, final[..., 3:], spec.joint_stiffness[:, 3:],
                               spec.rod_dahl_eps, spec.rod_dahl_tau, torch.ones_like(final[..., 0], dtype=torch.bool))
    candidate = {'rod_dual': context.rod.dual, 'rod_kappa': value.kappa, 'rod_increment': value.increment,
                 'rod_stress': value.stress, 'rod_previous_x': output.x, 'rod_previous_q': output.q,
                 'rod_initialized': torch.ones_like(start.history['rod_initialized'])}
    if scene.metadata.get('rod_finite_joint_contract'):
        angle=scene.joint_coordinates(output,start.history['joint_angle_prev'],jacobian=False).axis[...,3]
        candidate['joint_angle_prev']=torch.where(scene.joint_spec.angular_mode==1,angle,0.)
    if scene.metadata.get('cross_step_contact_history'):
        from .rigid_contact_memory import commit
        candidate.update(commit(scene,context,output))
    if any(not bool(torch.isfinite(v).all()) for v in candidate.values()):
        raise ValueError('Nonfinite rod/contact material history; physical step not committed')
    def choose(value, previous): return torch.where(accepted.reshape(len(accepted), *([1]*(value.ndim-1))), value, previous)
    if scene.metadata.get('cross_step_contact_history'):
        if len(accepted)!=1 or not bool(accepted.all()):raise ValueError('Matched history commits one accepted physical sequence')
        # Complete sparse contact caches can have a different unused capacity
        # suffix from their input. Batch1 acceptance publishes the new cache
        # as one transaction; a rejected step already returned above.
        physical_keys=HISTORY_KEYS|{'joint_angle_prev'}
        history={key:(choose(value,start.history[key]) if key in physical_keys else value) for key,value in candidate.items()}
    else:history = {key: choose(value, start.history[key]) for key, value in candidate.items()}
    fixed = ~scene.movable
    if getattr(scene,'rod_caller_pose_history',False):
        # Native update_body_velocity measures every body against the raw
        # previous accepted pose, including dynamic poses edited by a caller.
        fixed=torch.ones_like(fixed)
    v = torch.where(fixed[None, :, None], (output.x-context.material_start.x)/context.dt, output.v)
    omega = torch.where(fixed[None, :, None], so3.log(so3.multiply(output.q, so3.conjugate(context.material_start.q)))/context.dt, output.omega)
    return State(output.x, output.q, choose(v, start.v), choose(omega, start.omega), history)
