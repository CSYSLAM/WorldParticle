"""Shared finite elasticity, surface/joint and reciprocal particle self rows."""
import math
from types import SimpleNamespace

import torch

from . import self_contact_rows as self_rows, surface_joint_system as base
from .data import State
from .mixed_metric import FrozenMixedMetric
from .mixed_physics import residual_system
from .mixed_solver import relation_inputs

supports_rigid_component_batch = False
contact_safety = 'BOUNDED_COMPLETE_VF_EE_CANDIDATES_AND_LINEAR_PARTICLE_PATH_CERTIFICATES; ANALYTIC_SHAPE_CCD_NOT_CLAIMED'
HISTORY_KEYS = base.HISTORY_KEYS
coarse = base.coarse
validate_history = base.validate_history
acceptance_groups = base.acceptance_groups
inverse_mass = base.inverse_mass
tangent = base.tangent
retract = base.retract
finish = base.finish


def begin(scene, start, candidates, dt, *, material_start_override=None):
    context = base.begin(scene, start, candidates.base, dt,material_start_override=material_start_override)
    n = scene.metadata['particle_count']
    if not torch.equal(context.material_start.x[0, :n].detach(), candidates.self_snapshot.reference):
        raise ValueError('Self query must originate at previous accepted material state, including prior pins')
    context.contacts = candidates
    context.self_contact = self_rows.begin(candidates.self_contacts, start, scene.body_com, dt,
                                           *candidates.self_material, scene.metadata['friction_epsilon'],
                                           material_start=context.material_start)
    context.self_contact.compact_frozen_contact_metric=bool(getattr(scene,'compact_frozen_contact_metric',False))
    context.self_contact.contact_row_checkpointing=bool(getattr(scene,'contact_row_checkpointing',False))
    if getattr(scene,'learned_contact_repair',False):
        from .self_contact_repair import tolerances
        context.self_contact.allowed_overlap=torch.cat(tolerances(candidates.self_snapshot,scene.rest_x[:n],scene.length))
    count = len(candidates.self_contacts.kind)
    zero = start.x.new_zeros(len(start.x), count, 3)
    context.dual_t = torch.cat((context.dual_t, zero), 1)
    # Native self friction is direct regularized Coulomb; it owns no ALM
    # multiplier. Unit dummy support only keeps the common zero-slot algebra.
    context.rho_t = torch.cat((context.rho_t, zero[..., 0]+1.), 1)
    return context


def rows(scene, state, start, context):
    c = context.contacts
    proxy = SimpleNamespace(**{**vars(context), 'contacts': c.base,
                               'dual_t': context.dual_t[:, :c.surface_count]})
    result = base.rows(scene, state, start, proxy)
    if getattr(scene,'self_contact_refresh_geometry',False):
        from .self_contact_refresh import contexts
        live,frozen,live_context,merit_context=contexts(c.self_contacts,state,scene.body_com,
                                                       context.self_contact,context.material_start)
        sr=self_rows.rows(live,state,scene.body_com,live_context)
        sr.update(evaluation_contacts=live,frozen_geometry=frozen,merit_context=merit_context)
    else:sr = self_rows.rows(c.self_contacts, state, scene.body_com, context.self_contact)
    result['residual'] = (result['residual']+sr['residual'])*scene.dof
    result['terms'] = [*result['terms'], sr['metric_term']]
    for key in ('active', 'raw_depth', 'tangent_force', 'valid'):
        result[key] = torch.cat((result[key], sr[key]), 1)
    result['self_rows'] = sr
    return result


def system(scene, state, start, tx, tq, context):
    residual, _, extras, local = residual_system(scene, state, context.material_start, tx, tq, context.dt, return_metric_base=True)
    jacobian, _ = relation_inputs(scene.elastic_scene, state, context.material_start, extras,
                                  context.dt, torch.ones_like(state.x[..., 0]))
    elastic = FrozenMixedMetric(scene, state, context.material_start, tx, tq, context.dt,
                                extras, jacobian, base=local, elastic_mode='objective')
    result = rows(scene, state, start, context)
    result['elastic_extras'] = extras
    result['topology_active'] = result['active']
    anticipated = []
    if getattr(scene, 'predictive_contact_metric', False):
        from .surface_contact_anticipation import metric as prepare
        # Keep the exact same joint-only baseline used by the surface path;
        # do not count existing contact rows twice in secant compliance.
        joint_term = result['terms'][-2]
        material = base.rigid.ContactMetric(elastic.local, [*elastic.terms, joint_term], scene.dof)
        joint_residual = state.x.new_zeros(*state.x.shape[:2], 6).index_add(
            1, scene.joint_spec.owners.flatten(), result['joint_rows']['pair_residual'].flatten(1, 2))*scene.dof
        anticipated, topology, valid = prepare(scene, state, State(tx, tq, start.v, start.omega),
                                                material, context.contacts, result, residual+joint_residual)
        result['topology_active'] = topology
        result['valid'] = torch.cat((result['valid'], valid[:, None]), -1)
        context.anticipated_contacts = int((topology & ~result['active']).sum())
    metric = base.rigid.ContactMetric(elastic.local, [*elastic.terms, *result['terms'], *anticipated], scene.dof)
    return (residual+result['residual'])*scene.dof, metric, result


def merit(scene, state, start, tx, tq, context, frozen):
    energy = base.merit(scene, state, start, tx, tq, context, frozen)
    c = context.contacts.self_contacts
    if getattr(scene,'self_contact_refresh_geometry',False):
        from .self_contact_refresh import merit as refreshed_merit
        pair=refreshed_merit(c,state,scene.body_com,context.self_contact,frozen['self_rows'])
    else:pair = self_rows.merit(c, state, scene.body_com, context.self_contact, frozen['self_rows'])
    ids = c.ports.ids
    mobile = scene.movable[ids] & (c.ports.owners >= 0) & (c.ports.weights > 0).any(1)
    count = mobile.sum(-1).clamp_min(1)
    for slot in range(ids.shape[1]):
        energy = energy.index_add(1, ids[:, slot], pair*(mobile[:, slot].to(energy.dtype)/count)[None])
    return energy


def self_relation_features(scene, state, context, result, feedback, *, selected=None):
    """Original continuous feature formula on all or selected immutable rows."""
    c = result['self_rows'].get('evaluation_contacts',context.contacts.self_contacts)
    r, ctx = result['self_rows'], context.self_contact
    def row(value):return value if selected is None else value[selected]
    def batched(value):return value if selected is None else value[:,selected]
    ids,weights,kinds,margin=row(c.ports.ids),row(c.ports.weights),row(c.kind),row(c.margin)
    b, count, dt, length = len(state.x), len(kinds), context.dt, scene.length
    value = state.x.new_zeros(b, count, 9)
    parameter, rate = torch.zeros_like(value), torch.zeros_like(value)
    tangent = batched(r['tangent_displacement'])
    mass = (scene.mass[ids][:, None]*weights).sum((-1, -2))
    # Four prescribed particle owners can all have zero exported mass. Their
    # geometry still needs a path certificate, but a zero mass cannot define a
    # neural feature scale. Use the physical scene's positive particle mass.
    particle_mass = scene.mass[:scene.metadata['particle_count']]
    positive = particle_mass[particle_mass > 0]
    reference_mass = positive.mean() if len(positive) else mass.new_tensor(1.)
    mass = torch.where(mass > 0, mass, reference_mass)
    value[..., 0] = batched(r['raw_depth'])/length
    value[..., 1:4] = tangent/length
    value[..., 4] = batched(r['normal_load'])*dt**2/(mass[None].clamp_min(1e-30)*length)
    value[..., 5:8] = weights[None, :, 0, :3]
    value[..., 8] = (kinds == 10).to(value.dtype)
    parameter[..., 0] = torch.log1p(row(ctx.stiffness)*length**2)/10
    parameter[..., 1] = torch.log1p(row(ctx.damping)*length**2/dt)/10
    parameter[..., 2] = row(ctx.friction)
    parameter[..., 3] = margin/length
    parameter[..., 4] = scene.metadata['friction_epsilon']*dt/length
    parameter[..., 5] = weights[:, 0].square().sum(-1)
    parameter[..., 6] = weights[:, 1].square().sum(-1)
    parameter[..., 7] = weights[:, 1, 2]
    parameter[..., 8] = weights[:, 1, 3]
    velocity = torch.cat((state.v, state.omega), -1)
    speed = c.ports.point_velocity(r['jacobian'], velocity,rows=selected)
    rate[..., 0] = ((speed[:, :, 1]-speed[:, :, 0])*row(c.normal)).sum(-1)*dt/length
    rate[..., 1:4] = tangent/length
    mean_weights = weights.sum(1)/2
    feed = (feedback[:, ids]*mean_weights[None]).sum(-1)
    scalar = torch.stack((batched(r['active']).to(value.dtype), value.new_full((b, count), math.log(dt)),
                          torch.log1p(mass)[None].expand(b, -1), feed, torch.ones_like(feed)), -1)
    return torch.cat((value, parameter, rate, scalar), -1)


def relation_features(scene, state, start, context, result, feedback):
    previous = base.relation_features(scene, state, start, context, result, feedback)
    count=len(context.contacts.self_contacts.kind)
    if getattr(scene,'compact_neural_relations',False):
        # The existing neural pack and coarse gain read only active rows.
        # Form their continuous features directly, retaining zero placeholders
        # for the same full physical layout. All force/metric/path rows and
        # their activity are still evaluated against the complete snapshot.
        selected=torch.where(result['self_rows']['active'].detach().any(0))[0]
        values=self_relation_features(scene,state,context,result,feedback,selected=selected)
        encoded=values.new_zeros(len(state.x),count,32).index_copy(1,selected,values)
    else:encoded=self_relation_features(scene,state,context,result,feedback)
    split = len(scene.structural_kind)+context.contacts.surface_count
    return torch.cat((previous[:, :split], encoded, previous[:, split:]), 1)


def path_feasible(scene, start, end):
    n = scene.metadata['particle_count']
    snapshot = scene.candidates.self_snapshot
    if getattr(scene,'learned_contact_repair',False):
        from .self_contact_repair import bounded_path
        return base.path_feasible(scene,start,end)&bounded_path(snapshot,end.x[0,:n],scene.rest_x[:n],scene.length)[None]
    return base.path_feasible(scene, start, end) & snapshot.path_feasible(end.x[0, :n])[None]


def proposal_scale(scene,state,update):
    if getattr(scene,'learned_contact_repair',False):return None
    if not getattr(scene,'self_geometric_step_cap',False):return None
    from .self_contact_step import proposal_scale as cap
    return cap(scene.candidates.self_snapshot,state,update)


def predictor_domain(scene, start, forecast, config, *, dt=None):
    valid = base.predictor_domain(scene, start, forecast, config, dt=dt)
    # The free inertial forecast is only a target. Requiring its path to pass
    # self-contact would incorrectly reject collisions before correction.
    return valid & path_feasible(scene, start, start)


def valid_state(scene, state):
    return base.valid_state(scene, state) & path_feasible(scene, state, state)


def diagnostics(scene, result):
    c = scene.candidates.self_contacts
    participating=(c.ports.owners>=0)&(c.ports.weights>0).any(1)
    kinds=scene.kind[c.ports.ids]
    cross=((kinds==0)&participating).any(-1)&((kinds==1)&participating).any(-1)
    responsive_only=bool(scene.metadata.get('self_contact_responsive_pairs_only',False))
    result_diagnostics={'self_candidate_rows_by_kind': {str(kind): int((c.kind == kind).sum()) for kind in (9, 10)},
            'active_self_rows_by_kind': {str(kind): int((result['self_rows']['active'] & (c.kind == kind)[None]).sum()) for kind in (9, 10)},
            'active_soft_cloth_self_rows':int((result['self_rows']['active']&cross[None]).sum()),
            'self_EE_reciprocal_topology_rule': 'union_of_source_directional_exclusions',
            'self_contact_pair_scope':'at_least_one_responsive_primitive_vertex' if responsive_only else 'all_authored_topology_pairs',
            'contact_safety':('BOUNDED_MOTION_AND_LOCAL_FINITE_ORIENTED_OVERLAP_WITH_REPAIR; NO_ZERO_INTERSECTION_CERTIFICATE' if getattr(scene,'learned_contact_repair',False) else
                'BOUNDED_COMPLETE_RESPONSIVE_VF_EE_CANDIDATES_AND_LINEAR_PARTICLE_PATH_CERTIFICATES; PRESCRIBED_PRESCRIBED_SEPARATION_NOT_ASSERTED; ANALYTIC_SHAPE_CCD_NOT_CLAIMED' if responsive_only else contact_safety)}
    if scene.metadata.get('prescribed_pose_history'):
        soft=scene.candidates.soft;n=scene.metadata['particle_count']
        participating=(soft.ports.owners>=0)&(soft.ports.weights>0).any(1)
        mobile_particle=((soft.ports.ids<n)&scene.movable[soft.ports.ids]&participating).any(-1)
        body=soft.ports.owners[:,0]
        counts={str(owner-n):int((result['soft_rows']['active']&(body==owner)[None]&mobile_particle[None]).sum())
                for owner in scene.prescribed_ids.tolist()}
        result_diagnostics.update(active_prescribed_particle_body_rows=sum(counts.values()),
                                  active_prescribed_particle_body_rows_by_body=counts)
    return result_diagnostics
