"""Authored pure-rod scenes: origin poses, material parameters and history.

No full solver is constructed. Active contacts, other joint types and external
coupling remain explicit admission failures rather than disabled features.
"""

import numpy as np


HISTORY_KEYS = ('dual', 'kappa', 'increment', 'stress', 'previous_x', 'previous_q', 'reset')


def export_fields(fields, configuration, identifier, dt, *, additional_joint_types=(), allow_elastic_particles=False):
    if type(allow_elastic_particles) is not bool:
        raise ValueError('Explicit Boolean elastic-particle descriptor contract required')
    def get(name):
        return np.asarray(fields[name])
    n = len(get('body_q'))
    types = get('joint_type').astype(np.int64)
    # Enum values come from the pinned public API, never fixture numbers.
    free, rod = int(configuration['free_joint_type']), int(configuration['rod_joint_type'])
    if np.any(~np.isin(types, (free, rod, *additional_joint_types))):
        raise ValueError('This adapter requires only FREE/ROD joints')
    for name in ('particle_count', 'attachment_particle_body_count', 'tendon_count', 'joint_mimic_count'):
        # Only material descriptors are exported here. A composing backend
        # must own the particles, inertia, contact rows and complete history.
        if name=='particle_count' and allow_elastic_particles:continue
        if int(fields.get(name, 0)):
            raise ValueError(f'Additional physics needs another adapter: {name}')
    if not configuration.get('rigid_compliant_alm', True) or configuration.get('integrate_with_external_rigid_solver'):
        raise ValueError('The rod history path requires compliant ALM and owned rigid integration')
    ids = np.where(types == rod)[0]
    if not len(ids) or not np.all(get('joint_enabled')[ids]):
        raise ValueError('At least one enabled rod relation is required; disabled topology is not admitted yet')
    pairs = np.stack((get('joint_parent')[ids], get('joint_child')[ids]), -1).astype(np.int64)
    if np.any(pairs < 0):
        raise ValueError('World-parent rods need an explicit world frame')
    gravity = get('gravity').reshape(-1, 3)
    if len(gravity) != 1:
        raise ValueError('Multiple gravity worlds need their own batching adapter')
    starts = get('joint_qd_start')[ids].astype(np.int64)
    dofs = starts[:, None] + np.arange(6)
    stiffness, damping = get('joint_target_ke')[dofs], get('joint_target_kd')[dofs]
    if np.any(stiffness < 0) or np.any(damping < 0):
        raise ValueError('Negative rod material coefficients are unsupported')
    # Pinned public joint_target_q_start aliases joint_q_start.
    target_starts = get('joint_target_q_start' if 'joint_target_q_start' in fields else 'joint_q_start')[ids].astype(np.int64)
    rest_q = get('joint_target_q')[target_starts[:, None] + np.arange(3, 7)]
    rest_q = rest_q.astype(np.float32)
    rest_q = rest_q / np.linalg.norm(rest_q, axis=-1, keepdims=True)
    x, y, z, w = rest_q.T
    tangent = np.stack((2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)), -1)
    normal = np.stack((1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)), -1)
    denominator = 1 + tangent[:, 2:]
    if np.any(denominator < .1):raise ValueError('Near-fold rest material frames are outside this rod contract')
    bend = 2 * np.cross(np.array([0., 0., 1.], np.float32), tangent) / denominator
    transported = np.array([1., 0., 0.], np.float32) - tangent[:, :1] / denominator * (tangent + np.array([0., 0., 1.], np.float32))
    transported /= np.linalg.norm(transported, axis=-1, keepdims=True)
    twist = np.arctan2(np.sum(tangent*np.cross(transported,normal),-1),np.sum(transported*normal,-1))
    rest = np.concatenate((bend[:, :2], twist[:, None]), -1)
    parent = list(range(n))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for a, b in pairs:
        parent[root(int(b))] = root(int(a))
    labels = {}; objects = np.empty(n, np.int64); groups = []
    lengths = []
    for i in range(n):
        label = root(i)
        if label not in labels:
            labels[label] = len(labels)
            groups.append({'name': f'rod_{len(groups)}', 'family': 'rod'})
        objects[i] = labels[label]
    for obj in range(len(groups)):
        points = get('body_q')[objects == obj, :3]
        lengths.append(max(float(np.linalg.norm(np.ptp(points, axis=0))), .1))
    poses = get('body_q').astype(np.float32)
    xp, xc = get('joint_X_p')[ids], get('joint_X_c')[ids]
    zeros = lambda shape: np.zeros(shape, np.float32)
    values = dict(rest_x=poses[:, :3], rest_q=poses[:, 3:], mass=get('body_mass').astype(np.float32), movable=get('body_inv_mass') > 0, inertia=get('body_inertia').astype(np.float32), kind=np.full(n, 3, np.int64), object_id=objects, gravity=gravity[0].astype(np.float32), reference_length=np.asarray(max(lengths), np.float32), body_com=get('body_com').astype(np.float32), joints=pairs, joint_tag=np.ones(len(ids), np.int64), joint_stiffness=stiffness.astype(np.float32), joint_material=stiffness[:, [2, 0, 3, 5]].astype(np.float32), anchor_p=xp[:, :3], anchor_c=xc[:, :3], anchor_qp=xp[:, 3:], anchor_qc=xc[:, 3:], joint_axis=np.tile([0., 0., 1.], (len(ids), 1)).astype(np.float32), joint_rest_q=rest_q, rod_rest_bend=np.pad(rest[:, :2], ((0, 0), (0, 1))), rod_rest_twist=rest[:, 2], rod_damping=damping.astype(np.float32), rod_inverse_mass=get('body_inv_mass').astype(np.float32), rod_inverse_inertia=get('body_inv_inertia').astype(np.float32), source_joint_ids=ids.astype(np.int64), kinematic_ids=np.where(get('body_inv_mass') == 0)[0].astype(np.int64))
    for key, shape, dtype in (('tets', (0, 4), np.int64), ('tet_Dm', (0, 3, 3), np.float32), ('tet_material', (0, 3), np.float32), ('triangles', (0, 3), np.int64), ('tri_Dm', (0, 2, 2), np.float32), ('tri_area', (0,), np.float32), ('tri_material', (0, 5), np.float32), ('bend_edges', (0, 4), np.int64), ('bend_angle', (0,), np.float32), ('bend_length', (0,), np.float32), ('bend_material', (0, 2), np.float32)):
        values[key] = np.empty(shape, dtype)
    values.update(site_owner=np.empty(0, np.int64), site_local=zeros((0, 3)), site_radius=zeros((0,)), site_margin=zeros((0,)), site_mu=zeros((0,)), plane_height=np.asarray(0., np.float32), contact_ke=np.asarray(0., np.float32), contact_kd=np.asarray(0., np.float32), friction_epsilon=np.asarray(configuration.get('friction_epsilon', .01), np.float32))
    eps = fields.get('vbd__dahl_eps_max', np.zeros(len(types), np.float32))
    tau = fields.get('vbd__dahl_tau', np.zeros(len(types), np.float32))
    values['rod_dahl_eps'] = np.asarray(eps, np.float32)[ids]
    values['rod_dahl_tau'] = np.asarray(tau, np.float32)[ids]
    joint_alpha=configuration.get('rigid_avbd_joint_alpha')
    if joint_alpha is None:joint_alpha=configuration.get('rigid_avbd_alpha')
    if joint_alpha is None:joint_alpha=0.  # Pinned compliant-ALM material default.
    metadata = {'family': 'mixed', 'official_demo': identifier, 'contract': 'rod_compliant_alm_v1', 'history_contract': 'rod_compliant_alm_v1', 'body_position_representation': 'origin', 'contacts': 'none', 'box_sites': [], 'groups': groups, 'particle_count': 0, 'body_count': n, 'source_joint_count':len(types), 'dt': dt, 'joint_alpha': float(joint_alpha), 'joint_gamma': float(configuration.get('rigid_avbd_gamma', .999)), 'solver_configuration': configuration}
    return values, metadata


def native_configuration(configuration,newton):
    """Bind the two derived enum identities used by the rod descriptor."""
    return {**configuration,'free_joint_type':int(newton.JointType.FREE),'rod_joint_type':int(newton.JointType.ROD)}


def export_model(model, configuration, identifier, newton, dt, *, allow_general_joints=False, allow_elastic_particles=False):
    from .reference_tape import arrays
    fields = arrays(model)
    configuration = native_configuration(configuration,newton)
    additional = tuple(int(getattr(newton.JointType, name)) for name in
                       ('BALL', 'FIXED', 'REVOLUTE', 'PRISMATIC', 'D6')) if allow_general_joints else ()
    return export_fields(fields, configuration, identifier, dt, additional_joint_types=additional,
                         allow_elastic_particles=allow_elastic_particles)


def initial_history(scene, state):
    import torch
    batch = len(state.x); count = len(scene.joints)
    return {'dual': state.x.new_zeros(batch, count, 6), 'kappa': state.x.new_zeros(batch, count, 3), 'increment': state.x.new_zeros(batch, count, 3), 'stress': state.x.new_zeros(batch, count, 3), 'previous_x': state.x.clone(), 'previous_q': state.q.clone(), 'reset': torch.ones(batch, 1, device=state.x.device, dtype=torch.bool)}
