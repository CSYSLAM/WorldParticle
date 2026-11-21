"""Finite elastic particles and free rigid bodies with shared contact DOFs."""
from types import SimpleNamespace

import numpy as np
import torch

from .adapters import export_static
from .mixed_state import MixedScene
from .unified_graph import build_topology


def export_model(model, configuration, identifier, newton, dt, device='cpu', *, finite_joint_export=False, self_contact_export=False, coupled_proxy_view=False):
    if self_contact_export and not finite_joint_export:
        raise ValueError('Self-contact static export requires its complete finite-joint assembly')
    if (configuration['particle_enable_self_contact'] and not self_contact_export) or configuration['rigid_contact_history'] or configuration['integrate_with_external_rigid_solver']:
        raise ValueError('Self-contact, retained contact history and external ownership need their own operators')
    if model.spring_count or model.attachment_particle_body_count or getattr(model,'tendon_count',0):
        raise ValueError('Additional spring/attachment operators are not admitted by this mixed contract')
    if not finite_joint_export and np.any(model.joint_type.numpy() != int(newton.JointType.FREE)):
        raise ValueError('Mixed finite joints require a separate operator integration')
    if coupled_proxy_view:
        from .coupled_proxy import validate_proxy_view
        validate_proxy_view(model,configuration,newton)
    elif np.any(model.body_flags.numpy() != int(newton.BodyFlags.DYNAMIC)):
        raise ValueError('Prescribed/proxy rigid motion requires a separate ownership contract')
    gravity = model.gravity.numpy().reshape(-1, 3)
    if len(gravity) != 1:
        raise ValueError('Packed mixed worlds need explicit particle/body gravity ownership')
    node = export_static('softbody', model, {}, gravity[0], newton)
    body = export_static('joint', model, {}, gravity[0], newton, include_joint_relations=not finite_joint_export)
    shared = ('rest_x', 'rest_q', 'mass', 'movable', 'inertia', 'kind', 'object_id')
    values = {key: np.concatenate((node[key], body[key])) for key in shared}
    values.update({key: value for key, value in node.items() if key not in shared})
    values.update({key: value for key, value in body.items() if key not in shared})
    n = model.particle_count; b = model.body_count; world = n + b
    if coupled_proxy_view:
        values['movable'][n:n+b]&=(model.body_flags.numpy()&int(newton.BodyFlags.DYNAMIC))!=0
    values['body_com'] = np.concatenate((np.zeros((n, 3), np.float32), body['body_com']))
    values['joints'] = body['joints'] + n
    values['joint_stiffness'] = np.empty((0, 6), np.float32)
    values['inverse_inertia'] = np.concatenate((np.zeros((n, 3, 3), np.float32), model.body_inv_inertia.numpy()))
    values['kind'][:n] = 1
    if model.tet_count: values['kind'][np.unique(values['tets'])] = 0
    # Connected material topology defines object identity. Proximity does not.
    parent = list(range(world))
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for key in ('tets', 'triangles'):
        for row in values[key]:
            first = root(int(row[0]))
            for vertex in row[1:]: parent[root(int(vertex))] = first
    groups = {}; descriptions = []
    for i in range(world):
        component = root(i)
        if component not in groups:
            groups[component] = len(groups)
            descriptions.append({'name': f'object_{groups[component]}', 'family': {0: 'softbody', 1: 'cloth', 2: 'rigid'}[int(values['kind'][i])]})
        values['object_id'][i] = groups[component]
    bounds = np.zeros(world, np.float64)
    for shape, owner in enumerate(model.shape_body.numpy()):
        if owner >= 0:
            radius = np.linalg.norm(model.shape_transform.numpy()[shape, :3] - body['body_com'][owner]) + model.shape_collision_radius.numpy()[shape] + model.shape_margin.numpy()[shape]
            bounds[n + owner] = max(bounds[n + owner], radius)
    values['body_radius'] = bounds
    values['reference_length'] = np.asarray(max(np.ptp(values['rest_x'], axis=0).max() if world else 0., 2 * bounds.max() if world else 0., .1))
    tails = dict(rest_x=[0., 0., 0.], rest_q=[0., 0., 0., 1.], mass=0., movable=False, inertia=np.eye(3), kind=2, object_id=len(groups), body_com=[0., 0., 0.], inverse_inertia=np.zeros((3, 3)), body_radius=0.)
    for key, tail in tails.items():
        array = values[key]
        values[key] = np.concatenate((array, np.asarray(tail, dtype=array.dtype).reshape(1, *array.shape[1:])))
    values.update(site_owner=np.empty(0, np.int64), site_local=np.empty((0, 3)), site_radius=np.empty(0), site_margin=np.empty(0), site_mu=np.empty(0), plane_height=np.asarray(0.), contact_ke=np.asarray(0.), contact_kd=np.asarray(0.), friction_epsilon=np.asarray(configuration['friction_epsilon']))
    metadata = {'family': 'mixed', 'official_demo': identifier, 'contract': 'elastic_shape_contact_v33', 'box_sites': [], 'groups': descriptions, 'particle_count': n, 'body_count': b, 'dt': dt, 'solver_configuration': configuration, 'contacts': 'particle_shape_and_rigid_shape', 'contact_alpha': 0., 'friction_epsilon': configuration['friction_epsilon'], 'body_position_representation': 'com', 'tet_min_det': .01, 'cross_step_contact_history': False, 'prescribed_particle_history': 'accepted_pin_positions_v33'}
    if finite_joint_export:metadata['requires_finite_joint_assembly']=True
    if self_contact_export:metadata['requires_self_contact_assembly']=True
    tensors = {key: torch.as_tensor(value, device=device, dtype=torch.bool if np.asarray(value).dtype.kind == 'b' else torch.long if np.asarray(value).dtype.kind in 'iu' else torch.float64) for key, value in values.items()}
    return MixedContactScene(tensors, metadata)


class MixedContactScene(MixedScene):
    def __init__(self, values, metadata):
        super().__init__(values, metadata)
        self.world = self.n - 1
        self.material_object_keys,self.material_object_index=torch.unique(self.object_id,sorted=True,return_inverse=True)
        self.redistribute_structural_energy = True
        self.objective_scene.redistribute_structural_energy = True
        self.objective_scene.movable = self.movable
        self.fixed_particle_ids = torch.where((self.kind < 2) & ~self.movable)[0]
        # Relation encoding for material rows uses the unchanged structural
        # layout. Contact rows have an independent codec and are appended.
        self.elastic_scene = SimpleNamespace(**{**self.values, **vars(self)})
        self.structural_incidence = self.layout.incidence
        self.structural_kind = self.layout.relation_kind
        self.topology_generation = 0; self._graph_signature = None
        empty = torch.empty(0, 2, device=self.device, dtype=torch.long)
        self.candidate_kind = torch.empty(0, device=self.device, dtype=torch.long)
        self.update_graph(empty, torch.empty(0, device=self.device, dtype=torch.bool))

    def all_relations(self, owners, active):
        incidence = torch.cat((self.structural_incidence, torch.nn.functional.pad(owners, (0, 2), value=-1)))
        kinds = torch.cat((self.structural_kind, self.candidate_kind))
        enabled = torch.cat((torch.ones(len(self.structural_kind), device=self.device, dtype=torch.bool), active))
        return incidence, kinds, enabled

    def update_graph(self, owners, active):
        previous=getattr(self,'_graph_device_signature',None)
        values=(owners,active,self.candidate_kind)
        if self._graph_signature is not None and previous is not None and all(a.shape==b.shape for a,b in zip(values,previous)):
            same=all(torch.equal(a,b) for a,b in zip(values,previous))
            if same:return
        signature = (tuple(map(tuple, owners.tolist())), tuple(active.tolist()), tuple(self.candidate_kind.tolist()))
        if self._graph_signature == signature: return
        self._graph_device_signature=tuple(value.detach().clone() for value in values)
        self._graph_signature = signature; self.topology_generation += 1
        incidence, kinds, enabled = self.all_relations(owners, active)
        graph = build_topology(self.kind, self.movable, self.object_id, incidence, kinds, potential_active=enabled, generation=self.topology_generation)
        self.layout = graph.layout
        self.layout.legacy_kind = torch.where(kinds >= 7, 6, kinds)
        # Activity/layout changes need a fresh generation, but not fresh
        # member tensors when the actual partition is exactly unchanged.
        # Keeping these immutable indices permits versioned map reuse.
        same_partition=(getattr(self,'response_topology_cache',False)
                        and torch.equal(self.interaction_id,graph.interaction_id))
        if not same_partition:
            self.interaction_id = graph.interaction_id
            self.interaction_entities = [torch.where(self.interaction_id == label)[0] for label in torch.unique(self.interaction_id).tolist()]
        for key in ('_protected_topology', '_hierarchy_plans', '_transport_plans', '_vector_active_components', '_hierarchy_material_masks'):
            if hasattr(self, key): delattr(self, key)
