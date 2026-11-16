"""Owned FREE/ROD COM scene with material relations and fresh shape contacts."""
from types import SimpleNamespace

import numpy as np
import torch

from .rigid_contact_scene import RigidContactScene
from .rod_adapter import export_model as export_rods
from .unified_graph import build_topology

CONTRACT = 'rod_shape_contact_com_v40'


def export_components(model, configuration, identifier, newton, dt, *, contact_policy=None,
                      allow_general_joints=False, prescribed_motion=False,bounded_analytic_buffers=False,
                      sdf_mesh_contacts=False,massless_prescribed=False):
    if type(sdf_mesh_contacts) is not bool or type(massless_prescribed) is not bool:
        raise ValueError('Explicit boolean mesh and massless-boundary contracts required')
    if massless_prescribed and not prescribed_motion:
        raise ValueError('Massless boundary motion needs explicit accepted prescribed-pose history')
    if sdf_mesh_contacts and (contact_policy is not None or bounded_analytic_buffers):
        raise ValueError('SDF witnesses require fresh queries without analytic-only capacity or matched history')
    if configuration.get('rigid_contact_history') and contact_policy is None:
        raise ValueError('Cross-step contact cache needs its explicit history contract')
    if contact_policy is not None and not configuration.get('rigid_contact_history'):
        raise ValueError('Matched contact history must be explicitly authored')
    flags=model.body_flags.numpy()
    admitted_flags=[int(newton.BodyFlags.DYNAMIC)]
    if prescribed_motion:admitted_flags.append(int(newton.BodyFlags.KINEMATIC))
    if np.any(~np.isin(flags,admitted_flags)):
        raise ValueError('Proxy/flagged kinematic ownership needs its explicit adapter')
    allowed = [int(getattr(newton.GeoType, key)) for key in
               ('PLANE', 'SPHERE', 'BOX', 'CAPSULE', 'CYLINDER', 'CONE', 'ELLIPSOID')]
    if sdf_mesh_contacts:
        allowed.append(int(newton.GeoType.MESH))
        if not np.any(model.shape_type.numpy()==int(newton.GeoType.MESH)):
            raise ValueError('Explicit SDF mesh contact path requires authored mesh geometry')
    if np.any(~np.isin(model.shape_type.numpy(), allowed)):
        raise ValueError('First rod/contact path admits analytic shapes only')
    rod, rod_metadata = export_rods(model, configuration, identifier, newton, dt,
                                   allow_general_joints=allow_general_joints)
    for name in ('joint_stiffness', 'rod_damping', 'rod_dahl_eps', 'rod_dahl_tau'):
        if not np.isfinite(rod[name]).all() or np.any(rod[name] < 0):
            raise ValueError('Finite nonnegative rod material required: '+name)
    values = {key: getattr(model, native).numpy().copy() for key, native in
              (('mass', 'body_mass'), ('inertia', 'body_inertia'), ('inverse_inertia', 'body_inv_inertia'),
               ('inverse_mass', 'body_inv_mass'), ('body_com', 'body_com'), ('rest_pose', 'body_q'))}
    values['gravity'] = rod['gravity']; values['reference_length'] = rod['reference_length']
    bounds = np.zeros(model.body_count, np.float64)
    for body, transform, radius, margin in zip(model.shape_body.numpy(), model.shape_transform.numpy(),
                                              model.shape_collision_radius.numpy(), model.shape_margin.numpy()):
        if body >= 0:
            bounds[body] = max(bounds[body], np.linalg.norm(transform[:3]-values['body_com'][body])+radius+margin)
    values['body_radius'] = bounds
    alpha = configuration.get('rigid_avbd_contact_alpha')
    if alpha is None: alpha = configuration.get('rigid_avbd_alpha') or 0.
    metadata = {**rod_metadata, 'contract': CONTRACT, 'history_contract': 'rod_contact_transaction_v40',
                'body_position_representation': 'com', 'contact_alpha': alpha,
                'friction_epsilon': configuration['friction_epsilon'],
                'contacts': 'fresh_analytic_rigid_shape', 'cross_step_contact_history': False,
                'scope': 'FREE/ROD forest and owned analytic contacts; no retained collision cache, finite non-ROD joints or external ownership.'}
    if contact_policy is not None:
        from .rigid_contact_memory import validate_policy
        from .rigid_contact_capacity import capacity
        metadata.update(cross_step_contact_history=True,contact_matching_policy=validate_policy(contact_policy),
                        history_contract='rod_matched_contact_transaction_v133',contacts='matched_analytic_rigid_shape',
                        scope='Owned analytic FREE/ROD forest, explicit accepted matched contact/rod history; no finite general joints or external ownership.')
        metadata['contact_history_capacity'],metadata['contact_capacity_policy']=capacity(model,SimpleNamespace(metadata=metadata),newton)
    if prescribed_motion:
        values['body_flags']=flags.copy()
        prescribed=flags==int(newton.BodyFlags.KINEMATIC)
        if massless_prescribed:
            mass=values['mass'];inverse_mass=values['inverse_mass']
            if not np.isfinite(mass).all() or not np.isfinite(inverse_mass).all() or np.any(mass<0) or np.any(inverse_mass<0) or np.any((mass==0)&(inverse_mass!=0)):
                raise ValueError('Finite nonnegative consistent massless boundary data required')
            prescribed=prescribed|((mass==0)&(inverse_mass==0))
            metadata['prescribed_body_policy']='native_kinematic_or_zero_mass_zero_inverse_mass_v195'
        metadata['prescribed_body_ids']=np.flatnonzero(prescribed).tolist()
    if sdf_mesh_contacts:
        metadata.update(contacts='fresh_sdf_mesh_rigid_shape',
            rigid_mesh_contact_contract='frozen_native_local_witnesses_v195',
            mesh_shape_ids=np.flatnonzero(model.shape_type.numpy()==int(newton.GeoType.MESH)).tolist())
    if bounded_analytic_buffers:
        shapes=[int(getattr(newton.GeoType,name)) for name in ('CAPSULE','SPHERE','PLANE')]
        if not np.isin(model.shape_type.numpy(),shapes).all():
            raise ValueError('Bounded analytic buffers require capsule/sphere/plane geometry')
        if contact_policy is None:raise ValueError('Bounded rod buffers require complete explicit accepted contact history')
        metadata.update(bounded_analytic_contact_buffers='current_expanded_aabb_v146',
                        initial_contact_history_storage_capacity=1)
    return values, metadata, rod


def export_model(model, configuration, identifier, newton, dt, device='cpu', *, contact_policy=None,bounded_analytic_buffers=False):
    values, metadata, rod=export_components(model,configuration,identifier,newton,dt,contact_policy=contact_policy,
                                           bounded_analytic_buffers=bounded_analytic_buffers)
    return RodContactScene(values, metadata, rod, device)


class RodContactScene(RigidContactScene):
    def __init__(self, values, metadata, rod, device='cpu'):
        integer = {'joints', 'source_joint_ids', 'object_id'}
        keys = ('joints', 'source_joint_ids', 'object_id', 'anchor_p', 'anchor_c', 'anchor_qp', 'anchor_qc',
                'rod_rest_bend', 'rod_rest_twist', 'joint_stiffness', 'rod_damping',
                'rod_dahl_eps', 'rod_dahl_tau', 'rod_inverse_mass', 'rod_inverse_inertia')
        self.rod = SimpleNamespace(**{key: torch.as_tensor(rod[key], device=device,
                                     dtype=torch.long if key in integer else torch.float64) for key in keys})
        super().__init__(values, metadata, device)
        self._prepare_forest()

    def _prepare_forest(self):
        source_parents = list(range(self.n)); incoming = set(); edges=self.rod.joints.cpu().tolist()
        adjacency=[[] for _ in range(self.n)]
        for row,(parent, child) in enumerate(edges):
            if child in incoming or parent == child:
                raise ValueError('ROD material prefix requires a directed forest')
            incoming.add(child); source_parents[child] = parent
            adjacency[parent].append((child,row));adjacency[child].append((parent,row))
        for node in range(self.n):
            seen = set(); current = node
            while source_parents[current] != current:
                if current in seen: raise ValueError('Cyclic rod material modes require a separate cyclic basis')
                seen.add(current); current = source_parents[current]
        # Only basis integration is re-rooted. Constitutive owners, material
        # frames, force pullbacks and graph rows retain their authored order.
        parents=list(range(self.n));integration_parent=[0]*len(edges);integration_child=[0]*len(edges)
        child_role=[1]*len(edges);order=[];visited=set();mobile=self.movable.cpu().tolist()
        self.rod_boundary_plans=[]
        for seed in range(self.n):
            if seed in visited:continue
            component=[];stack=[seed];discovered={seed}
            while stack:
                node=stack.pop();component.append(node)
                for other,_ in adjacency[node]:
                    if other not in discovered:discovered.add(other);stack.append(other)
            pins=[node for node in component if not mobile[node]]
            root=pins[0] if pins else next(node for node in component if source_parents[node]==node)
            queue=[root];visited.add(root)
            for node in queue:
                for other,row in adjacency[node]:
                    if other in visited:continue
                    visited.add(other);queue.append(other);parents[other]=node
                    integration_parent[row]=node;integration_child[row]=other
                    child_role[row]=int(edges[row][1]==other);order.append(row)
            if len(pins)>1:
                from .rod_boundary_ports import description
                self.rod_boundary_plans.append(description(component,parents,pins,self.rest_x.cpu().numpy(),self.device))
        self.rod_parent = torch.tensor(parents, device=self.device)
        self.rod_integration_parent=torch.tensor(integration_parent,device=self.device)
        self.rod_integration_child=torch.tensor(integration_child,device=self.device)
        self.rod_integration_child_role=torch.tensor(child_role,device=self.device)
        self.rod_integration_order=order
        self.rod_original_direction=all(role==1 for role in child_role)
        ancestors = self.rod_parent; self.rod_prefix_ancestors = []
        for _ in range(max(1, (self.n-1).bit_length())):
            self.rod_prefix_ancestors.append(ancestors)
            ancestors = ancestors[ancestors]

    def all_relations(self, owners, active):
        kinds = torch.cat((torch.full((len(owners),), 7, device=self.device, dtype=torch.long),
                           torch.full((len(self.rod.joints),), 4, device=self.device, dtype=torch.long)))
        material = (self.rod.joint_stiffness+self.rod.rod_damping).sum(-1) > 0
        return torch.cat((owners, self.rod.joints)), kinds, torch.cat((active, material))

    def update_graph(self, owners, active):
        self.kind = torch.full_like(self.kind, 2);self.kind[self.rod.joints.flatten()]=3
        self.object_id = torch.cat((self.rod.object_id, self.rod.object_id.new_tensor([int(self.rod.object_id.max())+1])))
        owners, kinds, enabled = self.all_relations(owners, active)
        signature = (tuple(map(tuple, owners.tolist())), tuple(enabled.tolist()))
        if self._graph_signature == signature: return
        self._graph_signature = signature; self.topology_generation += 1
        graph = build_topology(self.kind, self.movable, self.object_id, torch.nn.functional.pad(owners, (0, 2), value=-1),
                               kinds, potential_active=enabled, generation=self.topology_generation)
        self.layout = graph.layout; self.layout.legacy_kind = torch.where(kinds == 7, 6, 3)
        self.interaction_id = graph.interaction_id
        self.interaction_entities = [torch.where(self.interaction_id == key)[0] for key in torch.unique(self.interaction_id).tolist()]
        for key in ('_protected_topology', '_hierarchy_plans', '_transport_plans', '_vector_active_components', '_hierarchy_material_masks'):
            if hasattr(self, key): delattr(self, key)

    def initial_state(self, model, wp):
        from .rod_contact_system import initial_history
        from .data import State
        state = super().initial_state(model, wp)
        return State(state.x, state.q, state.v, state.omega, initial_history(self, state))
