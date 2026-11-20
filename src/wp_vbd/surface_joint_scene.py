"""Finite elastic material, filtered surface contacts and finite joints.

One real particle/body COM index space and one final static world slot.
Joint rows are exported separately, never as duplicate legacy rod energies.
"""
import torch

from .general_joint_export import export_spec
from .general_joint_scene import JointContactScene
from .mixed_contact_scene import MixedContactScene, export_model as export_elastic


CONTRACT = 'elastic_surface_joint_contact_v38'


def export_model(model, configuration, identifier, newton, dt, device='cpu'):
    import warp as wp
    allowed = [int(getattr(newton.GeoType, name)) for name in
               ('PLANE', 'SPHERE', 'BOX', 'CAPSULE', 'CYLINDER', 'CONE', 'ELLIPSOID', 'MESH', 'CONVEX_MESH')]
    collision = int(newton.ShapeFlags.COLLIDE_SHAPES) | int(newton.ShapeFlags.COLLIDE_PARTICLES)
    active = (wp.to_torch(model.shape_flags).long() & collision) != 0
    supported = torch.isin(wp.to_torch(model.shape_type).long(), torch.tensor(allowed, device=model.device.alias))
    if bool((active & ~supported).any()):
        raise ValueError('Surface joint path needs mesh force eligibility before mesh shapes are admitted')
    from .mimic_constraint import export_spec as export_mimic
    spec = export_spec(model, configuration, newton, device, allow_mimic=True)
    mimic = export_mimic(model, spec, newton, device)
    n = model.particle_count
    spec.owners = spec.owners+n
    base = export_elastic(model, configuration, identifier, newton, dt, device, finite_joint_export=True)
    if base.n != n+model.body_count+1 or len(base.joints):
        raise AssertionError('One world slot and no duplicate legacy joint energy required')
    metadata = {**base.metadata, 'contract': CONTRACT, 'mixed_finite_joints': True,
                'contacts': 'analytic_rigid_and_weighted_soft_features',
                'joint_history_contract': 'single_axis_winding_v26',
                'scope': 'Experimental shared elastic/contact/finite-joint path. Ideal scalar mimic and native per-configuration mesh force filtering are explicit. No self-contact, external ownership or full task claim.'}
    # A fixed-size accepted cache is part of the surface solver contract. It
    # is disabled by default so historical recordings and checkpoints remain
    # byte-compatible; new training scenes opt in explicitly.
    if metadata.get('surface_tangent_memory',False):
        metadata['surface_tangent_history_capacity']=int(metadata.get('surface_tangent_history_capacity',8192))
    if len(mimic.pairs): metadata['ideal_mimic_contract'] = 'scalar_mass_tangent_v27'
    if bool((active & (wp.to_torch(model.shape_type).long() == int(newton.GeoType.MESH))).any()):
        metadata['contacts'] = 'analytic_and_native_filtered_mesh_rigid_and_weighted_soft_features'
    return SurfaceJointScene(base.values, metadata, spec, mimic_spec=mimic if len(mimic.pairs) else None)


class SurfaceJointScene(MixedContactScene):
    joint_potential = JointContactScene.joint_potential
    joint_coordinates = JointContactScene.joint_coordinates

    def __init__(self, values, metadata, spec, *, mimic_spec=None):
        if metadata.get('ideal_mimic_contract') and mimic_spec is None:
            raise ValueError('A serialized mimic scene must restore its explicit ideal constraint specification')
        self.joint_spec = spec; self.mimic_spec = mimic_spec
        super().__init__(values, metadata)
        if metadata.get('prescribed_pose_history'):
            from . import prescribed_motion as prescribed
            n,b=metadata['particle_count'],metadata['body_count']
            expected=list(range(n,n+b))
            if not prescribed.enabled(self) or metadata.get('prescribed_rigid_ownership')!='massless_commands_v122':
                raise ValueError('An explicit mixed prescribed-body ownership contract is required')
            ids=metadata.get('prescribed_rigid_node_ids')
            if not isinstance(ids,list) or any(type(i) is not int for i in ids) or ids!=expected or not b:
                raise ValueError('This mixed boundary contract requires every authored massless rigid owner, in source order')
            if bool((self.mass[n:n+b]!=0).any()) or bool(self.movable[n:n+b].any()):
                raise ValueError('Prescribed massless ownership cannot replace a dynamic body')
            prescribed.configure(self,prescribed_ids=torch.tensor(ids,device=self.device,dtype=torch.long))

    def all_relations(self, owners, active):
        if owners.shape[1] > 4: raise ValueError('Surface incidence exceeds four physical owners')
        contact = torch.nn.functional.pad(owners, (0, 4-owners.shape[1]), value=-1)
        joint = torch.nn.functional.pad(self.joint_spec.owners.repeat(3, 1), (0, 2), value=-1)
        kinds = torch.tensor([3, 11, 12], device=self.device).repeat_interleave(len(self.joint_spec.owners))
        incidence = torch.cat((self.structural_incidence, contact, joint))
        kinds = torch.cat((self.structural_kind, self.candidate_kind, kinds))
        enabled = torch.cat((torch.ones(len(self.structural_kind), device=self.device, dtype=torch.bool), active, self.joint_potential()))
        if self.mimic_spec is not None:
            from .mimic_constraint import relation
            ids, mimic_active = relation(self)
            incidence = torch.cat((incidence, ids))
            kinds = torch.cat((kinds, torch.full((len(ids),), 13, device=self.device, dtype=torch.long)))
            enabled = torch.cat((enabled, mimic_active))
        return incidence, kinds, enabled

    def update_graph(self, owners, active):
        current=self.joint_potential()
        previous=getattr(self,'_joint_potential_device',None)
        if previous is not None and torch.equal(current,previous):
            potential=self._joint_graph_signature
        else:
            potential = tuple(current.tolist())
            self._joint_potential_device=current.detach().clone()
        if getattr(self, '_joint_graph_signature', None) != potential:
            self._graph_signature = None; self._joint_graph_signature = potential
        super().update_graph(owners, active)
        kinds = self.layout.relation_kind
        self.layout.legacy_kind = torch.where((kinds == 11) | (kinds == 12) | (kinds == 13), 3,
                                              torch.where(kinds >= 7, 6, kinds))
