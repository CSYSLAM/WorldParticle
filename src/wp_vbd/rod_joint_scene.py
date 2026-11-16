"""One COM ownership space for rod, finite joints and analytic contacts."""
import numpy as np
import torch

from .general_joint_export import export_spec, read_controls
from .general_joint_scene import JointContactScene
from .rod_contact_scene import RodContactScene, export_components

CONTRACT='rod_finite_joint_contact_com_v143'
JOINT_CONTRACT='objective_finite_and_rod_v143'


def export_model(model,configuration,identifier,newton,dt,device='cpu',*,contact_policy=None,bounded_analytic_buffers=False,
                 sdf_mesh_contacts=False,massless_prescribed=False):
    values,metadata,rod=export_components(model,configuration,identifier,newton,dt,
        contact_policy=contact_policy,allow_general_joints=True,prescribed_motion=True,
        bounded_analytic_buffers=bounded_analytic_buffers,sdf_mesh_contacts=sdf_mesh_contacts,
        massless_prescribed=massless_prescribed)
    spec=export_spec(model,configuration,newton,device,allow_rods=True)
    if not len(spec.owners):raise ValueError('The composed adapter requires finite joints as well as rods')
    # Every authored non-FREE relation must have exactly one constitutive
    # owner; extracting two subsets must never silently omit a joint.
    ids=np.concatenate((rod['source_joint_ids'],spec.source_joint_ids.cpu().numpy()))
    expected=np.flatnonzero(model.joint_type.numpy()!=int(newton.JointType.FREE))
    if len(np.unique(ids))!=len(ids) or not np.array_equal(np.sort(ids),expected):
        raise ValueError('Rod/finite-joint partition does not cover the authored asset')
    metadata.update(contract=CONTRACT,rod_finite_joint_contract=JOINT_CONTRACT,
        joint_history_contract='single_axis_winding_v26',
        prescribed_rod_pose_history='accepted_full_com_pose_v143',
        scope='Owned analytic rods, finite BALL/FIXED/REVOLUTE/PRISMATIC/D6 joints and explicit prescribed COM poses in one response. No mesh, particles, mimic or external solver ownership.')
    if sdf_mesh_contacts:
        metadata['scope']='Experimental owned rod/finite-joint COM assembly with explicit native SDF-mesh local witness contacts and accepted boundary poses. Mesh queries, force fidelity, training and native task quality require separate validation; no particles, mimic or external ownership.'
    scene=RodJointContactScene(values,metadata,rod,spec,device)
    scene.joint_controls=read_controls(spec,model.control())
    return scene


class RodJointContactScene(RodContactScene):
    joint_coordinates=JointContactScene.joint_coordinates
    joint_potential=JointContactScene.joint_potential

    def __init__(self,values,metadata,rod,spec,device='cpu'):
        self.joint_spec=spec;self.mimic_spec=None
        super().__init__(values,metadata,rod,device)
        from .prescribed_motion import configure
        configure(self,prescribed_ids=torch.tensor(metadata['prescribed_body_ids'],device=self.device,dtype=torch.long))
        # Native mass/inertia/flags stay intact. Only effective response
        # mobilities vanish for commanded bodies with positive native mass.
        self.rod.rod_inverse_mass=self.rod.rod_inverse_mass.clone()
        self.rod.rod_inverse_inertia=self.rod.rod_inverse_inertia.clone()
        self.rod.rod_inverse_mass[~self.movable[:-1]]=0.
        self.rod.rod_inverse_inertia[~self.movable[:-1]]=0.
        self._prepare_forest()

    def all_relations(self,owners,active):
        ids,kinds,enabled=super().all_relations(owners,active)
        spec=self.joint_spec;n=len(spec.owners)
        return (torch.cat((ids,spec.owners.repeat(3,1))),
                torch.cat((kinds,torch.tensor([3,11,12],device=self.device).repeat_interleave(n))),
                torch.cat((enabled,self.joint_potential() & self.movable[spec.owners].any(-1).repeat(3))))
