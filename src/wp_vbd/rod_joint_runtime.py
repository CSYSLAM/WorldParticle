"""Native rod/finite-joint controls with one accepted history transaction."""
import torch

from .general_joint_export import export_spec,read_controls
from .rod_contact_runtime import RodContactExampleSolver


class RodJointExampleSolver(RodContactExampleSolver):
    def __init__(self,network,scene,native,dt,wp,newton,source_model,*,learned=True):
        self.newton=newton
        super().__init__(network,scene,native,dt,wp,newton,source_model,learned=learned)

    def validate_controls(self,control):
        model=self.source_model;ids=self.scene.rod.source_joint_ids
        q_start=self.wp.to_torch(model.joint_target_q_start).long()[ids]
        qd_start=self.wp.to_torch(model.joint_qd_start).long()[ids]
        for name,start,size in [('joint_target_q',q_start,7),('joint_target_qd',qd_start,6)]:
            indices=start[:,None]+torch.arange(size,device=start.device)
            if not torch.equal(self.wp.to_torch(getattr(control,name))[indices],self.wp.to_torch(getattr(model,name))[indices]):
                raise ValueError('Changing rod rest material requires rebuilding its history: '+name)
        effort=self.wp.to_torch(control.joint_f)
        owned=self.scene.joint_spec.dof_index
        owned=owned[owned>=0]
        unsupported=torch.ones(len(effort),device=effort.device,dtype=torch.bool);unsupported[owned]=False
        if bool(effort[unsupported].any()):raise ValueError('Only finite joint scalar efforts are supported; rod/FREE efforts need a separate force contract')

    def refresh(self,control):
        super().refresh(control)
        spec=export_spec(self.source_model,self.scene.metadata['solver_configuration'],self.newton,
                         str(self.scene.device),allow_rods=True)
        for key in ('owners','source_joint_ids','frames','angular_mode','linear_axes','angular_axes',
                    'coordinate_offset','rest_parent','rest_child','dof_index','target_index'):
            if not torch.equal(getattr(spec,key),getattr(self.scene.joint_spec,key)):
                raise ValueError('Finite joint coordinate changed; rebuild its accepted winding history: '+key)
        self.scene.joint_spec=spec
        self.scene.joint_controls=read_controls(spec,self.source_model.control() if control is None else control)
