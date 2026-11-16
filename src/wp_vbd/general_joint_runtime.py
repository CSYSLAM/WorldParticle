"""Native rigid joint/contact inference with explicit history commit and rollback."""
import numpy as np
import torch
from . import math3d as so3
from .data import State
from .general_joint_export import export_spec,read_controls
from .rigid_contact_runtime import ContactExampleSolver


class JointExampleSolver(ContactExampleSolver):
    def __init__(self,model,scene,initial,dt,wp,newton,source_model,*,learned=True):
        super().__init__(model,scene,initial,dt,wp,newton,source_model,learned=learned)
        self.source_model=source_model;self.newton=newton
        self.history={key:value.clone() for key,value in initial.history.items()}
        self.configuration=scene.metadata['solver_configuration']
        self.last_input=None;self.last_output=None
        self.asset_snapshot={name:getattr(source_model,name).numpy().copy() for name in ('body_mass','body_inertia','body_inv_inertia','body_inv_mass','body_com','body_flags','body_q','shape_type','shape_body','shape_transform','shape_scale','shape_margin','shape_collision_radius','shape_flags')}
        if scene.metadata.get('world_gravity_contract'):
            for name in ('body_world','shape_world'):self.asset_snapshot[name]=getattr(source_model,name).numpy().copy()

    def read(self,state):
        current=super().read(state)
        return State(current.x,current.q,current.v,current.omega,{key:value.clone() for key,value in self.history.items()})

    def refresh(self,control):
        for name,value in self.asset_snapshot.items():
            if not np.array_equal(getattr(self.source_model,name).numpy(),value):raise ValueError(f'Asset property {name} changed; rebuild the physical scene and collision oracle')
        if self.scene.metadata.get('world_gravity_contract'):
            from .packed_world import refresh
            refresh(self.scene,self.source_model,self.wp)
        else:
            gravity=self.wp.to_torch(self.source_model.gravity).reshape(-1,3)
            if len(gravity)!=1 or not bool(torch.isfinite(gravity).all()):raise ValueError('Invalid single-world gravity')
            self.scene.gravity=gravity[0].double();self.scene.values['gravity']=self.scene.gravity
        spec=export_spec(self.source_model,self.configuration,self.newton,str(self.scene.device),allow_mimic=self.scene.mimic_spec is not None)
        # A topology/frame/axis mutation needs a new asset initialization;
        # retaining an angle measured in another coordinate would be wrong.
        for key in ('owners','frames','angular_mode','linear_axes','angular_axes','coordinate_offset','rest_parent','rest_child'):
            if not torch.equal(getattr(spec,key),getattr(self.scene.joint_spec,key)):raise ValueError(f'Joint coordinate property {key} changed; rebuild the scene/history')
        self.scene.joint_spec=spec
        if self.scene.mimic_spec is not None:
            from .mimic_constraint import export_spec as export_mimic
            mimic=export_mimic(self.source_model,spec,self.newton,str(self.scene.device))
            if not torch.equal(mimic.pairs,self.scene.mimic_spec.pairs):raise ValueError('Mimic ownership changed; rebuild the scene')
            self.scene.mimic_spec=mimic
        self.scene.joint_controls=read_controls(spec,control)

    @torch.no_grad()
    def step(self,state_in,state_out,control,contacts,dt,*,observables=None):
        from . import contact_observables as readout
        readout.validate(self,observables,contacts)
        if dt!=self.dt:raise ValueError('Authored joint timestep changed')
        self.refresh(control)
        start=self.read(state_in);self.last_input=start.detach()
        self.scene.candidates=self.oracle.query(start);self.scene.geometry_oracle=self.oracle
        wrench=self.wp.to_torch(state_in.body_f).double();zero=start.x.new_zeros(1,1,3)
        force=torch.cat((wrench[:,:3][None],zero),1);torque=torch.cat((wrench[:,3:][None],zero),1)
        output,diagnostic=self.network(self.scene,start,dt,learned=self.learned,force=force,torque=torque)
        self.last_diagnostic=diagnostic;self.last_output=output.detach()
        if not bool(diagnostic['accepted'].all()):raise RuntimeError(f'Joint proposal rejected at substep {self.steps}: {diagnostic["iterations"]}')
        prepared=readout.prepare(self,diagnostic,contacts,observables)
        n=self.scene.world;pose=self.wp.to_torch(state_out.body_q);velocity=self.wp.to_torch(state_out.body_qd)
        prescribed_pose=self.wp.to_torch(state_in.body_q)[self.scene.prescribed_ids].clone() if self.scene.metadata.get('prescribed_pose_history') else None
        pose[:,:3].copy_(output.x[0,:n]-so3.rotate(output.q[0,:n],self.scene.body_com[:n]));pose[:,3:].copy_(output.q[0,:n])
        if self.scene.metadata.get('prescribed_pose_history'):
            ids=self.scene.prescribed_ids
            pose[ids]=prescribed_pose
        velocity[:,:3].copy_(output.v[0,:n]);velocity[:,3:].copy_(output.omega[0,:n])
        readout.commit(self,prepared,contacts,observables)
        self.history={key:value.clone() for key,value in output.history.items()};self.steps+=1
        if self.on_substep is not None:self.on_substep(state_out)
