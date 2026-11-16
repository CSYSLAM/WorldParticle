"""Newton State I/O for the shared model, without importing a VBD solver."""
import torch

from . import math3d as so3
from .data import State
from .runtime import CapturedStepper


class LearnedExampleSolver:
    def __init__(self,model,scene,initial,dt,wp,*,learned=True,capture=True):
        self.model=model;self.scene=scene;self.dt=dt;self.wp=wp
        self.learned=learned
        self.graph=CapturedStepper(model,scene,initial,dt,learned=learned) if capture else None
        self.steps=0;self.last_diagnostic=None
        self.on_substep=None
        self.history=None if initial.history is None else {key:value.clone() for key,value in initial.history.items()}
        self.device=wp.get_device(str(scene.device))

    def read(self,state):
        scene=self.scene;n=scene.metadata['particle_count'];b=scene.metadata['body_count']
        x=scene.rest_x[None].clone();q=scene.rest_q[None].clone();v=torch.zeros_like(x);omega=torch.zeros_like(x)
        if n:x[0,:n]=self.wp.to_torch(state.particle_q);v[0,:n]=self.wp.to_torch(state.particle_qd)
        if b:
            pose=self.wp.to_torch(state.body_q);twist=self.wp.to_torch(state.body_qd)
            q[0,n:n+b]=pose[:,3:]
            x[0,n:n+b]=pose[:,:3] if scene.metadata.get('body_position_representation')=='origin' else pose[:,:3]+so3.rotate(pose[:,3:],scene.body_com[n:n+b])
            v[0,n:n+b]=twist[:,:3];omega[0,n:n+b]=twist[:,3:]
        return State(x,q,v,omega,self.history)

    def write(self,state,value):
        scene=self.scene;n=scene.metadata['particle_count'];b=scene.metadata['body_count']
        if n:
            self.wp.to_torch(state.particle_q).copy_(value.x[0,:n]);self.wp.to_torch(state.particle_qd).copy_(value.v[0,:n])
        if b:
            pose=self.wp.to_torch(state.body_q);twist=self.wp.to_torch(state.body_qd)
            pose[:,:3]=value.x[0,n:n+b] if scene.metadata.get('body_position_representation')=='origin' else value.x[0,n:n+b]-so3.rotate(value.q[0,n:n+b],scene.body_com[n:n+b])
            pose[:,3:]=value.q[0,n:n+b]
            twist[:,:3]=value.v[0,n:n+b];twist[:,3:]=value.omega[0,n:n+b]

    @torch.no_grad()
    def step(self,state_in,state_out,control,contacts,dt):
        if dt!=self.dt:raise ValueError('The authored timestep changed')
        if self.history is not None and contacts is not None:raise ValueError('Rod contact has not been admitted')
        # Both libraries use this Torch stream (selected by the outer driver).
        current=self.read(state_in);force=torch.zeros_like(current.x);torque=torch.zeros_like(current.omega)
        n=self.scene.metadata['particle_count'];b=self.scene.metadata['body_count']
        if n:force[0,:n]=self.wp.to_torch(state_in.particle_f)
        if b:
            wrench=self.wp.to_torch(state_in.body_f);force[0,n:n+b]=wrench[:,:3];torque[0,n:n+b]=wrench[:,3:]
        if self.graph is None:
            candidate,diagnostic=self.model(self.scene,current,self.dt,learned=self.learned,force=force,torque=torque)
        else:candidate,diagnostic=self.graph.step(current,force,torque)
        self.last_diagnostic=diagnostic
        if not bool(diagnostic['accepted'].all()):raise RuntimeError(f'Learned physical proposal rejected at substep {self.steps}; no reference fallback')
        self.write(state_out,candidate);self.history=candidate.history;self.steps+=1
        if self.on_substep is not None:self.on_substep(state_out)

    def _history_view(self,key):
        from types import SimpleNamespace
        value=self.scene.rest_x.new_zeros(self.scene.metadata['source_joint_count'],3)
        value=value.index_copy(0,self.scene.source_joint_ids,self.history[key][0])
        return SimpleNamespace(numpy=lambda:value.detach().cpu().numpy())

    @property
    def joint_sigma_prev(self):return self._history_view('stress')

    @property
    def joint_kappa_prev(self):return self._history_view('kappa')
