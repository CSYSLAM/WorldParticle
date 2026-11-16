"""Native particle/body I/O and independent geometry-only mixed queries."""
import copy

import torch
from newton.solvers import SolverBase

from . import math3d as so3
from .data import State
from .mixed_collision import MixedContacts
from .mixed_contact_system import validate_history, imposed_path_feasible
from .native_array import native_tensor
from .coupled_history import CoupledStepTransaction


class MixedCollisionOracle:
    def __init__(self, model, scene, wp, newton, *, motion_budget=None, full_surface=False):
        self.wp = wp; self.scene = scene; self.source_model = model
        self.motion_budget = .05 * scene.length if motion_budget is None else float(motion_budget)
        if not torch.isfinite(torch.tensor(self.motion_budget)) or self.motion_budget <= 0:
            raise ValueError('Finite positive detection motion budget required')
        self.query_model = copy.copy(model)
        # ModelView stores overrides in a dictionary. A shallow view copy
        # otherwise changes the source view when query padding is assigned.
        for name in ('_overrides','_cache'):
            value=vars(model).get(name)
            if isinstance(value,dict):object.__setattr__(self.query_model,name,value.copy())
        padding = self.motion_budget / .85
        self.query_model.shape_gap = wp.array(model.shape_gap.numpy() + padding, dtype=wp.float32, device=model.device)
        # Soft narrow-phase uses its scalar gap directly; unlike rigid pairs
        # it does not sum two shape gaps. Cover BOTH endpoints' travel.
        self.pipeline = newton.CollisionPipeline(self.query_model, deterministic=True, rigid_contact_max=max(1024, model.rigid_contact_max or 0), soft_contact_gap=2*padding, enable_rigid_soft_full_surface_contact=full_surface, publish_capacity=False)
        self.contacts = self.pipeline.contacts(); self.query_state = self.query_model.state(); self.queries = 0

    def query(self, state):
        if len(state.x) != 1: raise ValueError('Mixed contact runtime requires batch_size=1')
        for name in ('soft_contact_ke', 'soft_contact_kd', 'soft_contact_mu'):
            setattr(self.query_model, name, getattr(self.source_model, name))
        scene = self.scene; n = scene.metadata['particle_count']; b = scene.metadata['body_count']
        if n: self.wp.to_torch(self.query_state.particle_q).copy_(state.x[0, :n].detach())
        if b:
            pose = self.wp.to_torch(self.query_state.body_q); q = state.q[0, n:n+b]
            pose[:, :3].copy_((state.x[0, n:n+b] - so3.rotate(q, scene.body_com[n:n+b])).detach())
            pose[:, 3:].copy_(q.detach())
        self.pipeline.collide(self.query_state, self.contacts); self.queries += 1
        return self.snapshot(state)

    def snapshot(self, state):
        from .contact_order import canonical_contacts
        return canonical_contacts(MixedContacts.snapshot(self.query_model, self.contacts, self.wp, state))

    def motion(self, start, trial):
        angle = so3.log(so3.multiply(trial.q, so3.conjugate(start.q))).norm(dim=-1)
        travel = (trial.x - start.x).norm(dim=-1) + self.scene.body_radius * angle
        valid = torch.isfinite(travel).all(-1) & (travel[:, self.scene.movable] <= self.motion_budget).all(-1)
        return valid, travel


class MixedContactExampleSolver(SolverBase):
    def __init__(self, network, scene, dt, wp, newton, source_model, *, learned=True, oracle_type=MixedCollisionOracle):
        super().__init__(source_model)
        self.network = network; self.scene = scene; self.dt = dt; self.wp = wp; self.learned = learned
        from .self_contact_repair import configure
        configure(scene,network.config)
        self.source_model = source_model; self.steps = 0; self.last_diagnostic = None; self.on_substep = None
        self.oracle = oracle_type(source_model, scene, wp, newton)
        self.history = {'pin_x_prev': scene.rest_x[scene.fixed_particle_ids][None].clone(), 'pin_initialized': torch.zeros(1, device=scene.device, dtype=torch.bool)}
        self._coupling_transaction=CoupledStepTransaction(scene.n)
        self._coupling_active=False;self._coupling_restarts_enabled=False;self.coupling_step_attempts=0
        names = ('particle_flags', 'particle_mass', 'particle_inv_mass', 'particle_radius', 'body_flags', 'body_mass', 'body_inv_mass', 'body_com', 'body_inertia', 'body_inv_inertia', 'shape_body', 'shape_type', 'shape_transform', 'shape_scale', 'shape_world', 'particle_world', 'body_world', 'tri_indices', 'tri_poses', 'tri_materials', 'tet_indices', 'tet_poses', 'tet_materials', 'edge_indices', 'edge_bending_properties')
        self.static_source = {name: native_tensor(wp, getattr(source_model, name)).clone() for name in names if getattr(source_model, name, None) is not None}

    def coupling_notify_iteration_restarts(self,enabled):
        self._coupling_restarts_enabled=bool(enabled)

    def coupling_notify_input_state_update(self,state,flags,*,iteration_restart=False,dt=0.):
        del state,flags
        if dt<=0:
            if iteration_restart:raise ValueError('A coupling restart needs a positive physical entry interval')
            if self.steps:raise NotImplementedError('Coupled reset after advancement needs an explicit restored material history')
            return
        transaction=self._coupling_transaction
        if iteration_restart:
            if not self._coupling_restarts_enabled:raise ValueError('Coupling restarts were not enabled by the caller')
            self.history,self.steps=transaction.restart(dt)
        elif not transaction.has_baseline or transaction.complete or transaction.phase=='aborted':
            transaction.begin(self.history,self.steps,dt)
        elif dt!=transaction.dt:raise ValueError('Coupling input changed the current physical entry interval')
        self._coupling_active=True

    def enable_proxy_feedback(self,proxy_mask,owned_response):
        if self._coupling_active:raise ValueError('Configure physical proxy ownership before the first coupling entry')
        self._coupling_transaction=CoupledStepTransaction(self.scene.n,proxy_mask=proxy_mask,owned_response=owned_response)
        self.scene.request_physical_load_packets=True

    def _proxy_output(self,mapping,output,begin,count,dt,*,particle=False):
        transaction=self._coupling_transaction
        if not self._coupling_active or dt!=transaction.dt:raise ValueError('Proxy readout must match the current physical entry')
        value=transaction.feedback()[0,begin:begin+count]
        index=native_tensor(self.wp,mapping).long();target=native_tensor(self.wp,output)
        if index.shape!=(count,) or bool(((index< -1)|(index>=len(target))).any()):raise ValueError('Native proxy mapping changed')
        selected=index>=0
        if bool((selected&~transaction.proxy_mask[begin:begin+count]).any()):raise ValueError('Native mapping names a non-proxy owner')
        if particle and bool(value[:,3:].any()):raise ValueError('A particle proxy cannot discard angular load')
        result=target.new_zeros(target.shape)
        values=value[:,:3] if particle else value
        if len(index[selected]):result.index_add_(0,index[selected],values[selected].to(result))
        target.copy_(result)

    def coupling_harvest_proxy_wrenches(self,body_local_to_proxy_global,out_body_f,*,body_qd_before,state,state_out,contacts,dt):
        del body_qd_before,state,state_out,contacts
        self._proxy_output(body_local_to_proxy_global,out_body_f,self.scene.metadata['particle_count'],self.scene.metadata['body_count'],dt)

    def coupling_harvest_proxy_particle_forces(self,particle_local_to_proxy_global,out_particle_f,*,particle_qd_before,state,state_out,contacts,dt):
        del particle_qd_before,state,state_out,contacts
        self._proxy_output(particle_local_to_proxy_global,out_particle_f,0,self.scene.metadata['particle_count'],dt,particle=True)

    def set_rigid_history_update(self, update):
        if not bool(update): raise ValueError('Retained contact caches require a separate contract')

    def refresh_control(self, control):
        if control is not None and getattr(control, 'joint_f', None) is not None and bool(self.wp.to_torch(control.joint_f).any()): raise ValueError('Free-joint generalized efforts require an explicit force adapter')

    def validate_start(self, start):
        validate_history(self.scene, start)

    def read(self, native):
        scene = self.scene; n = scene.metadata['particle_count']; b = scene.metadata['body_count']
        x = scene.rest_x[None].clone(); q = scene.rest_q[None].clone()
        v = torch.zeros_like(x); omega = torch.zeros_like(x)
        if n:
            x[0, :n] = self.wp.to_torch(native.particle_q).double()
            v[0, :n] = self.wp.to_torch(native.particle_qd).double()
        if b:
            pose = self.wp.to_torch(native.body_q).double(); rotation = so3.normalize(pose[:, 3:])
            if not bool((pose[:, 3:].norm(dim=-1) > 1e-12).all()): raise ValueError('Degenerate rigid orientation')
            x[0, n:n+b] = pose[:, :3] + so3.rotate(rotation, scene.body_com[n:n+b]); q[0, n:n+b] = rotation
            velocity = self.wp.to_torch(native.body_qd).double(); v[0, n:n+b] = velocity[:, :3]; omega[0, n:n+b] = velocity[:, 3:]
        result = State(x, q, v, omega, self.history)
        if any(not bool(torch.isfinite(value).all()) for value in (x, q, v, omega)): raise ValueError('Nonfinite native mixed state')
        return result

    @torch.no_grad()
    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        saved={}
        if self._coupling_active:
            self.coupling_step_attempts+=1
            for name in ('particle_q','particle_qd','body_q','body_qd'):
                value=getattr(state_out,name,None)
                if value is not None:saved[name]=native_tensor(self.wp,value).clone()
        try:return self._step_impl(state_in,state_out,control,contacts,dt,observables=observables)
        except Exception:
            if self._coupling_active:
                self.history,self.steps=self._coupling_transaction.abort()
                for name,value in saved.items():native_tensor(self.wp,getattr(state_out,name)).copy_(value)
            raise
        finally:
            policy=getattr(self,'query_budget_policy',None)
            if policy is not None:policy.finish_attempt()

    def _step_impl(self, state_in, state_out, control, contacts, dt, *, observables=None):
        if observables is not None: raise ValueError('Mixed contact wrench export needs an explicit native readout contract')
        if dt != self.dt: raise ValueError('Authored mixed timestep changed')
        for name, expected in self.static_source.items():
            if not torch.equal(native_tensor(self.wp, getattr(self.source_model, name)), expected): raise ValueError(f'Mixed static source changed; rebuild required: {name}')
        gravity = self.wp.to_torch(self.source_model.gravity).double()
        if gravity.shape != (1, 3) or not bool(torch.isfinite(gravity).all()): raise ValueError('Invalid mixed gravity layout')
        self.refresh_control(control)
        start = self.read(state_in); self.validate_start(start)
        ids = self.scene.fixed_particle_ids
        travel = (start.x[:, ids] - self.history['pin_x_prev']).norm(dim=-1)
        if bool(self.history['pin_initialized'].any()) and bool((travel > self.oracle.motion_budget).any()): raise ValueError('Imposed particle motion exceeds detection envelope')
        if not getattr(self.scene,'learned_contact_repair',False) and not bool(imposed_path_feasible(self.scene,start).all()): raise ValueError('Imposed particle path collapses or reverses a material element')
        force = torch.zeros_like(start.x); torque = torch.zeros_like(start.x)
        n = self.scene.metadata['particle_count']; b = self.scene.metadata['body_count']
        if n: force[0, :n] = self.wp.to_torch(state_in.particle_f).double()
        if b:
            wrench = self.wp.to_torch(state_in.body_f).double(); force[0, n:n+b] = wrench[:, :3]; torque[0, n:n+b] = wrench[:, 3:]
        if not bool(torch.isfinite(force).all() & torch.isfinite(torque).all()): raise ValueError('Nonfinite mixed external force')
        self.scene.values['gravity'] = gravity[0]
        policy=getattr(self,'query_budget_policy',None)
        if policy is not None:
            try:policy.prepare(start,force,torque,dt)
            except Exception:self.last_diagnostic=None;raise
        try:candidates = self.oracle.query(start)
        except Exception:
            if policy is not None:policy.queried(False)
            raise
        if policy is not None:policy.queried(True)
        self.scene.candidates = candidates; self.scene.candidate_kind = candidates.relation_kind; self.scene.geometry_oracle = self.oracle
        result, diagnostic = self.network(self.scene, start, dt, learned=self.learned, force=force, torque=torque)
        if policy is not None:policy.check_frozen()
        self.last_diagnostic = diagnostic
        if not bool(diagnostic['accepted'].all()):
            summary=[]
            for row in diagnostic['iterations']:
                selected=torch.as_tensor(row['selected'],device=self.scene.device)
                summary.append({**{k:v for k,v in row.items() if k!='selected'},'rejected_mobile_entities':int((~selected&self.scene.movable[None]).sum())})
            raise RuntimeError(f'Mixed contact proposal rejected at substep {self.steps}: {summary}')
        if self._coupling_active:
            self._coupling_transaction.record(result.history,self.steps+1,dt,packets=diagnostic.get('physical_load_packets'))
        if n:
            self.wp.to_torch(state_out.particle_q).copy_(result.x[0, :n]); self.wp.to_torch(state_out.particle_qd).copy_(result.v[0, :n])
        if b:
            pose = self.wp.to_torch(state_out.body_q); velocity = self.wp.to_torch(state_out.body_qd)
            pose[:, :3].copy_(result.x[0, n:n+b] - so3.rotate(result.q[0, n:n+b], self.scene.body_com[n:n+b])); pose[:, 3:].copy_(result.q[0, n:n+b])
            velocity[:, :3].copy_(result.v[0, n:n+b]); velocity[:, 3:].copy_(result.omega[0, n:n+b])
        self.history = {key: value.detach().clone() for key, value in result.history.items()}
        self.steps += 1
        if self.on_substep is not None: self.on_substep(state_out)
