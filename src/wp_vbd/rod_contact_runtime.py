"""Native rod/contact I/O with accepted COM/material history and rollback."""
import torch

from . import math3d as so3
from .data import State
from .rigid_contact_runtime import ContactExampleSolver
from .rod_contact_system import history_keys,validate_history


class RodContactExampleSolver(ContactExampleSolver):
    @property
    def body_body_contact_counts(self):
        """Current owned witness incidence for the original viewer diagnostic."""
        count=self.scene.world
        values=torch.zeros(count,device=self.scene.device,dtype=torch.int32)
        candidates=getattr(self.scene,'candidates',None)
        if candidates is not None:
            owners=candidates.owners.reshape(-1)
            owners=owners[(owners>=0)&(owners<count)]
            values.index_add_(0,owners,torch.ones(len(owners),device=values.device,dtype=values.dtype))
        return self.wp.array(values.cpu().numpy(),dtype=self.wp.int32,device=self.source_model.device)

    @property
    def body_body_contact_buffer_pre_alloc(self):
        # This backend has no separate per-body truncation. Its complete
        # global query buffer is the bound displayed by the source diagnostic.
        return len(self.oracle.contacts.rigid_contact_shape0)

    def __init__(self, network, scene, native, dt, wp, newton, source_model, *, learned=True):
        initial = scene.initial_state(source_model, wp)
        super().__init__(network, scene, initial, dt, wp, newton, source_model, learned=learned)
        self.history = {key: value.clone() for key, value in initial.history.items()}
        self.prescribed_fresh_inputs=bool(network.config.get('rod_prescribed_fresh_inputs',False))
        self.native_float64_state=bool(network.config.get('rod_native_float64_state',False))
        if self.native_float64_state:
            self.history['rod_native_v']=initial.v.clone();self.history['rod_native_omega']=initial.omega.clone()
        self.source_cache_reuse_requests=0
        self.removal_event_contract=bool(network.config.get('rod_prescribed_removal_events',False))
        self.prescribed_event_provider=None;self.geometry_removal_event_steps=[]
        initial = self.read(native)
        self.history['rod_previous_x'] = initial.x.clone(); self.history['rod_previous_q'] = initial.q.clone()
        if self.native_float64_state:
            self.history['rod_native_v']=initial.v.clone();self.history['rod_native_omega']=initial.omega.clone()
        names = ('body_mass', 'body_inv_mass', 'body_inertia', 'body_inv_inertia', 'body_com', 'body_flags',
                 'body_q', 'shape_body', 'shape_type', 'shape_transform', 'shape_scale', 'shape_world', 'body_world',
                 'shape_margin', 'shape_collision_radius', 'shape_flags', 'shape_gap',
                 'joint_type', 'joint_parent', 'joint_child', 'joint_X_p', 'joint_X_c', 'joint_axis',
                 'joint_q_start', 'joint_qd_start', 'joint_target_q', 'joint_target_qd', 'joint_enabled')
        if scene.metadata.get('cross_step_contact_history'):
            # Runtime material/filter edits require rebuilding this accepted
            # cache and the associated material history, never silent reuse.
            names+=('joint_target_ke','joint_target_kd','shape_material_ke','shape_material_kd',
                    'shape_material_mu')
        self.static_filter_pairs=(torch.as_tensor(source_model.shape_collision_filter_pairs.pairs_array().copy())
                                  if scene.metadata.get('cross_step_contact_history') else None)
        self.static_source = {name: wp.to_torch(getattr(source_model, name)).clone() for name in names
                              if getattr(source_model, name, None) is not None}

    def set_rigid_history_update(self, update):
        if self.scene.metadata.get('cross_step_contact_history'):
            if not bool(update):raise ValueError('Matched rod contact history requires a fresh detection every physical step')
            return
        if not self.prescribed_fresh_inputs:
            return super().set_rigid_history_update(update)
        # An explicit policy override: this solver still queries fresh geometry
        # every physical step and owns no retained source contact row state.
        # Matched persistent contact history remains rejected by scene export.
        self.source_cache_reuse_requests += int(not bool(update))

    def read(self, native):
        if not bool((self.wp.to_torch(native.body_q)[:, 3:].norm(dim=-1) > 1e-12).all()):
            raise ValueError('Degenerate native rod/contact orientation')
        state = super().read(native)
        if any(not bool(torch.isfinite(value).all()) for value in (state.x, state.q, state.v, state.omega)):
            raise ValueError('Nonfinite native rod/contact state')
        if not self.native_float64_state:
            return State(state.x, state.q, state.v, state.omega, self.history)
        keys=history_keys(self.scene)
        if set(self.history)!=keys|{'rod_native_v','rod_native_omega'}:
            raise ValueError('Complete restartable native float64 rod state required')
        history={key:value for key,value in self.history.items() if key in keys}
        validate_history(self.scene,State(state.x,state.q,state.v,state.omega,history))
        for key in ('rod_native_v','rod_native_omega'):
            value=self.history[key]
            if value.shape!=state.v.shape or value.dtype!=state.v.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Invalid native float64 rod state: '+key)
        n=self.scene.world;h=self.history;pose=self.wp.to_torch(native.body_q);velocity=self.wp.to_torch(native.body_qd)
        # Reuse the accepted state only when this exact native buffer value is
        # its projection. Changed pose/velocity inputs remain explicit inputs.
        previous_x=h['rod_previous_x'];previous_q=h['rod_previous_q']
        projected_pose=torch.cat((previous_x[0,:n]-so3.rotate(previous_q[0,:n],self.scene.body_com[:n]),previous_q[0,:n]),-1).to(pose.dtype)
        projected_velocity=torch.cat((h['rod_native_v'][0,:n],h['rod_native_omega'][0,:n]),-1).to(velocity.dtype)
        def mask(equal):
            return torch.cat((equal,equal.new_ones(1)))[None,:,None]&h['rod_initialized'][:,None,None]
        same_pose=mask((pose==projected_pose).all(-1));same_velocity=mask((velocity==projected_velocity).all(-1))
        return State(torch.where(same_pose,previous_x,state.x),torch.where(same_pose,previous_q,state.q),
                     torch.where(same_velocity,h['rod_native_v'],state.v),torch.where(same_velocity,h['rod_native_omega'],state.omega),history)

    def refresh(self, control):
        model = self.source_model; spec = self.scene.rod
        if self.static_filter_pairs is not None:
            current=torch.as_tensor(model.shape_collision_filter_pairs.pairs_array().copy())
            if not torch.equal(current,self.static_filter_pairs):
                raise ValueError('Rod/contact collision filtering changed; rebuild complete history')
        for name, expected in self.static_source.items():
            if not torch.equal(self.wp.to_torch(getattr(model, name)), expected):
                raise ValueError('Rod/contact static asset changed; rebuild complete history: '+name)
        if control is not None:self.validate_controls(control)
        starts = self.wp.to_torch(model.joint_qd_start).long()[spec.source_joint_ids]
        dofs = starts[:, None]+torch.arange(6, device=starts.device)
        for field, source in (('joint_stiffness', 'joint_target_ke'), ('rod_damping', 'joint_target_kd')):
            value = self.wp.to_torch(getattr(model, source))[dofs].double()
            if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
                raise ValueError('Finite nonnegative rod constitutive parameters required')
            setattr(spec, field, value)
        for name in ('shape_material_ke', 'shape_material_kd', 'shape_material_mu'):
            value = getattr(model, name)
            if not bool(torch.isfinite(self.wp.to_torch(value)).all()) or bool((self.wp.to_torch(value) < 0).any()):
                raise ValueError('Invalid rod/contact shape material')
            setattr(self.oracle.query_model, name, value)
        gravity = self.wp.to_torch(model.gravity).double()
        if gravity.shape != (1, 3) or not bool(torch.isfinite(gravity).all()): raise ValueError('Single finite gravity world required')
        self.scene.gravity = gravity[0]

    def validate_controls(self,control):
        for name in ('joint_target_q', 'joint_target_qd'):
            if not torch.equal(self.wp.to_torch(getattr(control,name)),self.wp.to_torch(getattr(self.source_model,name))):
                raise ValueError('Changing rod material target requires its explicit rest-frame history contract: '+name)
        if bool(self.wp.to_torch(control.joint_f).any()):
            raise ValueError('Generalized rod efforts need an explicit physical force adapter')

    @torch.no_grad()
    def step(self, state_in, state_out, control, contacts, dt, *, observables=None):
        if dt != self.dt: raise ValueError('Authored rod/contact physical timestep changed')
        if observables is not None: raise ValueError('Rod/contact wrench export requires its explicit readout contract')
        self.refresh(control); start = self.read(state_in); validate_history(self.scene, start)
        removed=()
        if self.prescribed_event_provider is not None:
            removed=self.prescribed_event_provider(state_in,start)
            if len(removed):
                if not self.removal_event_contract:raise ValueError('Explicit prescribed geometry removal contract required')
                from .rod_prescribed_events import rebase_removed_geometry
                start=rebase_removed_geometry(self.scene,start,removed)
        wrench = self.wp.to_torch(state_in.body_f).double(); zero = start.x.new_zeros(1, 1, 3)
        if not bool(torch.isfinite(wrench).all()): raise ValueError('Nonfinite native rod/contact force')
        force = torch.cat((wrench[None, :, :3], zero), 1); torque = torch.cat((wrench[None, :, 3:], zero), 1)
        candidates = self.oracle.query(start); self.scene.candidates = candidates; self.scene.geometry_oracle = self.oracle
        result, diagnostic = self.network(self.scene, start, dt, learned=self.learned, force=force, torque=torque)
        self.last_diagnostic = diagnostic
        if not bool(diagnostic['accepted'].all()):
            raise RuntimeError(f'Rod/contact proposal rejected at substep {self.steps}: {diagnostic["iterations"]}')
        n = self.scene.world; pose = self.wp.to_torch(state_out.body_q); velocity = self.wp.to_torch(state_out.body_qd)
        prescribed_pose=(self.wp.to_torch(state_in.body_q)[self.scene.prescribed_ids].clone()
                         if self.scene.metadata.get('prescribed_rod_pose_history') else None)
        pose[:, :3].copy_(result.x[0, :n]-so3.rotate(result.q[0, :n], self.scene.body_com[:n])); pose[:, 3:].copy_(result.q[0, :n])
        if prescribed_pose is not None:pose[self.scene.prescribed_ids]=prescribed_pose
        velocity[:, :3].copy_(result.v[0, :n]); velocity[:, 3:].copy_(result.omega[0, :n])
        self.history = {key: value.detach().clone() for key, value in result.history.items()}
        if self.native_float64_state:
            self.history['rod_native_v']=result.v.detach().clone();self.history['rod_native_omega']=result.omega.detach().clone()
        if len(removed):self.geometry_removal_event_steps.append({'physical_step':self.steps+1,'bodies':torch.as_tensor(removed).cpu().tolist()})
        self.steps += 1
        if self.on_substep is not None: self.on_substep(state_out)
