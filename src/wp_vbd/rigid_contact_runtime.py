"""Independent expanded collision query and bounded pose-motion envelope."""
import copy
import math
import torch
from newton.solvers import SolverBase
from . import math3d as so3
from .data import State
from .rigid_collision import RigidContacts


class CollisionOracle:
    def __init__(self,model,scene,wp,newton,*,motion_budget=None):
        self.wp=wp;self.scene=scene;self.source_model=model
        self.motion_budget=.05*scene.length if motion_budget is None else float(motion_budget)
        if not math.isfinite(self.motion_budget) or self.motion_budget<=0:raise ValueError('Finite positive detection motion budget required')
        # Only detection gap changes; mass, geometry, material margin and
        # force coefficients remain the original asset. Pair gap padding is
        # larger than the sum of both endpoints' allowed surface travel.
        self.query_model=copy.copy(model)
        self.query_model.shape_gap=wp.array(model.shape_gap.numpy()+self.motion_budget/.85,dtype=wp.float32,device=model.device)
        from .rigid_contact_capacity import capacity as complete_capacity
        capacity,self.capacity_policy=complete_capacity(model,scene,newton)
        self.declared_contact_capacity=capacity
        self.bounded_analytic_buffers=scene.metadata.get('bounded_analytic_contact_buffers')=='current_expanded_aabb_v146'
        matching={}
        if scene.metadata.get('cross_step_contact_history'):
            from .rigid_contact_memory import validate_policy
            policy=validate_policy(scene.metadata['contact_matching_policy'])
            if scene.metadata['contact_history_capacity']!=capacity:raise ValueError('Contact history/query capacity differs')
            matching={'contact_matching':policy['mode'],'contact_matching_pos_threshold':policy['position_threshold'],
                      'contact_matching_normal_dot_threshold':policy['normal_dot_threshold']}
        self._matching_options=matching;self._newton=newton
        self.capacity_reallocations=[];self.maximum_aabb_pair_bound=0
        if self.bounded_analytic_buffers:
            from .analytic_contact_capacity import AnalyticAabbCapacity
            self.aabb_counter=AnalyticAabbCapacity(self.query_model,wp,newton)
            capacity=min(1024,capacity)
            self.capacity_policy='complete_current_expanded_aabb_two_witness_bound_with_accepted_cache'
        self.pipeline=newton.CollisionPipeline(self.query_model,deterministic=True,rigid_contact_max=capacity,**matching)
        self.contacts=self.pipeline.contacts();self.query_state=self.query_model.state();self.queries=0
        self.maximum_candidate_count=0;self.maximum_allocated_contact_capacity=capacity

    def query(self,state):
        if len(state.x)!=1:raise ValueError('First contact runtime explicitly requires batch_size=1')
        if self.scene.metadata.get('cross_step_contact_history') and not self.bounded_analytic_buffers:
            from .rigid_contact_memory import restore_query
            restore_query(self,state)
        n=self.scene.world
        pose=self.wp.to_torch(self.query_state.body_q)
        pose[:,:3].copy_((state.x[0,:n]-so3.rotate(state.q[0,:n],self.scene.body_com[:n])).detach())
        pose[:,3:].copy_(state.q[0,:n].detach())
        if self.bounded_analytic_buffers:
            from .rigid_contact_memory import validate,restore_query
            validate(self.scene,state)
            plan=self.aabb_counter.measure(self.query_state)
            self.maximum_aabb_pair_bound=max(self.maximum_aabb_pair_bound,plan['overlapping_aabb_pairs_before_filters'])
            previous_count=int(state.history['contact_match_count'][0,0])
            required=max(1024,min(self.declared_contact_capacity,plan['primitive_witness_bound']),previous_count)
            if required>len(self.contacts.rigid_contact_shape0):
                capacity=min(self.declared_contact_capacity,1<<(required-1).bit_length())
                if capacity<required:raise ValueError('Accepted contact cache exceeds declared complete witness capacity')
                # Both geometry and all accepted history are owned snapshots.
                # Drop old internal buffers before allocating their replacement.
                del self.contacts,self.pipeline
                import gc
                gc.collect()
                self.pipeline=self._newton.CollisionPipeline(self.query_model,deterministic=True,
                    rigid_contact_max=capacity,**self._matching_options)
                self.contacts=self.pipeline.contacts()
                self.capacity_reallocations.append({'before_query':self.queries,'required':required,
                    'allocated':capacity,'accepted_cached_rows':previous_count,
                    'complete_geometry_witness_bound':plan['primitive_witness_bound']})
                self.maximum_allocated_contact_capacity=max(self.maximum_allocated_contact_capacity,capacity)
            restore_query(self,state)
        self.pipeline.collide(self.query_state,self.contacts)
        self.queries+=1
        result=RigidContacts.snapshot(self.query_model,self.contacts,self.wp)
        if self.scene.metadata.get('cross_step_contact_history'):
            from .rigid_contact_memory import capture_query
            if len(result.owners)!=result.source_count:raise ValueError('Matched history requires a complete valid sorted contact stream')
            self.contact_history_proposal=capture_query(self)
        self.maximum_candidate_count=max(self.maximum_candidate_count,result.source_count)
        return result

    def motion(self,start,trial):
        angle=so3.log(so3.multiply(trial.q,so3.conjugate(start.q))).norm(dim=-1)
        travel=(trial.x-start.x).norm(dim=-1)+self.scene.body_radius*angle
        finite=torch.isfinite(travel).all(-1)
        valid=finite&(travel[:,self.scene.movable].amax(-1)<=self.motion_budget)
        return valid,travel


class ContactExampleSolver(SolverBase):
    """Native State I/O with geometry queries, never a VBD constructor/step."""
    def __init__(self,model,scene,initial,dt,wp,newton,source_model,*,learned=True):
        super().__init__(source_model)
        self.network=model;self.source_model=source_model;self.scene=scene;self.dt=dt;self.wp=wp;self.learned=learned
        budget=model.config.get('native_contact_motion_budget') if scene.metadata.get('contract') in ('rod_shape_contact_com_v40','rod_finite_joint_contact_com_v143') else None
        self.oracle=CollisionOracle(source_model,scene,wp,newton,motion_budget=budget)
        self.steps=0;self.on_substep=None;self.last_diagnostic=None;self.contact_exports=0

    def set_rigid_history_update(self,update):
        """Accept a fresh-snapshot request; retained contact caches are separate.

        This backend performs its own fresh collision query on every step.
        It cannot silently honor a request to reuse source solver history.
        """
        if not bool(update):raise ValueError('Retained rigid contact cache requires a separate history contract')

    @property
    def supported_observables(self):
        from .contact_observables import enabled
        return frozenset({self.ObservableKind.CONTACT_F}) if enabled(self.scene) else frozenset()

    def read(self,state):
        pose=self.wp.to_torch(state.body_q).double();q=so3.normalize(pose[:,3:]);n=len(pose)
        x=pose[:,:3]+so3.rotate(q,self.scene.body_com[:n]);velocity=self.wp.to_torch(state.body_qd).double()
        return State(torch.cat((x,x.new_zeros(1,3)))[None],torch.cat((q,q.new_tensor([[0.,0.,0.,1.]])))[None],torch.cat((velocity[:,:3],x.new_zeros(1,3)))[None],torch.cat((velocity[:,3:],x.new_zeros(1,3)))[None])

    @torch.no_grad()
    def step(self,state_in,state_out,control,contacts,dt,*,observables=None):
        from . import contact_observables as readout
        readout.validate(self,observables,contacts)
        if dt!=self.dt:raise ValueError('Authored contact timestep changed')
        start=self.read(state_in);candidates=self.oracle.query(start)
        self.scene.candidates=candidates;self.scene.geometry_oracle=self.oracle
        wrench=self.wp.to_torch(state_in.body_f).double();zero=start.x.new_zeros(1,1,3)
        force=torch.cat((wrench[:,:3][None],zero),1);torque=torch.cat((wrench[:,3:][None],zero),1)
        result,diagnostic=self.network(self.scene,start,dt,learned=self.learned,force=force,torque=torque)
        self.last_diagnostic=diagnostic
        if not bool(diagnostic['accepted'].all()):raise RuntimeError(f'Contact proposal rejected at substep {self.steps}: {diagnostic["iterations"]}; finite={diagnostic["finite"]}')
        prepared=readout.prepare(self,diagnostic,contacts,observables)
        n=self.scene.world;pose=self.wp.to_torch(state_out.body_q);velocity=self.wp.to_torch(state_out.body_qd)
        pose[:,:3].copy_(result.x[0,:n]-so3.rotate(result.q[0,:n],self.scene.body_com[:n]));pose[:,3:].copy_(result.q[0,:n])
        velocity[:,:3].copy_(result.v[0,:n]);velocity[:,3:].copy_(result.omega[0,:n])
        readout.commit(self,prepared,contacts,observables);self.steps+=1
        if self.on_substep is not None:self.on_substep(state_out)
