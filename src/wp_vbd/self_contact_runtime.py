"""Native self-contact geometry queries and accepted pin/joint history."""
import torch

from .mixed_contact_runtime import MixedCollisionOracle, MixedContactExampleSolver, native_tensor
from .self_contact_query import SelfContactQuery
from .self_contact_scene import CONTRACT
from .self_contact_system import validate_history
from .self_surface_collision import SelfSurfaceContacts
from .surface_joint_collision import SurfaceJointContacts
from .surface_joint_runtime import SurfaceJointExampleSolver


class SelfContactCollisionOracle(MixedCollisionOracle):
    validate_snapshot_history=staticmethod(validate_history)
    def __init__(self, model, scene, wp, newton):
        self.newton = newton
        super().__init__(model, scene, wp, newton,
                         motion_budget=scene.metadata['self_contact_motion_budget'],
                         full_surface=scene.metadata['self_shape_full_surface'])
        responsive_only=scene.metadata.get('self_contact_responsive_pairs_only',False)
        if type(responsive_only) is not bool:
            raise ValueError('Explicit Boolean self-contact pair eligibility required')
        self.self_query = SelfContactQuery(model, scene.metadata['solver_configuration'],
                                           device=scene.device, motion_budget=self.motion_budget,
                                           max_rows=scene.metadata.get('self_contact_max_rows',200000),
                                           broad_phase=scene.metadata.get('self_contact_broad_phase','chunked'),
                                           path_subdivision_depth=scene.metadata.get('self_contact_path_subdivision_depth',0),
                                           responsive_particles=scene.movable[:scene.metadata['particle_count']] if responsive_only else None)
        self.self_snapshot = None

    def snapshot(self, state):
        scene = self.scene
        self.validate_snapshot_history(scene, state)
        # Pins are an authored boundary input, not a new accepted material
        # state. Include their previous accepted positions from step zero.
        reference = self.material_reference(state)
        if not bool(state.history['pin_initialized'].all()):
            raise ValueError('Self-contact must initialize accepted pin history before the first boundary command')
        snapshot = self.self_query.snapshot(reference,finite_overlap_length=scene.length if getattr(scene,'learned_contact_repair',False) else None)
        if not getattr(scene,'learned_contact_repair',False) and not bool(snapshot.path_feasible(state.x[0, :len(reference)])):
            raise ValueError('Imposed particle input violates self-contact path or detection envelope')
        base = SurfaceJointContacts.snapshot(self.query_model, self.contacts, self.wp, self.newton)
        self.self_snapshot = snapshot
        return SelfSurfaceContacts(base, snapshot, stiffness=self.source_model.soft_contact_ke,
                                    damping=self.source_model.soft_contact_kd, friction=self.source_model.soft_contact_mu)

    def material_reference(self,state):
        scene=self.scene
        reference=state.x[0,:scene.metadata['particle_count']].detach().clone()
        reference[scene.fixed_particle_ids]=state.history['pin_x_prev'][0]
        return reference

    def motion(self, start, trial):
        valid, travel = super().motion(start, trial)
        if self.self_snapshot is not None:
            # The inertial forecast is an unconstrained TARGET. Only accepted
            # candidates get whole-feature path certificates in physics.choose;
            # a colliding forecast must remain available for correction.
            reference = self.self_snapshot.reference
            distance = (trial.x[0, :self.self_query.particle_count]-reference).norm(dim=-1)
            valid = valid & torch.isfinite(distance).all() & (distance <= self.motion_budget).all()
        return valid, travel


class SelfContactExampleSolver(MixedContactExampleSolver):
    def refresh_control(self, control):
        if control is not None:
            for name in ('tri_activations', 'tet_activations'):
                value = getattr(control, name, None)
                if value is not None and bool((native_tensor(self.wp, value) != 0).any()):
                    raise ValueError('Active material control requires its own self-contact input operator: '+name)
        if self.source_model.joint_dof_count:
            return SurfaceJointExampleSolver.refresh_control(self, control)
        if control is not None and any(getattr(control,name,None) is not None and getattr(control,name).size
                                       for name in ('joint_target_q','joint_target_qd','joint_f')):
            raise ValueError('Nonempty joint inputs on a zero-joint self-contact asset')
        empty = self.scene.rest_x.new_zeros(1, len(self.scene.joint_spec.owners), 6)
        self.scene.joint_controls = (empty, empty.clone(), empty.clone())

    def __init__(self, network, scene, dt, wp, newton, source_model, initial_native, *, learned=True):
        if scene.metadata.get('contract') != CONTRACT or not network.config.get('particle_self_contact_v58'):
            raise ValueError('An explicitly enabled shared self-contact model and scene are required')
        super().__init__(network, scene, dt, wp, newton, source_model, learned=learned,
                         oracle_type=SelfContactCollisionOracle)
        self.newton = newton
        self.source_bvh_rebuild_requests=0
        initial = self.read(initial_native)
        self.history['pin_x_prev'] = initial.x[:, scene.fixed_particle_ids].detach().clone()
        self.history['pin_initialized'] = torch.ones(1, device=scene.device, dtype=torch.bool)
        spec = scene.joint_spec
        value = scene.joint_coordinates(initial, spec.coordinate_offset[None, :, 3]).axis[..., 3]
        self.history['joint_angle_prev'] = torch.where(spec.angular_mode == 1, value, 0.).detach().clone()
        from . import prescribed_motion as prescribed
        if prescribed.enabled(scene):self.history.update(prescribed.initialize(scene,initial))
        for name in ('body_q', 'shape_margin', 'shape_collision_radius', 'shape_flags', 'edge_rest_angle',
                     'edge_rest_length', 'joint_type', 'joint_parent', 'joint_child', 'joint_X_p', 'joint_X_c',
                     'joint_axis', 'joint_dof_dim', 'joint_q', 'joint_q_start', 'joint_qd_start', 'joint_target_q_start'):
            array = getattr(source_model, name, None)
            if array is not None:
                self.static_source[name] = native_tensor(wp, array).clone()

    def validate_start(self, start):
        validate_history(self.scene, start)

    def rebuild_bvh(self, state):
        # Some authored controllers request a frame-start BVH refresh. This
        # backend builds fresh complete geometry at EVERY physical query, after
        # its boundary input; no stale BVH is retained between calls.
        value=native_tensor(self.wp,state.particle_q)
        if value.shape!=(self.scene.metadata['particle_count'],3) or not bool(torch.isfinite(value).all()):
            raise ValueError('Finite matched particles required for a geometry rebuild request')
        self.source_bvh_rebuild_requests+=1
