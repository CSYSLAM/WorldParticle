"""Native weighted surface/joint state, controls and atomic history ownership."""
import torch

from .general_joint_export import export_spec, read_controls
from .mixed_contact_runtime import MixedCollisionOracle, MixedContactExampleSolver
from .surface_joint_collision import SurfaceJointContacts
from .surface_joint_scene import CONTRACT
from .surface_joint_system import validate_history


class SurfaceJointCollisionOracle(MixedCollisionOracle):
    def __init__(self, model, scene, wp, newton):
        self.newton = newton
        super().__init__(model, scene, wp, newton, full_surface=True,
                         motion_budget=scene.metadata.get('detection_motion_budget'))

    def snapshot(self, state):
        return SurfaceJointContacts.snapshot(self.query_model, self.contacts, self.wp, self.newton)


class SurfaceJointExampleSolver(MixedContactExampleSolver):
    def __init__(self, network, scene, dt, wp, newton, source_model, initial_native, *, learned=True):
        self.native_roundoff_projection_count=0
        self.maximum_native_position_correction=0.
        if scene.metadata.get('contract') != CONTRACT:
            raise ValueError('A complete surface/finite-joint scene is required')
        super().__init__(network, scene, dt, wp, newton, source_model, learned=learned, oracle_type=SurfaceJointCollisionOracle)
        self.newton = newton
        spec = scene.joint_spec; initial = self.read(initial_native)
        value = scene.joint_coordinates(initial, spec.coordinate_offset[None, :, 3]).axis[..., 3]
        self.history['joint_angle_prev'] = torch.where(spec.angular_mode == 1, value, 0.).detach().clone()
        if scene.metadata.get('surface_tangent_memory',False):
            from .surface_tangent_memory import initial_history
            self.history.update(initial_history(initial,scene.metadata['surface_tangent_history_capacity']))
        from . import prescribed_motion as prescribed
        if prescribed.enabled(scene):self.history.update(prescribed.initialize(scene,initial))
        names = ('body_q', 'shape_margin', 'shape_collision_radius', 'shape_flags', 'edge_rest_angle',
                 'edge_rest_length', 'joint_type', 'joint_parent', 'joint_child', 'joint_X_p', 'joint_X_c',
                 'joint_axis', 'joint_dof_dim', 'joint_q', 'joint_q_start', 'joint_qd_start', 'joint_target_q_start',
                 'joint_mimic_joint', 'joint_mimic_coeffs', 'joint_enabled', 'joint_target_ke', 'joint_target_kd',
                 'joint_target_mode', 'joint_limit_lower', 'joint_limit_upper', 'joint_limit_ke', 'joint_limit_kd')
        for name in names:
            array = getattr(source_model, name, None)
            if array is not None: self.static_source[name] = wp.to_torch(array).clone()

    def read(self, native):
        state=super().read(native)
        if self.scene.metadata.get('repair_native_mimic_roundoff',False):
            from .native_mimic_roundoff import project
            state,detail=project(self.scene,state)
            if detail is not None:
                self.native_roundoff_projection_count+=1
                self.maximum_native_position_correction=max(self.maximum_native_position_correction,detail['maximum_position_correction'])
                self.last_native_roundoff_projection=detail
        return state

    def validate_start(self, start):
        validate_history(self.scene, start)

    def refresh_control(self, control):
        model = self.source_model; scene = self.scene
        if self.network.config.get('resident_static_structure',False):
            spec=scene.joint_spec
        else:
            spec = export_spec(model, scene.metadata['solver_configuration'], self.newton, str(scene.device), allow_mimic=True)
            spec.owners = spec.owners+scene.metadata['particle_count']
            for key in ('owners', 'source_joint_ids', 'frames', 'angular_mode', 'linear_axes', 'angular_axes',
                        'coordinate_offset', 'rest_parent', 'rest_child', 'dof_index', 'target_index'):
                if not torch.equal(getattr(spec, key), getattr(scene.joint_spec, key)):
                    raise ValueError('Mixed joint coordinates changed; rebuild scene/history: '+key)
        if control is None: control = model.control(clone_variables=False)
        values = read_controls(spec, control)
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError('Nonfinite mixed joint controls')
        effort = self.wp.to_torch(control.joint_f)
        used = torch.zeros(len(effort), device=effort.device, dtype=torch.bool)
        index = spec.dof_index[spec.dof_index >= 0]
        if len(index): used[index.to(used.device)] = True
        if bool((effort[~used] != 0).any()):
            raise ValueError('Unmapped FREE/BALL generalized effort requires its own force adapter')
        scene.joint_spec = spec; scene.joint_controls = values
