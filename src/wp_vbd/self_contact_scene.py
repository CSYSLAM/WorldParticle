"""Explicit experimental assembly of elastic, surface, joint and VF/EE rows."""
from .general_joint_export import export_spec, read_controls
from .mixed_contact_scene import export_model as export_elastic
from .self_contact_query import interaction_geometry
from .surface_joint_scene import SurfaceJointScene
from .official import solver_configuration_record

CONTRACT = 'elastic_surface_joint_self_contact_v58'


def authored_full_surface(example):
    flags = []
    contacts = getattr(example, 'contacts', None)
    if contacts is not None:
        flags.append(bool(contacts._enable_rigid_soft_full_surface_contact))
    pipeline = getattr(example, 'collision_pipeline', None)
    if pipeline is not None:
        flags.append(bool(pipeline.enable_rigid_soft_full_surface_contact))
    if len(set(flags)) > 1:
        raise ValueError('Conflicting source shape-contact feature eligibility')
    # The pinned CollisionPipeline default is particle-only. Examples that
    # explicitly enable full surface expose their pipeline or contacts.
    return flags[0] if flags else False


def export_model(model, configuration, identifier, newton, dt, device='cpu', *, motion_budget=None, full_surface=False, broad_phase='chunked', max_rows=200000, prescribed_massless_bodies=False,path_subdivision_depth=0):
    if type(prescribed_massless_bodies) is not bool:
        raise ValueError('Explicit Boolean prescribed-body ownership required')
    if type(max_rows) is not int or max_rows<=0:
        raise ValueError('Positive integer complete self-contact row capacity required')
    if not configuration['particle_enable_self_contact']:
        raise ValueError('An enabled original self-contact configuration is required')
    allowed = {int(getattr(newton.GeoType, name)) for name in
               ('PLANE', 'SPHERE', 'BOX', 'CAPSULE', 'CYLINDER', 'CONE', 'ELLIPSOID')}
    if any(int(value) not in allowed for value in model.shape_type.numpy()):
        raise ValueError('Self-contact assembly has no mesh shape force contract')
    interaction_geometry(configuration)
    spec = export_spec(model, configuration, newton, device)
    spec.owners = spec.owners+model.particle_count
    base = export_elastic(model, configuration, identifier, newton, dt, device,
                          finite_joint_export=True, self_contact_export=True)
    metadata = {**base.metadata, 'contract': CONTRACT,
                'solver_configuration': solver_configuration_record(configuration),
                'contacts': 'analytic_rigid_weighted_soft_and_reciprocal_particle_VF_EE',
                'mixed_finite_joints': True, 'joint_history_contract': 'single_axis_winding_v26',
                'self_contact_EE_topology_rule': 'union_of_source_directional_exclusions',
                'self_shape_full_surface': bool(full_surface),
                'self_contact_broad_phase': broad_phase,
                'self_contact_max_rows':max_rows,
                'self_contact_path_subdivision_depth':path_subdivision_depth,
                'self_contact_motion_budget': .05*base.length if motion_budget is None else float(motion_budget),
                'scope': 'Experimental complete self-contact assembly; no training, large-scene, external ownership or realtime claim.'}
    if prescribed_massless_bodies:
        import numpy as np
        if not model.body_count or np.any(model.body_mass.numpy()!=0) or np.any(model.body_inv_mass.numpy()!=0):
            raise ValueError('This prescribed-boundary contract requires authored massless rigid bodies')
        if np.any(model.joint_type.numpy()!=int(newton.JointType.FREE)) or len(spec.owners):
            raise ValueError('Prescribed finite-joint/drive ownership needs a separate explicit contract')
        if np.any(model.joint_target_ke.numpy()) or np.any(model.joint_target_kd.numpy()):
            raise ValueError('Prescribed FREE drives need an explicit generalized force contract')
        metadata.update(prescribed_pose_history='accepted_com_pose_v30',
                        prescribed_rigid_ownership='massless_commands_v122',
                        prescribed_rigid_node_ids=list(range(model.particle_count,model.particle_count+model.body_count)))
    scene = SurfaceJointScene(base.values, metadata, spec)
    scene.joint_controls = read_controls(spec, model.control(clone_variables=False))
    return scene
