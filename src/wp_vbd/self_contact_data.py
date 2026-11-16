"""Explicit accepted material/pin states for the experimental self assembly."""
import json
from types import SimpleNamespace

import numpy as np
import torch

from .self_contact_runtime import SelfContactCollisionOracle
from .self_contact_scene import CONTRACT, export_model, authored_full_surface
from .surface_joint_data import restore_joint_source
from .surface_joint_scene import SurfaceJointScene


def load_reference(path, wp, newton, device='cpu', *, state_device=None, compact_reference_constants=False):
    from .official import make_example
    from .general_joint_data import set_controls
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        values = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
    if metadata['contract'] != CONTRACT or metadata['reference_final_check'] != 'PASS':
        raise ValueError('Qualified complete self-contact reference required')
    example, configurations, _ = make_example(metadata['official_demo'], reference=False, preserve_defaults=True,
                                              contact_capacity=metadata.get('reference_contact_capacity_overrides'))
    if len(configurations) != 1:
        raise ValueError('A single owned self-contact subsystem is required')
    if 'self_shape_full_surface' not in metadata or authored_full_surface(example) != metadata['self_shape_full_surface']:
        raise ValueError('Self-contact data must explicitly preserve authored shape-contact feature eligibility')
    roundoff = restore_joint_source(example.model, values)
    for name, value in metadata['source_soft_material'].items():
        if getattr(example.model, name) != value:
            raise ValueError('Self-contact source material changed: '+name)
    current = export_model(example.model, configurations[0][1], metadata['official_demo'], newton,
                           example.sim_dt, device, motion_budget=metadata['self_contact_motion_budget'],
                           full_surface=metadata['self_shape_full_surface'],max_rows=metadata.get('self_contact_max_rows',200000),
                           prescribed_massless_bodies=metadata.get('prescribed_rigid_ownership')=='massless_commands_v122')
    if current.metadata['solver_configuration'] != metadata['solver_configuration']:
        raise ValueError('Self-contact solver configuration changed')
    for key in ('prescribed_pose_history','prescribed_rigid_ownership','prescribed_rigid_node_ids'):
        if current.metadata.get(key)!=metadata.get(key):raise ValueError('Self-contact rigid ownership changed: '+key)
    static = {};derived_roundoff={}
    for name, value in current.values.items():
        recorded = torch.as_tensor(values[name], device=device, dtype=value.dtype)
        if not torch.equal(value, recorded):
            if name!='tet_volume':raise ValueError('Self-contact derived asset changed: '+name)
            dm=torch.as_tensor(values['tet_Dm'],device=device,dtype=current.tet_Dm.dtype)
            if not torch.equal(dm,current.tet_Dm):raise ValueError('Self-contact inverse rest geometry changed')
            from .derived_geometry import certify_tet_volume_roundoff
            derived_roundoff[name]=certify_tet_volume_roundoff(value,recorded,dm)
        static[name] = recorded
    spec = {}
    for name, value in vars(current.joint_spec).items():
        recorded = torch.as_tensor(values['joint__'+name], device=device, dtype=value.dtype)
        if not torch.equal(value, recorded):
            raise ValueError('Self-contact joint schema changed: '+name)
        spec[name] = recorded
    scene = SurfaceJointScene(static, metadata, SimpleNamespace(**spec))
    canonical_volume=static['tet_volume']
    if not torch.equal(scene.tet_volume,canonical_volume):
        from .derived_geometry import certify_tet_volume_roundoff
        derived_roundoff['tet_volume']=certify_tet_volume_roundoff(scene.tet_volume,canonical_volume,scene.tet_Dm)
        scene.values['tet_volume']=canonical_volume
        scene.objective_scene.tet_volume=canonical_volume
        scene.elastic_scene.tet_volume=canonical_volume
    scene.current_device_static_roundoff = roundoff
    scene.current_device_derived_roundoff=derived_roundoff
    scene.geometry_oracle = SelfContactCollisionOracle(example.model, scene, wp, newton)
    names = ('x', 'q', 'v', 'omega', 'history__pin_x_prev', 'history__pin_initialized',
             'history__joint_angle_prev', 'input_pin_x', 'input_joint_target',
             'input_joint_target_rate', 'input_joint_effort', 'input_force', 'input_torque')
    if metadata.get('prescribed_pose_history'):
        names+=('history__prescribed_x_prev','history__prescribed_q_prev','history__prescribed_initialized',
                'input_prescribed_x','input_prescribed_q','input_prescribed_v','input_prescribed_omega')
    from .reference_storage import temporal_tensor, storage_summary
    states,storage_fields = {},{}
    for name in names:
        states[name],storage_fields[name] = temporal_tensor(values.pop(name),state_device or device,
                                                           compact_constants=compact_reference_constants)
    scene.reference_storage = storage_summary(states,storage_fields)
    if any(not bool(torch.isfinite(value).all()) for value in states.values()):
        raise ValueError('Nonfinite self-contact state or physical input')
    if not bool(states['history__pin_initialized'].all()):
        raise ValueError('Self-contact accepted pin history must be initialized before commands')
    set_controls(scene, states, 0)
    # A host trajectory can accompany device-resident static physics. Keep
    # even empty initial control tensors on the scene's device; later calls
    # use controls from the complete transferred prediction window.
    scene.joint_controls=tuple(value.to(device) for value in scene.joint_controls)
    return scene, states
