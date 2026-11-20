"""Qualified weighted-contact joint episodes with explicit input channels."""
import json
from types import SimpleNamespace

import numpy as np
import torch

from .mixed_contact_data import STATIC_FIELDS, restore_source
from .surface_joint_scene import CONTRACT, SurfaceJointScene, export_model
from .surface_joint_runtime import SurfaceJointCollisionOracle


JOINT_FIELDS = ('joint_X_p', 'joint_X_c', 'joint_axis', 'joint_dof_dim', 'joint_q_start',
                'joint_target_q_start', 'joint_enabled', 'joint_target_ke', 'joint_target_kd',
                'joint_limit_lower', 'joint_limit_upper', 'joint_limit_ke', 'joint_limit_kd',
                'joint_mimic_joint')


def restore_joint_source(model, source):
    roundoff = restore_source(model, source)
    for name in JOINT_FIELDS:
        value = getattr(model, name, None)
        if value is None:
            if 'source__'+name in source: raise ValueError('Joint source field disappeared: '+name)
            continue
        if not np.array_equal(value.numpy(), source['source__'+name]):
            raise ValueError('Authored finite-joint field changed: '+name)
    return roundoff


def load_reference(path, wp, newton, device='cpu', *, state_device=None):
    from .official import make_example
    from .general_joint_data import set_controls
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        values = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
    if metadata.get('training_schema')=='authored_franka_substep_reference_v310':
        from .franka_reference_data import load_reference as load_authored_franka
        return load_authored_franka(path,wp,newton,device,state_device=state_device)
    if metadata['contract'] != CONTRACT or metadata['reference_final_check'] != 'PASS':
        raise ValueError('Qualified complete surface/joint reference required')
    example, solvers, _ = make_example(metadata['official_demo'], reference=False, preserve_defaults=True)
    if len(solvers) != 1 or not example.contacts._enable_rigid_soft_full_surface_contact:
        raise ValueError('Authored single-owner full-surface source required')
    roundoff = restore_joint_source(example.model, values)
    for name, value in metadata['source_soft_material'].items():
        if getattr(example.model, name) != value: raise ValueError('Soft contact material changed: '+name)
    current = export_model(example.model, solvers[0][1], metadata['official_demo'], newton, example.sim_dt, device)
    if current.metadata['solver_configuration'] != metadata['solver_configuration']:
        raise ValueError('Surface joint solver configuration changed')
    static = {}
    for name, value in current.values.items():
        recorded = torch.as_tensor(values[name], device=device, dtype=value.dtype)
        if not torch.equal(value, recorded): raise ValueError('Surface joint derived asset changed: '+name)
        static[name] = recorded
    spec = {}
    for name, value in vars(current.joint_spec).items():
        recorded = torch.as_tensor(values['joint__'+name], device=device, dtype=value.dtype)
        same = torch.equal(value, recorded) if not value.is_floating_point() else torch.allclose(value, recorded, atol=1e-12, rtol=1e-12)
        if not same: raise ValueError('Surface joint coordinate/material changed: '+name)
        if not torch.equal(value, recorded): roundoff['joint__'+name] = float((value-recorded).abs().max())
        spec[name] = recorded
    scene = SurfaceJointScene(static, metadata, SimpleNamespace(**spec))
    scene.current_device_static_roundoff = roundoff
    scene.geometry_oracle = SurfaceJointCollisionOracle(example.model, scene, wp, newton)
    names = ('x', 'q', 'v', 'omega', 'history__pin_x_prev', 'history__pin_initialized',
             'history__joint_angle_prev', 'input_pin_x', 'input_joint_target',
             'input_joint_target_rate', 'input_joint_effort', 'input_force', 'input_torque')
    states = {name: torch.as_tensor(values[name], device=state_device or device) for name in names}
    if any(not bool(torch.isfinite(value).all()) for value in states.values()):
        raise ValueError('Nonfinite weighted-contact training state or input')
    set_controls(scene, states, 0)
    return scene, states
