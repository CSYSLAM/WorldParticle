"""Qualified native robot/cloth trajectories, rebuilt without a dynamics solver."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .data import ROOT
from .franka_reference_data import tensors
from .franka_cloth_corner_scene import build
from .surface_joint_scene import export_model
from .surface_joint_runtime import SurfaceJointCollisionOracle
from .newton_source import newton_revision

SCHEMA = 'authored_robot_cloth_substeps_v1'
STATE_KEYS = ('x','q','v','omega','history__pin_x_prev','history__pin_initialized',
              'history__joint_angle_prev','input_pin_x','input_force','input_torque',
              'input_joint_target','input_joint_target_rate','input_joint_effort')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_policy(scene):
    # Match the inference entry, including accepted tangential contact memory.
    scene.metadata.update(detection_motion_budget=.012,bounded_motion_forecast_target_only=True,
        joint_tree_coarse_modes='scalar_tree_twelve_columns_v307',
        articulation_material_motion_modes=True,independent_object_motion_modes=True,
        soft_tangent_support_scale=64.,force_aware_material_anticipation=True,
        compact_numerical_diagnostics=True,repair_native_mimic_roundoff=True,
        surface_tangent_memory=True,surface_tangent_history_capacity=8192)


def load_reference(path,wp,newton,device='cpu',*,state_device=None):
    path=Path(path)
    proof=json.loads(path.with_suffix('.qualification.json').read_text())
    if proof['status']!='QUALIFIED_ROBOT_CLOTH_NATIVE_DATA' or proof['sha256']!=digest(path):
        raise ValueError('Exact qualified robot/cloth trajectory required')
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));values={k:z[k].copy() for k in z.files if k!='metadata'}
    if meta['training_schema']!=SCHEMA or meta['newton_sha']!=newton_revision():
        raise ValueError('Robot/cloth schema or Newton version changed')
    if meta['builder_sha256']!=digest(ROOT/'src/wp_vbd/franka_cloth_corner_scene.py'):
        raise ValueError('Robot/cloth source builder changed')
    e=build(wp,newton,device=device,initialize_controls=False,**meta['build_options'])
    e.model.joint_q.assign(values['source__joint_q'])
    scratch=e.model.state()
    newton.eval_fk(e.model,e.model.joint_q,e.model.joint_qd,scratch)
    wp.copy(e.model.body_q,scratch.body_q)
    e.model.joint_target_q.assign(e.model.joint_q)
    if 'source__body_q' not in values:raise ValueError('Complete frozen source assets required')
    from .surface_joint_data import restore_joint_source
    restore_joint_source(e.model,values)
    scene=export_model(e.model,meta['solver_configuration'],meta['official_demo'],newton,meta['dt'],device)
    for key,value in scene.values.items():
        saved=torch.as_tensor(values[key],device=device,dtype=value.dtype)
        same=torch.allclose(value,saved,rtol=1e-10,atol=1e-10) if value.is_floating_point() else torch.equal(value,saved)
        if not same:raise ValueError('Robot/cloth derived asset changed: '+key)
    for prefix,spec in [('joint__',scene.joint_spec),('mimic__',scene.mimic_spec)]:
        for key,value in vars(spec).items():
            saved=torch.as_tensor(values[prefix+key],device=device,dtype=value.dtype)
            same=torch.allclose(value,saved,rtol=1e-10,atol=1e-10) if value.is_floating_point() else torch.equal(value,saved)
            if not same:raise ValueError('Robot/cloth joint changed: '+prefix+key)
    scene.metadata.update(meta)
    model_policy(scene)
    scene.geometry_oracle=SurfaceJointCollisionOracle(e.model,scene,wp,newton)
    states={k:torch.as_tensor(values[k],device=state_device or device) for k in STATE_KEYS}
    return scene,states
