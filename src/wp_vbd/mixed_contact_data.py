"""Explicit source assets, pin commands and accepted-history training views."""
import json

import numpy as np
import torch

from . import math3d as so3
from .data import State
from .mixed_contact_runtime import MixedCollisionOracle
from .mixed_contact_scene import export_model, MixedContactScene

STATIC_FIELDS = ('particle_q', 'particle_qd', 'particle_flags', 'particle_mass', 'particle_inv_mass', 'particle_radius', 'body_q', 'body_qd', 'body_mass', 'body_inv_mass', 'body_com', 'body_inertia', 'body_inv_inertia', 'body_flags', 'tri_indices', 'tri_poses', 'tri_areas', 'tri_materials', 'tet_indices', 'tet_poses', 'tet_materials', 'edge_indices', 'edge_rest_angle', 'edge_rest_length', 'edge_bending_properties', 'shape_body', 'shape_type', 'shape_transform', 'shape_scale', 'shape_margin', 'shape_collision_radius', 'shape_flags', 'shape_material_ke', 'shape_material_kd', 'shape_material_mu', 'particle_world', 'body_world', 'shape_world', 'gravity', 'joint_type', 'joint_parent', 'joint_child', 'joint_q', 'joint_qd_start')


def restore_source(model, source):
    """Verify authored geometry first, then restore bounded derived roundoff."""
    geometry = {}
    for shape, item in enumerate(model.shape_source):
        for name in ('vertices', 'indices', 'points'):
            value = getattr(item, name, None)
            if value is None: continue
            value = value.numpy() if hasattr(value, 'numpy') else np.asarray(value)
            if not value.dtype.hasobject: geometry[f'source_geometry{shape}__{name}'] = value
    recorded_geometry = {key: value for key, value in source.items() if key.startswith('source_geometry')}
    if geometry.keys() != recorded_geometry.keys(): raise ValueError('Mixed source mesh representation changed')
    for name, value in geometry.items():
        if not np.array_equal(value, recorded_geometry[name]): raise ValueError('Mixed source mesh geometry changed: ' + name)
    derived = {'particle_inv_mass', 'body_mass', 'body_inv_mass', 'body_com', 'body_inertia', 'body_inv_inertia', 'body_q', 'shape_collision_radius'}
    roundoff = {}
    for name in STATIC_FIELDS:
        array = getattr(model, name, None)
        if array is None:
            if 'source__' + name in source: raise ValueError('Mixed source field disappeared: ' + name)
            continue
        recorded = source['source__' + name]; current = array.numpy()
        if np.array_equal(current, recorded): continue
        difference = float(np.max(np.abs(current - recorded), initial=0.))
        scale = float(np.max(np.abs(recorded), initial=0.))
        if name not in derived or difference > 1e-5 * max(scale, 1e-30) + (1e-7 if name in ('body_com', 'body_q') else 0.):
            raise ValueError(f'Mixed source asset changed: {name}, max difference={difference}')
        roundoff[name] = difference; array.assign(recorded)
    return roundoff


def load_reference(path, wp, newton, device='cuda:0', *, state_device=None):
    from .official import make_example
    with np.load(path, allow_pickle=False) as archive:
        values = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
        metadata = json.loads(str(archive['metadata']))
    if metadata['contract'] != 'elastic_shape_contact_v33' or metadata['reference_final_check'] != 'PASS': raise ValueError('Admitted mixed contact reference required')
    example, solvers, _ = make_example(metadata['official_demo'], reference=False, preserve_defaults=True)
    roundoff = restore_source(example.model, values)
    for name, value in metadata['source_soft_material'].items():
        if getattr(example.model, name) != value: raise ValueError('Mixed source soft material changed: ' + name)
    current = export_model(example.model, solvers[0][1], metadata['official_demo'], newton, example.sim_dt, device)
    if current.metadata['solver_configuration'] != metadata['solver_configuration']: raise ValueError('Mixed reference solver configuration changed')
    static = {}
    derived_roundoff={}
    for name, value in current.values.items():
        recorded = torch.as_tensor(values[name], device=device, dtype=value.dtype)
        if not torch.equal(value, recorded):
            if name!='tet_volume':raise ValueError('Mixed derived scene changed: ' + name)
            dm=torch.as_tensor(values['tet_Dm'],device=device,dtype=current.tet_Dm.dtype)
            if not torch.equal(dm,current.tet_Dm):raise ValueError('Mixed inverse rest geometry changed')
            from .derived_geometry import certify_tet_volume_roundoff
            derived_roundoff[name]=certify_tet_volume_roundoff(value,recorded,dm)
        static[name] = recorded
    canonical_volume=static['tet_volume']
    scene = MixedContactScene(static, metadata)
    if not torch.equal(scene.tet_volume,canonical_volume):
        from .derived_geometry import certify_tet_volume_roundoff
        recorded=canonical_volume
        derived_roundoff['tet_volume']=certify_tet_volume_roundoff(scene.tet_volume,recorded,scene.tet_Dm)
        scene.values['tet_volume']=recorded
        scene.objective_scene.tet_volume=recorded
        scene.elastic_scene.tet_volume=recorded
    scene.current_device_static_roundoff = roundoff
    scene.current_device_derived_roundoff=derived_roundoff
    scene.geometry_oracle = MixedCollisionOracle(example.model, scene, wp, newton)
    keys = ('x', 'q', 'v', 'omega', 'history__pin_x_prev', 'history__pin_initialized', 'input_pin_x', 'input_force', 'input_torque')
    states = {name: torch.as_tensor(values[name], device=state_device or device) for name in keys}
    if any(not bool(torch.isfinite(value).all()) for value in states.values()): raise ValueError('Nonfinite mixed training view')
    return scene, states


def apply_pin_input(scene, state, states, index):
    result=State(state.x.index_copy(1, scene.fixed_particle_ids, states['input_pin_x'][index:index+1]), state.q, state.v, state.omega, state.history)
    if scene.metadata.get('prescribed_pose_history'):
        from .general_joint_data import apply_prescribed_input
        result=apply_prescribed_input(scene,result,states,index)
    return result


def query(scene, state, wp):
    def run():
        scene.candidates = scene.geometry_oracle.query(state)
        scene.candidate_kind = scene.candidates.relation_kind
    if state.x.is_cuda:
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())): run()
    else: run()
