"""Qualified owned rod/contact data with causal COM and material history."""
import json
import functools

import numpy as np
import torch

from .rod_contact_scene import CONTRACT, export_model
from .rigid_contact_runtime import CollisionOracle

SOURCE_FIELDS = ('body_q', 'body_mass', 'body_inv_mass', 'body_inertia', 'body_inv_inertia', 'body_com',
                 'body_flags', 'body_world', 'gravity', 'joint_type', 'joint_parent', 'joint_child',
                 'joint_X_p', 'joint_X_c', 'joint_axis', 'joint_q_start', 'joint_qd_start', 'joint_enabled',
                 'joint_target_q', 'joint_target_qd', 'joint_target_ke', 'joint_target_kd',
                 'shape_body', 'shape_type', 'shape_transform', 'shape_scale', 'shape_margin', 'shape_gap',
                 'shape_collision_radius', 'shape_world', 'shape_flags', 'shape_material_ke', 'shape_material_kd',
                 'shape_material_mu', 'shape_collision_filter_pairs')
FINITE_SOURCE_FIELDS=('joint_dof_dim','joint_target_q_start','joint_q','joint_limit_lower',
                      'joint_limit_upper','joint_limit_ke','joint_limit_kd')


def verify_caller_configuration(configuration,recorded,newton):
    from .rod_adapter import native_configuration
    from .official import solver_configuration_record
    current=solver_configuration_record(native_configuration(configuration,newton))
    if current!=recorded:
        changed=sorted(key for key in current.keys()|recorded.keys() if key not in current or key not in recorded or current[key]!=recorded[key])
        raise ValueError('Caller source physical configuration changed: '+','.join(changed))
    return True


def load_reference(path, wp, newton, device='cpu', *, state_device=None):
    from .official import make_example
    from .reference_tape import arrays
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive['metadata']))
        values = {key: archive[key].copy() for key in archive.files if key != 'metadata'}
    from .rod_joint_scene import CONTRACT as composed_contract
    composed=metadata['contract']==composed_contract
    from .rigid_caller_program import SCHEMA as caller_schema
    caller=metadata.get('rigid_caller_training_schema')
    if caller is not None and caller!=caller_schema:raise ValueError('Unknown rigid caller training schema')
    if caller:
        if not composed or metadata.get('rigid_mesh_reference_schema')!='authored_sdf_mesh_identity_and_reference_mass_v216':
            raise ValueError('Complete caller mesh/reference identity required')
        if torch.device(device).type!='cuda':raise ValueError('Authored SDF-mesh caller loading requires CUDA; CPU descriptors cannot collide')
        if metadata.get('matched_contact_training_schema') or metadata.get('prescribed_geometry_removal_training_schema'):
            raise ValueError('Caller mesh data does not admit matched caches or geometry-removal events')
    if metadata['contract'] not in (CONTRACT,composed_contract) or metadata['reference_final_check'] != 'PASS':
        raise ValueError('Complete qualified rod/contact data required')
    if composed and metadata.get('finite_rod_training_schema')!='accepted_joint_controls_winding_v144':
        raise ValueError('Composed rod/joint data requires complete controls and accepted winding')
    if composed and not metadata.get('prescribed_rod_training_schema'):
        raise ValueError('Composed rod/joint data requires explicit prior poses and current commands')
    if bool(metadata.get('rod_finite_joint_contract'))!=composed:
        raise ValueError('Rod/finite-joint scene and history contracts differ')
    if not any(metadata.get(name) for name in ('zero_Dahl_unprescribed_training_schema','zero_Dahl_stationary_training_schema','zero_Dahl_prescribed_fresh_training_schema','dahl_prescribed_fresh_training_schema')):
        raise ValueError('Dahl/externally prescribed training states need the extended causal preparation schema')
    if caller:
        requests=[];original_build=newton.Mesh.build_sdf
        @functools.wraps(original_build)
        def tracked_build(mesh,*args,**kwargs):
            requests.append({'arguments':list(args),'options':kwargs});return original_build(mesh,*args,**kwargs)
        newton.Mesh.build_sdf=tracked_build
        try:
            with wp.ScopedDevice(str(device)):
                example,solvers,_=make_example(metadata['official_demo'],reference=False,preserve_defaults=True)
        finally:newton.Mesh.build_sdf=original_build
        if json.loads(json.dumps(requests))!=metadata['source_constructor_sdf_requests']:
            raise ValueError('Authored mesh SDF build parameters changed')
        from .rigid_mesh_reference import mesh_identity,restore_mass_inputs,filter_pairs,MASS_FIELDS
        if mesh_identity(example.model,newton,require_sdf=True)!=metadata['mesh_geometry_identity']:
            raise ValueError('Authored mesh vertex/index identity changed')
        restore_mass_inputs(example.model,{name:values['source__'+name] for name in MASS_FIELDS})
    else:example, solvers, _ = make_example(metadata['official_demo'], reference=False, preserve_defaults=True)
    if len(solvers) != 1: raise ValueError('One owned source subsystem required')
    source = arrays(example.model)
    if composed:source['joint_target_q_start']=example.model.joint_target_q_start.numpy()
    if caller:
        source['shape_collision_filter_pairs']=filter_pairs(example.model)
        verify_caller_configuration(solvers[0][1],metadata['solver_configuration'],newton)
        if example.sim_dt!=metadata['dt'] or example.sim_substeps*metadata['teacher_frames']!=metadata['teacher_substeps']:
            raise ValueError('Caller reference physical clock differs from the authored scene')
    contact_policy=None
    if metadata.get('matched_contact_training_schema'):
        from .rigid_contact_memory import authored_policy
        contact_policy=authored_policy(example)
        if metadata.get('matched_contact_training_schema')!='accepted_matcher_cache_v133' or contact_policy!=metadata.get('contact_matching_policy'):
            raise ValueError('Qualified authored matching policy changed')
        source['shape_collision_filter_pairs']=example.model.shape_collision_filter_pairs.pairs_array()
    for name in (*SOURCE_FIELDS,*(FINITE_SOURCE_FIELDS if composed else ())):
        if 'source__'+name not in values:
            if name in source: raise ValueError('Source field appeared: '+name)
            continue
        if name not in source or not np.array_equal(source[name], values['source__'+name]):
            raise ValueError('Authored rod/contact source changed: '+name)
    exporter=export_model
    if composed:
        from .rod_joint_scene import export_model as exporter
    options={'contact_policy':contact_policy}
    if caller:options.update(sdf_mesh_contacts=True,massless_prescribed=True)
    bounded=metadata.get('bounded_analytic_contact_buffers')
    if bounded is not None:
        if bounded!='current_expanded_aabb_v146':raise ValueError('Unknown bounded contact allocation policy')
        options['bounded_analytic_buffers']=True
    scene = exporter(example.model, solvers[0][1], metadata['official_demo'], newton, metadata['dt'], device,**options)
    if metadata.get('zero_Dahl_stationary_training_schema'):
        fixed = np.flatnonzero(~scene.movable[:-1].cpu().numpy()).tolist()
        if fixed != metadata['stationary_body_indices']:
            raise ValueError('Stationary prescribed body ownership changed')
    prescribed=metadata.get('prescribed_rod_training_schema') or metadata.get('zero_Dahl_prescribed_fresh_training_schema')
    if prescribed:
        if np.flatnonzero(~scene.movable[:-1].cpu().numpy()).tolist()!=metadata['prescribed_body_indices']:
            raise ValueError('Prescribed rod body ownership changed')
    for name, value in scene.values.items():
        recorded = torch.as_tensor(values[name], device=device, dtype=value.dtype)
        if not torch.equal(recorded, value): raise ValueError('Derived rod/contact asset changed: '+name)
    for name, value in vars(scene.rod).items():
        recorded = torch.as_tensor(values['rod__'+name], device=device, dtype=value.dtype)
        if not torch.equal(recorded, value): raise ValueError('Derived rod/contact material changed: '+name)
    if composed:
        for name,value in vars(scene.joint_spec).items():
            recorded=torch.as_tensor(values['joint__'+name],device=device,dtype=value.dtype)
            if not torch.equal(recorded,value):raise ValueError('Derived finite-joint coordinate/material changed: '+name)
    scene.metadata = metadata
    if caller:
        from .caller_program_io import from_values,from_example,to_values
        scene.caller_program=from_values(scene,values)
        expected_program=to_values(from_example(scene,example));recorded_program=to_values(scene.caller_program)
        if any(not np.array_equal(value,recorded_program[name]) for name,value in expected_program.items()):
            raise ValueError('Serialized caller program differs from authored control geometry')
    scene.geometry_oracle = CollisionOracle(example.model, scene, wp, newton)
    names = ('x', 'q', 'v', 'omega', 'history__rod_dual', 'history__rod_kappa', 'history__rod_increment',
             'history__rod_stress', 'history__rod_previous_x', 'history__rod_previous_q', 'history__rod_initialized')
    if not caller:names+=('input_force','input_torque')
    state_device=device if state_device is None else state_device
    states = {name:torch.as_tensor(values[name], device=state_device) for name in names}
    if caller:
        from .caller_program_io import time_channels
        states.update(time_channels(scene,values,state_device))
    if composed:
        for name in ('history__joint_angle_prev','input_joint_target','input_joint_target_rate','input_joint_effort'):
            states[name]=torch.as_tensor(values[name],device=state_device)
            expected=(metadata['teacher_substeps']+(name.startswith('history__')),len(scene.joint_spec.owners))
            if not name.startswith('history__'):expected+= (6,)
            if states[name].shape!=expected or states[name].dtype!=torch.float64:
                raise ValueError('Complete float64 finite-joint time channel required: '+name)
    if metadata.get('matched_contact_training_schema'):
        from .rigid_contact_memory import history_keys
        for name in history_keys(scene):states['history__'+name]=torch.as_tensor(values['history__'+name],device=state_device)
    if prescribed and not caller:
        for name in ('input_pin_x','input_pin_q'):
            states[name]=torch.as_tensor(values[name],device=state_device)
        scene.prescribed_body_ids=torch.tensor(metadata['prescribed_body_indices'],device=device,dtype=torch.long)
    if metadata.get('prescribed_geometry_removal_training_schema'):
        from .rod_prescribed_events import validate_removal_bodies
        validate_removal_bodies(scene,scene.prescribed_body_ids)
        mask=torch.as_tensor(values['input_geometry_removal_mask'],device=state_device)
        if mask.dtype!=torch.bool or mask.shape!=states['input_pin_x'].shape[:2]:
            raise ValueError('Complete explicit geometry removal inputs required')
        states['input_geometry_removal_mask']=mask
    if any(not bool(torch.isfinite(value).all()) for value in states.values()):
        raise ValueError('Nonfinite rod/contact state or input')
    return scene, states


def apply_prescribed_input(scene, state, states, index):
    if scene.metadata.get('rigid_caller_training_schema'):
        raise ValueError('Use the causal caller-window step; recorded pin poses are not inputs')
    if scene.metadata.get('finite_rod_training_schema'):
        scene.joint_controls=tuple(states[name][index:index+1] for name in
            ('input_joint_target','input_joint_target_rate','input_joint_effort'))
    if not (scene.metadata.get('prescribed_rod_training_schema') or scene.metadata.get('zero_Dahl_prescribed_fresh_training_schema')):return state
    from .data import State
    x=state.x.index_copy(1,scene.prescribed_body_ids,states['input_pin_x'][index:index+1])
    q=state.q.index_copy(1,scene.prescribed_body_ids,states['input_pin_q'][index:index+1])
    result=State(x,q,state.v,state.omega,state.history)
    if scene.metadata.get('prescribed_geometry_removal_training_schema'):
        if len(state.x)!=1:raise ValueError('Explicit boundary-event training currently requires one sequence')
        from .rod_prescribed_events import rebase_removed_geometry
        ids=scene.prescribed_body_ids[states['input_geometry_removal_mask'][index]]
        result=rebase_removed_geometry(scene,result,ids)
    return result


def query(scene, state, wp):
    def run(): scene.candidates = scene.geometry_oracle.query(state)
    if state.x.is_cuda:
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())): run()
    else: run()
