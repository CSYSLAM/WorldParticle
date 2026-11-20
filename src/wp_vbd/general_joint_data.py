"""Joint data with recorded physical parameters, controls and winding state."""
import json
from types import SimpleNamespace
import numpy as np
import torch
from . import math3d as so3
from .general_joint_scene import export_model,JointContactScene
from .rigid_contact_runtime import CollisionOracle


def load_reference(path,wp,newton,device='cuda:0',*,state_device=None):
    from .official import make_example
    with np.load(path,allow_pickle=False) as archive:
        values={key:archive[key].copy() for key in archive.files if key!='metadata'};metadata=json.loads(str(archive['metadata']))
    if metadata['reference_final_check']!='PASS' or metadata['contract']!='rigid_joint_contact_finite_v26':raise ValueError('Admitted finite-joint reference required')
    example,solvers,_=make_example(metadata['official_demo'],reference=False,preserve_defaults=metadata.get('native_preserve_defaults',False),arguments=metadata.get('authored_arguments',()),host_scalar_compat=metadata.get('native_host_scalar_compat',False))
    for key,recorded in values.items():
        if not key.startswith('source__'):continue
        name=key.removeprefix('source__');current_value=getattr(example.model,name).numpy()
        if name=='body_q':
            if float(np.abs(current_value-recorded).max())>1e-6:raise ValueError('FK rest-pose difference exceeds float32 bound')
            example.model.body_q.assign(recorded)
        elif name=='joint_q' and metadata.get('native_host_scalar_compat'):
            from .asset_roundoff import restore_free_quaternions
            restore_free_quaternions(example.model,recorded,newton)
        elif not np.array_equal(current_value,recorded):raise ValueError('Authored joint asset changed: '+name)
    current=export_model(example.model,solvers[0][1],metadata['official_demo'],newton,example.sim_dt,device,native_contact_export=bool(metadata.get('native_contact_export')),prescribed_motion=bool(metadata.get('prescribed_pose_history')),packed_worlds=bool(metadata.get('world_gravity_contract')))
    if current.metadata['solver_configuration']!=metadata['solver_configuration']:raise ValueError('Joint reference configuration changed')
    roundoff={}
    for key,value in current.values.items():
        reference=torch.as_tensor(values[key],device=device,dtype=value.dtype)
        difference=float((value-reference).abs().max())
        if difference:
            scale=max(float(reference.abs().max()),1e-30)
            if key not in {'mass','inertia','inverse_inertia','body_com','inverse_mass','body_radius','reference_length'} or difference>1e-5*scale:raise ValueError(f'Joint asset mismatch: {key}, difference={difference}')
            roundoff[key]=difference
    spec={}
    for key,value in vars(current.joint_spec).items():
        if torch.is_tensor(value):
            recorded=torch.as_tensor(values['joint__'+key],device=device)
            same=torch.equal(value,recorded) if not value.is_floating_point() else torch.allclose(value,recorded,rtol=1e-12,atol=1e-12,equal_nan=False)
            if not same:raise ValueError('Joint coordinate or material mismatch: '+key)
            spec[key]=recorded
    metadata['groups']=[{'name':f'body_{i}','family':'rigid','body_start':i,'body_count':1} for i in range(metadata['body_count'])]
    mimic=None
    if current.mimic_spec is not None:
        mimic={}
        for key,value in vars(current.mimic_spec).items():
            recorded=torch.as_tensor(values['mimic__'+key],device=device)
            if not torch.equal(value,recorded):raise ValueError('Mimic static property mismatch: '+key)
            mimic[key]=recorded
        mimic=SimpleNamespace(**mimic)
    scene=JointContactScene({key:values[key] for key in current.values},metadata,SimpleNamespace(**spec),device,mimic_spec=mimic)
    scene.current_device_static_roundoff=roundoff;scene.geometry_oracle=CollisionOracle(example.model,scene,wp,newton)
    states={}
    for key in ('x','q','v','omega'):
        value=torch.as_tensor(values[key],device=state_device or device).double()
        if key=='q':value=so3.normalize(value)
        world=value.new_zeros(len(value),1,value.shape[-1])
        if key=='q':world[...,3]=1
        states[key]=torch.cat((value,world),1)
    constants=set(metadata.get('constant_time_inputs',()))
    for key in ('history__joint_angle_prev','input_joint_target','input_joint_target_rate','input_joint_effort'):
        value=torch.as_tensor(values[key],device=state_device or device).double()
        if key in constants:
            if len(value)!=1:raise ValueError('Constant input must have exactly one stored time sample')
            value=value.expand(metadata['teacher_substeps'],*value.shape[1:])
        states[key]=value
    if metadata.get('prescribed_pose_history'):
        for key in ('history__prescribed_x_prev','history__prescribed_q_prev','input_prescribed_x','input_prescribed_q','input_prescribed_v','input_prescribed_omega'):
            value=torch.as_tensor(values[key],device=state_device or device).double()
            if key.endswith('_q') or key.endswith('_q_prev'):value=so3.normalize(value)
            states[key]=value
        states['history__prescribed_initialized']=torch.as_tensor(values['history__prescribed_initialized'],device=state_device or device,dtype=torch.bool)
    wrench=torch.as_tensor(values['force'],device=state_device or device).double();wrench=torch.cat((wrench,wrench.new_zeros(len(wrench),1,6)),1)
    if 'force' in constants:
        if len(wrench)!=1:raise ValueError('Constant force must have exactly one stored time sample')
        wrench=wrench.expand(metadata['teacher_substeps'],*wrench.shape[1:])
    states['input_force']=wrench[...,:3];states['input_torque']=wrench[...,3:]
    if not all(bool(torch.isfinite(value).all()) for value in states.values()):raise ValueError('Nonfinite joint reference state or controls')
    return scene,states


def set_controls(scene,states,index):
    scene.joint_controls=tuple(states[key][index:index+1].to(scene.device) for key in ('input_joint_target','input_joint_target_rate','input_joint_effort'))


def apply_prescribed_input(scene,current,states,index):
    if not scene.metadata.get('prescribed_pose_history'):return current
    from .data import State
    ids=scene.prescribed_ids
    return State(*(getattr(current,key).index_copy(1,ids,states['input_prescribed_'+key][index:index+1]) for key in ('x','q','v','omega')),current.history)
