"""Stateless rigid-contact reference inputs and independent query ownership."""
import json
import numpy as np
import torch
from . import math3d as so3
from .rigid_contact_scene import export_model,RigidContactScene
from .rigid_contact_runtime import CollisionOracle


def load_reference(path,wp,newton,device='cuda:0'):
    from .official import make_example
    with np.load(path,allow_pickle=False) as archive:
        values={key:archive[key].copy() for key in archive.files if key!='metadata'}
        metadata=json.loads(str(archive['metadata']))
    if metadata['reference_final_check']!='PASS' or metadata['cross_step_contact_history'] or metadata['contact_alpha']!=0:raise ValueError('Stateless accepted reference contract required')
    example,solvers,_=make_example(metadata['official_demo'],reference=False,preserve_defaults=metadata.get('native_preserve_defaults',False))
    static,exported=export_model(example.model,solvers[0][1],metadata['official_demo'],newton,example.sim_dt)
    # Mesh mass properties can differ at float32 reduction roundoff between
    # CPU and CUDA. Verify the authored asset, then keep the recorded physical
    # parameters exactly instead of replacing the teacher's inertia/COM.
    roundoff={};derived={'mass','inertia','inverse_inertia','body_com','inverse_mass','body_radius','reference_length'}
    for key,current in static.items():
        recorded=values[key]
        if np.array_equal(current,recorded):continue
        difference=float(np.max(np.abs(current-recorded)))
        scale=float(np.max(np.abs(recorded)))
        absolute=1e-7*float(values['reference_length']) if key in ('body_com','body_radius') else 0.
        if key not in derived or difference>1e-5*max(scale,1e-30)+absolute:raise ValueError(f'Reference asset mismatch: {key}, absolute difference={difference}')
        roundoff[key]={'maximum_absolute_difference':difference,'field_relative_difference':difference/max(scale,1e-30)}
    static={key:values[key] for key in static}
    if not exported['solver_configuration']==metadata['solver_configuration']:raise ValueError('Reference physical configuration changed')
    names=('mesh','cone','sphere','box','capsule','cylinder') if metadata['official_demo']=='vbd.example_vbd_rigid_rigid_contact' else [f'body_{i}' for i in range(example.model.body_count)]
    metadata['groups']=[{'name':name,'family':'rigid','body_start':i,'body_count':1} for i,name in enumerate(names)]
    scene=RigidContactScene(static,metadata,device=device)
    scene.current_device_static_roundoff=roundoff
    scene.geometry_oracle=CollisionOracle(example.model,scene,wp,newton)
    states={}
    for key in ('x','q','v','omega'):
        value=torch.as_tensor(values[key],device=device).double()
        if key=='q':value=so3.normalize(value)
        world=value.new_zeros(len(value),1,value.shape[-1])
        if key=='q':world[...,3]=1.
        states[key]=torch.cat((value,world),1)
    force=torch.as_tensor(values['force'],device=device).double()
    force=torch.cat((force,force.new_zeros(len(force),1,6)),1)
    states['input_force']=force[...,:3];states['input_torque']=force[...,3:]
    if not all(bool(torch.isfinite(value).all()) for value in states.values()):raise ValueError('Nonfinite contact labels')
    return scene,states


def query(scene,state,wp):
    """Freeze a fresh geometry snapshot on the same stream as Torch physics."""
    if state.x.is_cuda:
        with wp.ScopedStream(wp.stream_from_torch(torch.cuda.current_stream())):
            scene.candidates=scene.geometry_oracle.query(state)
    else:scene.candidates=scene.geometry_oracle.query(state)
