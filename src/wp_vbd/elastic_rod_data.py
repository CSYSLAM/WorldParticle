"""Causal native mixed-asset data with accepted shared material history."""
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import torch

from . import elastic_rod_system as physics,math3d as so3
from .data import State
from .elastic_rod_demo import build_asset
from .elastic_rod_scene import CONTRACT,export_model
from .elastic_rod_runtime import ElasticRodCollisionOracle
from .reference_tape import arrays

SCHEMA='accepted_elastic_rod_inputs_and_history_v226'
TIME_INPUTS=('input_pin_x','input_joint_target','input_joint_target_rate','input_joint_effort',
             'input_force','input_torque')


def convert(scene,particle_q,particle_qd,body_q,body_qd):
    n=scene.metadata['particle_count'];b=scene.metadata['body_count']
    x=scene.rest_x.new_zeros(1,scene.n,3);q=scene.rest_q.new_zeros(1,scene.n,4);q[...,3]=1.
    v=torch.zeros_like(x);omega=torch.zeros_like(x)
    x[0,:n]=torch.as_tensor(particle_q,device=scene.device,dtype=torch.float64)
    v[0,:n]=torch.as_tensor(particle_qd,device=scene.device,dtype=torch.float64)
    pose=torch.as_tensor(body_q,device=scene.device,dtype=torch.float64)
    velocity=torch.as_tensor(body_qd,device=scene.device,dtype=torch.float64)
    if pose.shape!=(b,7) or not bool((pose[:,3:].norm(dim=-1)>1e-12).all()):
        raise ValueError('Finite nondegenerate native body pose layout required')
    q[0,n:n+b]=so3.normalize(pose[:,3:])
    x[0,n:n+b]=pose[:,:3]+so3.rotate(q[0,n:n+b],scene.body_com[n:n+b])
    v[0,n:n+b]=velocity[:,:3];omega[0,n:n+b]=velocity[:,3:]
    return State(x,q,v,omega)


def verify_source(model,source):
    """Exact physical source fields; validate device-owned BVH roots locally."""
    current=arrays(model)
    expected={k.removeprefix('source__'):v for k,v in source.items() if k.startswith('source__')}
    expected.pop('joint_target_q_start',None)  # Public alias is not in vars(model).
    if set(current)!=set(expected):
        raise ValueError('Native source descriptor fields changed: '+str(set(current)^set(expected)))
    acceleration={}
    for name,value in current.items():
        if np.array_equal(value,expected[name]):continue
        if model.device.is_cuda and name in ('bvh_particles_group_roots','bvh_shapes_group_roots'):
            # The pinned ModelBuilder uses device-dependent BVH constructors.
            # Node indices belong to that live tree, not the physical asset.
            # Restoring CPU indices into a GPU tree would corrupt traversal.
            import warp as wp
            from newton._src.geometry.bvh import compute_bvh_group_roots
            tree=getattr(model,name.removesuffix('_group_roots'));roots=getattr(model,name)
            if value.shape!=expected[name].shape or roots.dtype!=wp.int32 or len(roots)!=model.world_count+1:
                raise ValueError('Native acceleration world layout changed: '+name)
            verified=wp.empty_like(roots)
            wp.launch(compute_bvh_group_roots,dim=len(roots),inputs=[tree.id,verified],device=model.device)
            if not np.array_equal(verified.numpy(),value):raise ValueError('Native acceleration roots disagree with their own tree: '+name)
            acceleration[name]={'reference_cpu_roots':expected[name].tolist(),'native_device_roots':value.tolist(),
                'verified_against_current_device_tree':True}
            continue
        raise ValueError('Native mixed source field changed: '+name)
    model._wp_vbd_verified_acceleration_differences=acceleration
    return len(current)


def load_reference(path,wp,newton,device='cpu',*,state_device=None,allow_short_probe=False,
                   allow_definition_mismatch_probe=False):
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        values={key:archive[key].copy() for key in archive.files if key!='metadata'}
    if metadata.get('contract')!=CONTRACT or metadata.get('training_schema')!=SCHEMA or not metadata.get('reference_format_qualified'):
        raise ValueError('Qualified explicit elastic/rod reference required')
    if not metadata.get('eligible_normal_course_length',metadata.get('eligible_normal_course')) and not allow_short_probe:
        raise ValueError('Short mixed reference probes cannot enter a normal training course')
    names=('x','q','v','omega',*('history__'+name for name in sorted(physics.HISTORY_KEYS)),*TIME_INPUTS)
    if set(names)-set(values):raise ValueError('Missing explicit mixed temporal fields')
    if {k for k in values if k.startswith('history__')}!={'history__'+k for k in physics.HISTORY_KEYS}:
        raise ValueError('Unknown or missing mixed accepted history channels')
    if {k for k in values if k.startswith('input_')}!=set(TIME_INPUTS):
        raise ValueError('Unknown or missing exogenous mixed input channels')
    newton_root=Path('/home/oem/code/engine/newton')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=newton_root,text=True).strip()!=metadata['newton_sha']:
        raise ValueError('Pinned Newton revision changed')
    factory=Path(__file__).with_name('native_assets.py' if metadata.get('asset_definition') else 'elastic_rod_demo.py')
    if hashlib.sha256(factory.read_bytes()).hexdigest()!=metadata['factory_sha256']:
        raise ValueError('Qualified native mixed asset factory changed')
    trace=Path(metadata['source_trace_path'])/'manifest.json'
    if hashlib.sha256(trace.read_bytes()).hexdigest()!=metadata['source_trace_manifest_sha256']:
        raise ValueError('Source teacher trace changed')
    if metadata.get('asset_definition'):
        from .native_asset_training_definition import rebuild
        from .elastic_rod_configuration import record_configuration,decode_configuration
        model,built_configuration,provenance=rebuild(metadata['asset_definition'],newton,wp,device=device)
        if provenance['dt']!=metadata['dt'] or provenance['motion_budget']!=metadata['self_contact_motion_budget']:
            raise ValueError('Authored reference physical timestep or motion envelope changed')
        if record_configuration(built_configuration)!=record_configuration(decode_configuration(metadata['solver_configuration'])):
            raise ValueError('Authored reference physical solver configuration changed')
    else:model,_=build_asset(newton,wp,device=device,**metadata['physical_parameters'])
    from .surface_joint_data import restore_joint_source
    roundoff=restore_joint_source(model,values)
    # Derived native input arrays have already been restored and all other
    # numeric source fields must still match. No geometry is rescaled here.
    verify_source(model,values)
    from .elastic_rod_configuration import decode_configuration,topology_definition
    from .self_contact_query import SelfContactQuery
    configuration=decode_configuration(metadata['solver_configuration'])
    scene=export_model(model,configuration,metadata['official_demo'],newton,
        metadata['dt'],device,motion_budget=metadata['self_contact_motion_budget'],
        full_surface=metadata['self_shape_full_surface'],max_rows=metadata['self_contact_max_rows'])
    for prefix,fields in [('scene__',scene.values),('joint__',vars(scene.joint_spec)),('rod__',vars(scene.rod))]:
        for name,current in fields.items():
            recorded=torch.as_tensor(values[prefix+name],device=device,dtype=current.dtype)
            if not torch.equal(current,recorded):
                if prefix!='scene__' or name!='tet_volume':
                    raise ValueError('Qualified mixed descriptor changed: '+prefix+name)
                from .derived_geometry import certify_tet_volume_roundoff
                roundoff['tet_volume']=certify_tet_volume_roundoff(current,recorded,scene.tet_Dm)
                scene.values['tet_volume']=recorded
                scene.objective_scene.tet_volume=recorded;scene.elastic_scene.tet_volume=recorded
    scene.metadata={**metadata,'solver_configuration':configuration}
    scene.current_device_static_roundoff=roundoff
    scene.current_device_acceleration_differences=model._wp_vbd_verified_acceleration_differences
    states={name:torch.as_tensor(values[name],device=state_device or device) for name in names}
    t,n,j=metadata['teacher_substeps'],scene.n,len(scene.joint_spec.owners)
    expected={'x':(t+1,n,3),'q':(t+1,n,4),'v':(t+1,n,3),'omega':(t+1,n,3),
        'input_pin_x':(t,len(scene.fixed_particle_ids),3),'input_force':(t,n,3),'input_torque':(t,n,3)}
    expected.update({k:(t,j,6) for k in TIME_INPUTS if k.startswith('input_joint_')})
    initial=physics.initial_history(scene,convert(scene,model.particle_q.numpy(),model.particle_qd.numpy(),
                                                  model.body_q.numpy(),model.body_qd.numpy()))
    expected.update({'history__'+k:(t+1,*v.shape[1:]) for k,v in initial.items()})
    for name,value in states.items():
        if tuple(value.shape)!=expected[name] or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid mixed temporal field: '+name)
        boolean=name in ('history__pin_initialized','history__rod_initialized')
        if value.dtype!=(torch.bool if boolean else torch.float64):raise ValueError('Invalid mixed temporal dtype: '+name)
    initialized=states['history__rod_initialized']
    if bool(initialized[0]) or not bool(initialized[1:].all()) or not bool(states['history__pin_initialized'].all()):
        raise ValueError('Invalid accepted mixed history initialization')
    for name in ('x','q'):
        if not torch.equal(states['history__rod_previous_'+name],states[name]):
            raise ValueError('Mixed material history is not the accepted current pose: '+name)
    pin_ids=scene.fixed_particle_ids.to(states['x'].device)
    if not torch.equal(states['history__pin_x_prev'],states['x'][:,pin_ids]):
        raise ValueError('Fixed-pin history changed during the constant-input episode')
    query=SelfContactQuery(model,configuration,device=device,motion_budget=metadata['self_contact_motion_budget'])
    definition=topology_definition(query,configuration)
    declared=metadata.get('source_self_contact_definition')
    if declared is not None and declared!=definition:raise ValueError('Native self-contact force-scope evidence changed')
    scene.metadata['source_self_contact_definition']=definition
    scene.metadata['eligible_normal_course']=bool(metadata['eligible_normal_course'] and definition['compatible'])
    if not definition['compatible'] and not allow_definition_mismatch_probe:
        raise ValueError('Native directional self-contact force ownership differs from the reciprocal model; diagnostic-only data')
    scene.metadata['definition_mismatch_probe']=not definition['compatible']
    scene.geometry_oracle=ElasticRodCollisionOracle(model,scene,wp,newton)
    return scene,states


def step(network,scene,current,states,index,wp,*,before_model=None):
    """One own geometry query and prediction; labels and future history unread."""
    from .self_contact_refresh import enabled as refresh_self_geometry
    from .self_contact_repair import configure
    configure(scene,network.config)
    if scene.metadata.get('source_self_contact_definition',{}).get('compatible') and not refresh_self_geometry(network.config,scene.metadata.get('contract')):
        raise ValueError('This source-consistent mixed reference requires current self-contact geometry')
    physics.validate_history(scene,current)
    device=scene.device
    commands=[states[key][index:index+1].to(device) for key in TIME_INPUTS]
    pins,target,rate,effort,force,torque=commands
    scene.joint_controls=(target,rate,effort)
    commanded=State(current.x.index_copy(1,scene.fixed_particle_ids,pins),current.q,
                     current.v,current.omega,current.history)
    candidates=scene.geometry_oracle.query(commanded)
    scene.candidates=candidates;scene.candidate_kind=candidates.relation_kind
    if before_model is not None:before_model()
    predicted,diagnostic=network(scene,commanded,scene.metadata['dt'],force=force,torque=torque)
    return (predicted if bool(diagnostic['accepted'].all()) else current),diagnostic
