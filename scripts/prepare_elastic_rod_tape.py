"""Qualify accepted mixed states/history from an offline native VBD trace."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def qualify(args):
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd import elastic_rod_system as physics,rod_contact_system as rod,mixed_contact_system as material
    from wp_vbd.elastic_rod_data import SCHEMA,convert,verify_source
    from wp_vbd.elastic_rod_demo import build_asset
    from wp_vbd.elastic_rod_scene import export_model
    from wp_vbd.joint_history import begin_dahl,commit_dahl,DahlMemory
    from wp_vbd.elastic_rod_configuration import decode_configuration,record_configuration,topology_definition
    from wp_vbd.reference_tape import read_tape,arrays
    from wp_vbd.self_contact_query import SelfContactQuery
    from wp_vbd.data import State
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    manifest_path=args.tape/'manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='PHYSICAL_SUBSTEP_TRACE_READY' or manifest['id']!='research.native_five_families':
        raise ValueError('Complete actual native five-family teacher required')
    if manifest['vbd_instances']!=1 or manifest['source_sha']!='25298aea1691e060c86f95e8d2b773c898c006d3':
        raise ValueError('One owned pinned native VBD teacher required')
    expected=manifest['frames_requested']*manifest['substeps_per_frame']
    if manifest['frames_completed']!=manifest['frames_requested'] or manifest['physical_vbd_calls']!=expected or manifest['reference_step_calls']!=expected:
        raise ValueError('Missing or failed native reference calls')
    factory=ROOT/'src/wp_vbd'/('native_assets.py' if manifest.get('asset_definition') else 'elastic_rod_demo.py')
    if hashlib.sha256(factory.read_bytes()).hexdigest()!=manifest['factory_sha256']:
        raise ValueError('Native mixed asset factory changed')
    if not manifest['shape_full_surface']:raise ValueError('Full source shape-contact feature eligibility required')
    calls={'construct':0,'step':0,'shape_collision':0}
    original_constructor=newton.solvers.SolverVBD.__init__
    @functools.wraps(original_constructor)
    def forbidden(*unused,**kwargs):calls['construct']+=1;raise AssertionError('No VBD construction during qualification')
    def forbidden_step(*unused,**kwargs):calls['step']+=1;raise AssertionError('No VBD step during qualification')
    def forbidden_collision(*unused,**kwargs):calls['shape_collision']+=1;raise AssertionError('No shape collision oracle during qualification')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden_step
    newton.CollisionPipeline.__init__=forbidden_collision
    if manifest.get('asset_definition'):
        from wp_vbd.native_asset_training_definition import rebuild
        asset,built_configuration,provenance=rebuild(manifest['asset_definition'],newton,wp)
        if provenance['dt']!=manifest['dt'] or provenance['substeps']!=manifest['substeps_per_frame'] or provenance['motion_budget']!=args.motion_budget:
            raise ValueError('Native authored time input changed')
        if record_configuration(built_configuration)!=record_configuration(decode_configuration(manifest['configuration'])):
            raise ValueError('Native authored physical configuration changed')
    else:asset,_=build_asset(newton,wp,**manifest['physical_parameters'])
    static_path=args.tape/manifest['static']['path']
    if hashlib.sha256(static_path.read_bytes()).hexdigest()!=manifest['static']['sha256']:
        raise ValueError('Native teacher static archive changed')
    with np.load(static_path,allow_pickle=False) as archive:
        source={key.replace('solver0__model__','source__'):archive[key].copy()
                for key in archive.files if key.startswith('solver0__model__')}
    source['source__joint_target_q_start']=source['source__joint_q_start'].copy()
    source_count=verify_source(asset,source)
    configuration=decode_configuration(manifest['configuration']);dt=manifest['dt']
    scene=export_model(asset,configuration,manifest['id'],newton,dt,motion_budget=args.motion_budget,
                       full_surface=True)
    geometry=SelfContactQuery(asset,configuration,motion_budget=args.motion_budget,
                              device='cpu',max_rows=scene.metadata['self_contact_max_rows'])
    definition=topology_definition(geometry,configuration)
    if manifest.get('source_self_contact_definition',definition)!=definition:
        raise ValueError('Source self-contact definition evidence changed')
    defaults=arrays(asset.control());pins=scene.fixed_particle_ids
    ids=scene.rod.source_joint_ids.numpy();spec=scene.joint_spec
    valid=spec.dof_index.numpy()>=0
    previous=None;last_dual=None;states=[];histories=[];forces=[];torques=[];pin_inputs=[];controls=[]
    maximum_motion=0.;minimum_tet_det=float('inf');count=0
    memory=None;angle=None
    def dual(data,prefix):
        return np.concatenate((data[prefix+'__joint_lambda_lin'][ids],data[prefix+'__joint_lambda_ang'][ids]),-1).astype(np.float64)
    for data,timing in read_tape(args.tape):
        if timing['sequence']!=count or timing['solver_slot']!=0 or timing['dt']!=dt or timing['observables_requested']:
            raise ValueError('Native reference timing/ownership changed')
        if timing.get('controller')!='constant_physical_parameters_and_fixed_pins':
            raise ValueError('Unknown native mixed controller inputs')
        incoming=tuple(data['input__'+key] for key in ('particle_q','particle_qd','body_q','body_qd'))
        output=tuple(data['output__'+key] for key in ('particle_q','particle_qd','body_q','body_qd'))
        if previous is not None and not all(np.array_equal(a,b) for a,b in zip(previous,incoming)):
            raise ValueError(f'Unrepresented native state edit before call {count}')
        if last_dual is not None and not np.array_equal(last_dual,dual(data,'aux_before')):
            raise ValueError('Rod numerical history did not continue from accepted past')
        for name,value in source.items():
            key='model_input__'+name.removeprefix('source__')
            if key in data and not np.array_equal(data[key],value):raise ValueError('Unrepresented mutable input: '+key)
        if {k.removeprefix('control__') for k in data if k.startswith('control__')}!=set(defaults):
            raise ValueError('Native controller descriptor fields changed')
        for name,value in defaults.items():
            if not np.array_equal(data['control__'+name],value):raise ValueError('Unrepresented controller change: '+name)
        start=convert(scene,*incoming);final=convert(scene,*output)
        if not all(bool(torch.isfinite(v).all()) for v in (start.x,start.q,start.v,start.omega,final.x,final.q,final.v,final.omega)):
            raise ValueError('Nonfinite mixed native states')
        if not torch.equal(start.x[:,pins],final.x[:,pins]) or not torch.equal(start.x[:,pins],scene.rest_x[pins][None]):
            raise ValueError('Unrepresented fixed-pin motion')
        if count==0:
            if len(scene.tets):
                points=start.x[:,scene.tets]
                ds=torch.stack([points[:,:,i]-points[:,:,0] for i in (1,2,3)],-1)
                minimum_tet_det=float(torch.linalg.det(ds@scene.tet_Dm).min())
            initial=physics.initial_history(scene,start)
            initial['rod_dual']=torch.from_numpy(dual(data,'aux_before'))[None]
            histories.append(initial);states.append(start)
            memory=DahlMemory(initial['rod_kappa'],initial['rod_increment'],initial['rod_stress'])
            angle=initial['joint_angle_prev']
        travel=(final.x-start.x).norm(dim=-1)
        from wp_vbd import math3d as so3
        travel=travel+scene.body_radius*so3.log(so3.multiply(final.q,so3.conjugate(start.q))).norm(dim=-1)
        maximum_motion=max(maximum_motion,float(travel.max()))
        if not bool((travel[:,scene.movable]<=args.motion_budget).all()):
            raise ValueError(f'Teacher call {count} exceeds shared motion envelope: {float(travel.max())}')
        if not bool((material.path_feasible(scene,start,final)&rod.path_feasible(scene,start,final)).all()):
            raise ValueError(f'Teacher material path leaves the local domain at call {count}')
        snapshot=geometry.snapshot(start.x[0,:asset.particle_count])
        if not bool(snapshot.path_feasible(final.x[0,:asset.particle_count])):
            raise ValueError(f'Teacher particle path lacks a complete VF/EE certificate at call {count}')
        if len(scene.tets):
            points=final.x[:,scene.tets]
            ds=torch.stack([points[:,:,i]-points[:,:,0] for i in (1,2,3)],-1)
            minimum_tet_det=min(minimum_tet_det,float(torch.linalg.det(ds@scene.tet_Dm).min()))
        old=rod.strain(scene,start)[...,3:];new=rod.strain(scene,final)[...,3:]
        enabled=torch.ones_like(old[...,0],dtype=torch.bool)
        frozen,_,_=begin_dahl(memory,old,scene.rod.joint_stiffness[:,3:],scene.rod.rod_dahl_eps,
                              scene.rod.rod_dahl_tau,enabled,torch.full_like(enabled,count==0))
        memory=commit_dahl(frozen,new,scene.rod.joint_stiffness[:,3:],scene.rod.rod_dahl_eps,
                           scene.rod.rod_dahl_tau,enabled)
        value=scene.joint_coordinates(final,angle,jacobian=False).axis[...,3]
        angle=torch.where(spec.angular_mode==1,value,0.)
        last_dual=dual(data,'aux_after')
        h={'rod_dual':torch.from_numpy(last_dual)[None],
           'rod_kappa':memory.kappa,'rod_increment':memory.increment,'rod_stress':memory.stress,
           'rod_previous_x':final.x,'rod_previous_q':final.q,'rod_initialized':torch.ones(1,dtype=torch.bool),
           'pin_x_prev':final.x[:,pins],'pin_initialized':torch.ones(1,dtype=torch.bool),'joint_angle_prev':angle}
        physics.validate_history(scene,State(final.x,final.q,final.v,final.omega,h))
        histories.append(h);states.append(final);pin_inputs.append(start.x[:,pins][0].numpy())
        force=np.zeros((scene.n,3));torque=np.zeros_like(force)
        force[:asset.particle_count]=data['input__particle_f']
        wrench=data['input__body_f'];n=asset.particle_count;b=asset.body_count
        force[n:n+b]=wrench[:,:3];torque[n:n+b]=wrench[:,3:]
        forces.append(force);torques.append(torque)
        command=[]
        for name,index in [('joint_target_q',spec.target_index.numpy()),('joint_target_qd',spec.dof_index.numpy()),('joint_f',spec.dof_index.numpy())]:
            field=np.zeros(index.shape);field[valid]=data['control__'+name][index[valid]];command.append(field)
        controls.append(command);previous=output;count+=1
        if count%100==0:print(f'qualified_steps={count}/{expected}',flush=True)
    if count!=expected:raise ValueError('Incomplete native trace')
    payload={key:np.concatenate([getattr(row,key).numpy() for row in states]) for key in ('x','q','v','omega')}
    payload.update({'history__'+key:np.concatenate([h[key].numpy() for h in histories]) for key in physics.HISTORY_KEYS})
    payload.update(input_pin_x=np.asarray(pin_inputs),input_force=np.asarray(forces),input_torque=np.asarray(torques))
    for i,name in enumerate(('input_joint_target','input_joint_target_rate','input_joint_effort')):
        payload[name]=np.asarray([row[i] for row in controls])
    for prefix,fields in [('scene__',scene.values),('joint__',vars(scene.joint_spec)),('rod__',vars(scene.rod))]:
        payload.update({prefix+key:value.numpy() for key,value in fields.items()})
    payload.update(source)
    metadata={**scene.metadata,'solver_configuration':record_configuration(configuration),
        'training_schema':SCHEMA,'reference_format_qualified':True,
        'teacher_frames':manifest['frames_completed'],'teacher_substeps':count,
        'teacher_iterations':manifest['native_solver_iterations'],'newton_sha':manifest['source_sha'],
        'eligible_normal_course_length':manifest['frames_completed']>=300 and count*dt>=5.,
        'eligible_normal_course':manifest['frames_completed']>=300 and count*dt>=5. and definition['compatible'],
        'source_self_contact_definition':definition,
        'source_force_equivalence_certified':False,
        'source_contact_geometry_limitation':'Scope compatibility checks pair exclusions only. Native directed EE can select different closest witnesses in the two directions for parallel ties; the model uses one reciprocal canonical witness.',
        'factory_sha256':manifest['factory_sha256'],'physical_parameters':manifest['physical_parameters'],
        'source_trace_path':str(args.tape.resolve()),'source_trace_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'original_example_final_check':'NOT_APPLICABLE_CUSTOM_ASSET',
        'history_provenance':'Numeric rod dual at t is from accepted source call t-1. COM/pin/winding/Dahl memory uses accepted poses through t only. No source future contacts or private old-pose buffer is an inference input.',
        'teacher_target':'Native default10-iteration VBD labels for this exact asset, physical parameters and explicit contact profile; not a converged-teacher claim.'}
    if manifest.get('asset_definition'):metadata['asset_definition']=manifest['asset_definition']
    payload['metadata']=np.asarray(json.dumps(metadata))
    args.output.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(args.output,**payload)
    return {'status':'NATIVE_MIXED_CAUSAL_HISTORY_AND_ALL_REFERENCE_PATHS_QUALIFIED',
        'teacher_substeps':count,'source_numeric_fields_verified':source_count,
        'maximum_teacher_surface_travel_m':maximum_motion,'minimum_teacher_tet_determinant':minimum_tet_det,
        'complete_reference_material_and_VF_EE_paths_verified':True,
        'eligible_normal_course_length':metadata['eligible_normal_course_length'],
        'source_self_contact_definition':definition,'eligible_normal_course':metadata['eligible_normal_course'],
        'output':str(args.output),'output_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'full_solver_or_shape_collision_calls':calls,'new_optimizer_updates':0,'new_training_admissions':0,
        'scope':'Data format, exact source continuity, causal accepted history and every local material/VF/EE path. No model, shape oracle, training admission, prediction quality or realtime claim.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tape',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--motion-budget',type=float,default=.01)
    args=parser.parse_args()
    if args.report.exists() or args.output.exists():raise FileExistsError('Preserve data qualification artifacts')
    try:result=qualify(args);code=0
    except Exception as error:
        result={'status':'NATIVE_MIXED_REFERENCE_REFUSED','failure':str(error),'traceback':traceback.format_exc(),
                'new_optimizer_updates':0,'new_training_admissions':0};code=1
    result['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(code)


if __name__=='__main__':main()
