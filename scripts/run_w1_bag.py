"""Run the shared model on a simplified W1 bag-and-cube packing task."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset',type=Path,default=Path.home()/'下载/assets/w1-030/robot.urdf')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/mixed_course/last.pt')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frames',type=int,default=288)
    parser.add_argument('--substeps',type=int,default=10)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--friction',type=float,default=2.5)
    parser.add_argument('--bag-profile',choices=('stiff','flexible'),default='stiff')
    parser.add_argument('--bag-grip',choices=('horizontal','oblique'),default='horizontal')
    parser.add_argument('--soft-tangent-support-scale',type=float,default=64.)
    parser.add_argument('--elastic-shape-response',action='store_true')
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--resume-frame',type=int)
    parser.add_argument('--geometry-only',action='store_true')
    args=parser.parse_args()
    if args.frames<1 or args.substeps<1:parser.error('Positive frames and substeps required')
    args.output=args.output.expanduser().resolve();args.asset=args.asset.expanduser().resolve()
    if args.asset.is_dir():args.asset=args.asset/'robot.urdf'
    if (args.output/'run.json').exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True,exist_ok=True)
    from wp_vbd.newton_source import activate_newton_source,source_record
    activate_newton_source()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda('cpu' if args.geometry_only else args.device,ROOT)
    import numpy as np
    import torch
    import warp as wp
    import newton
    from scipy.spatial.transform import Rotation
    from wp_vbd.w1_bag_scene import build,controller,DURATION,TABLE_TOP,TABLE_BOUNDS,assess_recording,BAG_PROFILES,deformation_metrics
    from wp_vbd.w1_table_clearance import TableClearance
    from wp_vbd.gpu_budget import GpuBudget
    if not np.isfinite(args.soft_tangent_support_scale) or args.soft_tangent_support_scale<=0:
        parser.error('Positive finite soft tangential support required')
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu' if args.geometry_only else args.device)
    if not args.geometry_only and args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    options=dict(iterations=10,integrate_with_external_rigid_solver=False,particle_enable_self_contact=False,
                 rigid_joint_linear_ke=1e6,rigid_joint_angular_ke=1e6,rigid_joint_linear_kd=0.,rigid_joint_angular_kd=0.)
    signature=inspect.signature(newton.solvers.SolverVBD).bind(None,**options);signature.apply_defaults()
    configuration={k:v for k,v in signature.arguments.items() if k not in ('model','collision_pipeline')}
    calls={'vbd_construct':0,'vbd_step':0,'featherstone_construct':0,'featherstone_step':0}
    def forbidden(key):
        def fail(*args,**kwargs):calls[key]+=1;raise AssertionError('Forbidden native solver: '+key)
        return fail
    for cls,prefix in ((newton.solvers.SolverVBD,'vbd'),(newton.solvers.SolverFeatherstone,'featherstone')):
        cls.__init__=forbidden(prefix+'_construct');cls.step=forbidden(prefix+'_step')
    e=build(wp,newton,asset=args.asset,output=args.output,device='cpu' if args.geometry_only else args.device,
            substeps=args.substeps,friction=args.friction,bag_profile=args.bag_profile,bag_grip=args.bag_grip)
    audit=TableClearance(e.model,payload_id=e.payload_id,bounds=TABLE_BOUNDS,top=TABLE_TOP)
    positions=[];bodies=[];times=[];velocities=[];body_velocities=[];tools=[];targets=[];phases=[];clearances=[];forces=[]
    frame=0;step=0;nn_calls=0;failure=None;continuation=None;begin=time.perf_counter();minimum_area=1.;maximum_area=1.
    def record():
        positions.append(e.state_0.particle_q.numpy().copy());bodies.append(e.state_0.body_q.numpy().copy());times.append(frame*e.frame_dt)
        velocities.append(e.state_0.particle_qd.numpy().copy());body_velocities.append(e.state_0.body_qd.numpy().copy())
        tools.append(np.asarray([bodies[-1][i,:3]+Rotation.from_quat(bodies[-1][i,3:]).apply(offset) for i,offset in e.tools]))
        targets.append(e.control.joint_target_q.numpy().copy());clearances.append({'time':times[-1],**audit.measure(bodies[-1])})
    def save(name='trajectory.npz'):
        temporary=args.output/(name+'.tmp')
        with temporary.open('wb') as f:
            np.savez_compressed(f,particle_position=positions,body_pose=bodies,physical_time=times,particle_velocity=velocities,
                body_velocity=body_velocities,tool_position=tools,motor_target=targets,phase=phases,faces=e.bag_faces)
        temporary.replace(args.output/name)
    provenance=source_record();asset_sha=hashlib.sha256(args.asset.read_bytes()).hexdigest()
    if args.geometry_only:
        for frame,time_value in enumerate(np.linspace(0.,DURATION,97)):
            controller(e,float(time_value),wp);q=e.model.joint_q.numpy();q[:e.robot_coordinates]=e.control.joint_target_q.numpy()[:e.robot_coordinates]
            newton.eval_fk(e.model,wp.array(q,dtype=wp.float32,device='cpu'),e.model.joint_qd,e.state_0)
            record();times[-1]=float(time_value);clearances[-1]['time']=float(time_value)
        minimum=min(row['minimum_robot_visual_table_clearance_m'] for row in clearances if row['minimum_robot_visual_table_clearance_m'] is not None)
        save();(args.output/'table_visual_clearance.json').write_text(json.dumps(clearances,indent=2)+'\n')
        report=dict(mode='asset_geometry_preview',scene='research.w1_bag',asset=str(args.asset),asset_sha256=asset_sha,
                    friction=args.friction,bag_profile=args.bag_profile,bag_grip=args.bag_grip,full_authored_timeline_recorded=False,minimum_planned_visual_table_clearance_m=minimum,
                    status='PASS' if minimum>.002 else 'FAIL',newton_source=provenance)
        (args.output/'run.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    budget=GpuBudget(args.device,12.)
    stored=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=stored.get('model_config') or stored['config']['model'];network=MixedSolver(config).to(args.device).eval();network.load_state_dict(stored['model'])
    network.config['line_search']=[2.**(-i) for i in range(13)]
    scene=export_model(e.model,configuration,'research.w1_bag',newton,e.sim_dt,args.device)
    scene.metadata.update(detection_motion_budget=.003,bounded_motion_forecast_target_only=True,
        joint_tree_coarse_modes='scalar_tree_twelve_columns_v307',independent_object_motion_modes=True,joint_tree_extra_columns=8,
        separate_coarse_ray_step=True,soft_tangent_support_scale=args.soft_tangent_support_scale,rigid_tangent_support_scale=64.,
        force_aware_material_anticipation=True,compact_numerical_diagnostics=False,
        repair_native_mimic_roundoff=True,repair_metric_ray_roundoff=True,elastic_shape_response=args.elastic_shape_response)
    import wp_vbd.surface_joint_system as physics
    original_diagnostics=physics.diagnostics
    def diagnostics(scene,result):
        value=original_diagnostics(scene,result);owners=scene.candidates.soft.ports.owners[:,0]-scene.metadata['particle_count']
        load=result['soft_rows']['normal_load'][0]
        value['left_finger_normal_load_N']=[float(load[owners==i].sum()) for i in (16,17)]
        value['right_finger_normal_load_N']=[float(load[owners==i].sum()) for i in (7,8)]
        return value
    physics.diagnostics=diagnostics
    solver=SurfaceJointExampleSolver(network,scene,e.sim_dt,wp,newton,e.model,e.state_0)
    if len(scene.fixed_particle_ids) or e.model.attachment_particle_body_count:raise AssertionError('The paper bag must be unpinned and unattached')
    def count(*args):
        nonlocal nn_calls
        nn_calls+=1
    network.protected_response.hierarchy.register_forward_hook(count)
    checkpoint_sha=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    record();save('initial_state.npz')
    def areas(x):
        p=x[e.bag_faces].astype(np.float64);return np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=-1)
    reference=areas(positions[0]);prefix=None
    if args.resume:
        source=args.resume.expanduser().resolve();parent=json.loads((source/'run.json').read_text())
        for key,expected in [('checkpoint_sha256',checkpoint_sha),('asset_sha256',asset_sha),('newton_source',provenance),
                             ('physical_dt',e.sim_dt),('friction',args.friction),('mode','shared_network_inference'),('scene','research.w1_bag')]:
            if parent.get(key)!=expected:raise ValueError('Incompatible own-model record: '+key)
        if parent.get('bag_profile','stiff')!=args.bag_profile:raise ValueError('Paper material profile changed across continuation')
        if parent.get('bag_grip','horizontal')!=args.bag_grip:raise ValueError('Bag grip changed across continuation')
        if parent.get('bag_material',BAG_PROFILES['stiff'])!=BAG_PROFILES[args.bag_profile]:
            raise ValueError('Paper material coefficients changed across continuation')
        if any(parent['forbidden_solver_calls'].values()) or parent['teacher_trajectory_reads']:raise ValueError('Independent own-model record required')
        prefix=dict(np.load(source/'trajectory.npz'));restore=args.resume_frame
        if restore is None:restore=len(prefix['physical_time'])-2
        if restore<0 or len(prefix['physical_time'])<restore+2 or restore>=args.frames:raise ValueError('Saved frame plus matching next frame required')
        for i in range(restore+1):
            if i:
                controller(e,(i-1)*e.frame_dt,wp)
                if not np.array_equal(e.control.joint_target_q.numpy(),prefix['motor_target'][i]):raise ValueError('Changed motor target prefix')
            e.state_0.body_q.assign(prefix['body_pose'][i]);current=solver.read(e.state_0)
            angle=scene.joint_coordinates(current,solver.history['joint_angle_prev']).axis[...,3]
            solver.history['joint_angle_prev']=torch.where(scene.joint_spec.angular_mode==1,angle,0.).detach()
        for field,key in [('particle_q','particle_position'),('particle_qd','particle_velocity'),('body_q','body_pose'),('body_qd','body_velocity')]:getattr(e.state_0,field).assign(prefix[key][restore])
        for values,key in [(positions,'particle_position'),(bodies,'body_pose'),(times,'physical_time'),(velocities,'particle_velocity'),(body_velocities,'body_velocity'),(tools,'tool_position'),(targets,'motor_target')]:values[:]=list(prefix[key][:restore+1])
        phases[:]=list(prefix['phase'][:restore]);frame=restore;step=restore*args.substeps;solver.steps=step;solver.history['pin_initialized'].fill_(True)
        if (source/'grasp_forces.json').exists():
            forces[:]=[row for row in json.loads((source/'grasp_forces.json').read_text()) if row['time']<=times[-1]]
        clearances[:]=[{'time':float(t),**audit.measure(pose)} for t,pose in zip(times,bodies)]
        ratios=np.asarray([areas(x)/reference for x in positions]);minimum_area=float(ratios.min());maximum_area=float(ratios.max())
        continuation={'source':str(source),'reused_frames':restore,'trajectory_sha256':hashlib.sha256((source/'trajectory.npz').read_bytes()).hexdigest(),'first_continuation_frame_verified':False}
        previous_support=parent.get('soft_tangent_support_scale',64.)
        transition=(parent.get('grasp_continuation') or {}).get('soft_tangent_support_transition')
        if transition and restore<transition['after_verified_frame']:previous_support=transition['from']
        if previous_support!=args.soft_tangent_support_scale:
            scene.metadata['soft_tangent_support_scale']=previous_support
            continuation['soft_tangent_support_transition']={'from':previous_support,'to':args.soft_tangent_support_scale,
                'after_verified_frame':restore+1,'physical_material_and_coulomb_cap_unchanged':True}
        if parent.get('elastic_shape_response',False)!=args.elastic_shape_response:
            scene.metadata['elastic_shape_response']=parent.get('elastic_shape_response',False)
            continuation['elastic_shape_response_transition']={'from':parent.get('elastic_shape_response',False),
                'to':args.elastic_shape_response,'after_verified_frame':restore+1,'neural_weights_and_physical_material_unchanged':True}
    sources=[Path(__file__),ROOT/'src/wp_vbd/w1_bag_scene.py',ROOT/'src/wp_vbd/w1_dual_grasp_scene.py',
             ROOT/'src/wp_vbd/elastic_shape_modes.py',ROOT/'src/wp_vbd/articulation_coarse.py']
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    def report(status):
        r=dict(status=status,mode='shared_network_inference',scene='research.w1_bag',asset=str(args.asset),asset_sha256=asset_sha,
            checkpoint=str(args.checkpoint.resolve()),checkpoint_sha256=checkpoint_sha,checkpoint_step=stored['step'],newton_source=provenance,
            device=args.device,physics_device=str(e.model.device),frames_completed=frame,frames_requested=args.frames,physical_seconds=step*e.sim_dt,
            physical_dt=e.sim_dt,substeps=args.substeps,nn_backbone_calls=nn_calls,wall_seconds=time.perf_counter()-begin,
            current_phase=phases[-1] if phases else 'initial',particle_count=e.model.particle_count,triangle_count=e.model.tri_count,
            payload_body_id=e.payload_id,friction=args.friction,bag_profile=args.bag_profile,bag_grip=args.bag_grip,bag_material=BAG_PROFILES[args.bag_profile],
            current_bag_deformation=deformation_metrics(e.bag_rest,positions[-1]),
            soft_tangent_support_scale=args.soft_tangent_support_scale,
            active_soft_tangent_support_scale=scene.metadata['soft_tangent_support_scale'],
            elastic_shape_response=args.elastic_shape_response,active_elastic_shape_response=scene.metadata['elastic_shape_response'],
            elastic_shape_mode_diagnostic=getattr(scene,'elastic_shape_mode_diagnostic',{}),
            full_authored_timeline_recorded=frame==args.frames and frame*e.frame_dt>=DURATION-1e-9,
            minimum_recorded_triangle_area_ratio=minimum_area,maximum_recorded_triangle_area_ratio=maximum_area,
            minimum_robot_visual_table_clearance_m=min(row['minimum_robot_visual_table_clearance_m'] for row in clearances if row['minimum_robot_visual_table_clearance_m'] is not None),
            forbidden_solver_calls=calls,teacher_trajectory_reads=0,optimizer_updates=0,grasp_continuation=continuation,
            source_sha256=source_hashes,failure=failure,
            scope='Coarse open paper shell derived from referenced dimensions, upright initially, no handles, no particle self-contact. Robot motors only. Dynamic robot, bag and cube. No native dynamics or source-demo execution.')
        temporary=args.output/'run.tmp';temporary.write_text(json.dumps(r,indent=2)+'\n');temporary.replace(args.output/'run.json')
    report('RUNNING')
    try:
        with torch.no_grad():
            while frame<args.frames:
                phase=controller(e,frame*e.frame_dt,wp)
                for _ in range(args.substeps):
                    e.state_0.clear_forces();e.state_1.clear_forces();solver.step(e.state_0,e.state_1,e.control,None,e.sim_dt)
                    e.state_0,e.state_1=e.state_1,e.state_0;step+=1
                frame+=1;phases.append(phase);record()
                forces.append({'time':times[-1],'left_finger_normal_load_N':solver.last_diagnostic['left_finger_normal_load_N']})
                ratio=areas(positions[-1])/reference;minimum_area=min(minimum_area,float(ratio.min()));maximum_area=max(maximum_area,float(ratio.max()))
                if continuation is not None and frame==continuation['reused_frames']+1:
                    errors={key:float(np.max(np.abs(value-prefix[key][frame]))) for key,value in [('particle_position',positions[-1]),('body_pose',bodies[-1]),('particle_velocity',velocities[-1]),('body_velocity',body_velocities[-1])]}
                    if max(errors['particle_position'],errors['body_pose'])>2e-7 or max(errors['particle_velocity'],errors['body_velocity'])>1e-5:raise ValueError('Continuation mismatch: '+str(errors))
                    continuation.update(first_continuation_frame_verified=True,maximum_errors=errors);print(json.dumps({'continuation_verified':errors}),flush=True)
                    scene.metadata['soft_tangent_support_scale']=args.soft_tangent_support_scale
                    scene.metadata['elastic_shape_response']=args.elastic_shape_response
                budget.observe();report('RUNNING')
                if frame%12==0:
                    save('progress.npz')
                    temporary=args.output/'grasp_forces.tmp'
                    temporary.write_text(json.dumps(forces,indent=2)+'\n');temporary.replace(args.output/'grasp_forces.json')
                if frame%6==0 or frame==1:
                    print(json.dumps({'frame':frame,'time':times[-1],'phase':phase,'wall':time.perf_counter()-begin,'minimum_area':minimum_area,
                        'cube_lift':float(bodies[-1][e.payload_id,2]-bodies[0][e.payload_id,2]),'left_contact':forces[-1]['left_finger_normal_load_N']}),flush=True)
    except BaseException:
        failure=traceback.format_exc();print(failure,flush=True)
        if solver.last_diagnostic:(args.output/'failure_diagnostic.json').write_text(json.dumps(solver.last_diagnostic,default=lambda x:x.detach().cpu().tolist() if isinstance(x,torch.Tensor) else str(x),indent=2)+'\n')
    finally:
        physics.diagnostics=original_diagnostics;save();report('FAILED' if failure else 'COMPLETE')
        (args.output/'table_visual_clearance.json').write_text(json.dumps(clearances,indent=2)+'\n');(args.output/'grasp_forces.json').write_text(json.dumps(forces,indent=2)+'\n')
        quality=assess_recording(args.output,json.loads((args.output/'run.json').read_text()))
        minimum=min(row['minimum_robot_visual_table_clearance_m'] for row in clearances if row['minimum_robot_visual_table_clearance_m'] is not None)
        quality['minimum_robot_visual_table_clearance_m']=minimum;quality['accepted']&=minimum>=-.001
        (args.output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
    if failure:sys.exit(1)
    print(args.output/'quality.json',flush=True)


if __name__=='__main__':main()
