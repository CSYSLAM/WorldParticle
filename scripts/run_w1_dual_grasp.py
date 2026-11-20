"""Shared-network W1 demo: left rigid grasp and right soft grasp."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--asset',type=Path,default=Path.home()/'下载/assets/w1-030/robot.urdf')
    p.add_argument('--checkpoint',type=Path,default=ROOT/'model_releases/shared_vbd_v24_cloth40/model.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int)
    p.add_argument('--objects',choices=('blocks','cylinder_ball'),default='cylinder_ball')
    p.add_argument('--style',choices=('overhead','front','side'),default='side')
    p.add_argument('--friction',type=float,default=2.5)
    p.add_argument('--soft-lambda',type=float,help='Lame bulk parameter; front defaults to nearly incompressible rubber')
    p.add_argument('--rigid-tangent-support-scale',type=float,help='Numerical friction support; side defaults to 64, other styles to 1')
    p.add_argument('--release',action='store_true',help='Gradually relax grip at 1.4 s, then allow a natural fall')
    p.add_argument('--resume-grasp',type=Path,help='Continue an independently recorded own-model grasp')
    p.add_argument('--resume-frame',type=int,default=84,help='Saved own-model frame to restore; verifies the following frame')
    p.add_argument('--substeps',type=int,default=3)
    p.add_argument('--motion-budget',type=float,default=.012)
    p.add_argument('--surface-tangent-history',action=argparse.BooleanOptionalAction,default=True)
    p.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0')
    p.add_argument('--window',action='store_true')
    a=p.parse_args()
    if a.frames is None:a.frames=180 if a.style=='side' else 168 if a.style in ('front','side') else 144 if a.release else 84
    if a.style in ('front','side') and a.release:p.error('Front grasp holds the payloads')
    if a.resume_frame<0:p.error('Nonnegative resume frame required')
    if a.resume_grasp and a.surface_tangent_history:
        p.error('Frame-only prefixes do not contain the accepted contact cache. Run a fresh current-mode trajectory, or use --no-surface-tangent-history for both matching legacy runs.')
    a.asset=a.asset.expanduser().resolve();a.checkpoint=a.checkpoint.expanduser().resolve();a.output=a.output.expanduser().resolve()
    if a.asset.is_dir():a.asset=a.asset/'robot.urdf'
    if a.frames<1 or a.substeps<1:p.error('Positive frames and substeps required')
    if (a.output/'run.json').exists():raise FileExistsError(a.output/'run.json')
    a.output.mkdir(parents=True,exist_ok=True)
    from wp_vbd.newton_source import activate_newton_source,source_record
    activate_newton_source()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd.gpu_budget import GpuBudget
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd.w1_dual_grasp_scene import build,controller,DURATION,LIFT,OPENING,assess_recording,scene_settings
    layout=scene_settings(a.style)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    budget=GpuBudget(a.device,12.)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(a.device)
    if a.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    options=dict(iterations=10,integrate_with_external_rigid_solver=False,particle_enable_self_contact=False,
                 rigid_joint_linear_ke=1e6 if a.style in ('front','side') else 1e5,
                 rigid_joint_angular_ke=1e6 if a.style in ('front','side') else 1e5,rigid_joint_linear_kd=0.,rigid_joint_angular_kd=0.)
    bound=inspect.signature(newton.solvers.SolverVBD).bind(None,**options);bound.apply_defaults()
    configuration={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    calls={'vbd_construct':0,'vbd_step':0,'featherstone_construct':0,'featherstone_step':0}
    def forbidden(key):
        def fail(*args,**kwargs):calls[key]+=1;raise AssertionError('Complete solver forbidden: '+key)
        return fail
    for cls,prefix in ((newton.solvers.SolverVBD,'vbd'),(newton.solvers.SolverFeatherstone,'featherstone')):
        cls.__init__=forbidden(prefix+'_construct');cls.step=forbidden(prefix+'_step')
    e=build(wp,newton,asset=a.asset,output=a.output,device=a.device,substeps=a.substeps,
            objects=a.objects,friction=a.friction,release=a.release,style=a.style,soft_lambda=a.soft_lambda)
    stored=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=dict(stored.get('model_config') or stored['config']['model'])
    config.update(mimic_metric_tangent=True,periodic_bending=True)
    network=MixedSolver(config).to(a.device).eval();network.load_state_dict(stored['model'])
    if a.style in ('front','side'):network.config['line_search']=[2.**(-i) for i in range(13)]
    scene=export_model(e.model,configuration,'research.w1_dual_grasp',newton,e.sim_dt,a.device)
    scene.metadata.update(detection_motion_budget=a.motion_budget,bounded_motion_forecast_target_only=True,
        joint_tree_coarse_modes='scalar_tree_twelve_columns_v307',independent_object_motion_modes=True,joint_tree_extra_columns=8,
        separate_coarse_ray_step=a.style in ('front','side'),soft_tangent_support_scale=64. if a.style in ('front','side') else 8.,force_aware_material_anticipation=True,
        compact_numerical_diagnostics=True,elastic_shape_response=True,
        surface_tangent_memory=a.surface_tangent_history,surface_tangent_history_capacity=8192)
    scene.metadata['repair_native_mimic_roundoff']=a.style in ('front','side')
    scene.metadata['repair_metric_ray_roundoff']=a.style in ('front','side')
    requested_rigid_support=a.rigid_tangent_support_scale if a.rigid_tangent_support_scale is not None else 64. if a.style=='side' else 1.
    if not np.isfinite(requested_rigid_support) or requested_rigid_support<=0:raise ValueError('Positive finite rigid tangential support required')
    scene.metadata['rigid_tangent_support_scale']=requested_rigid_support
    solver=SurfaceJointExampleSolver(network,scene,e.sim_dt,wp,newton,e.model,e.state_0)
    from wp_vbd.adaptive_step import AdaptiveStepConfig
    solver.enable_adaptive_steps(AdaptiveStepConfig(maximum_dt=e.sim_dt,minimum_dt=min(e.sim_dt,1/600),
        maximum_attempts=256,maximum_substeps=128))
    nn_calls=0;ray_roundoff_repairs=0
    def count(*args):
        nonlocal nn_calls
        nn_calls+=1
    network.protected_response.hierarchy.register_forward_hook(count)
    viewer=None
    if a.window:
        import newton.viewer
        viewer=newton.viewer.ViewerGL(width=1440,height=900,vsync=False)
        viewer.set_model(e.model);viewer.set_camera(wp.vec3(2.,-1.5,1.6));viewer.camera.look_at(np.array([.1,0.,.9]))
        if a.style in ('front','side'):viewer.set_camera(wp.vec3(2.2,-1.6,1.9));viewer.camera.look_at(np.array([.15,0.,1.05]))
    particles=[];bodies=[];times=[];phases=[];particle_velocity=[];body_velocity=[];tool_positions=[];motor_targets=[]
    table_measurements=[];table_audit=None
    if a.style=='side':
        from wp_vbd.w1_table_clearance import TableClearance
        table_audit=TableClearance(e.model,payload_id=e.payload_id)
    frames=0;steps=0;failure=None;begin=time.perf_counter()
    def record():
        particles.append(e.state_0.particle_q.numpy().copy());bodies.append(e.state_0.body_q.numpy().copy());times.append(solver.physical_time)
        particle_velocity.append(e.state_0.particle_qd.numpy().copy());body_velocity.append(e.state_0.body_qd.numpy().copy())
        from scipy.spatial.transform import Rotation
        tool_positions.append(np.asarray([bodies[-1][i,:3]+Rotation.from_quat(bodies[-1][i,3:]).apply(offset)
                                          for i,offset in e.tools]))
        motor_targets.append(e.control.joint_target_q.numpy().copy())
        if table_audit is not None:table_measurements.append({'time':times[-1],**table_audit.measure(bodies[-1])})
    record()
    np.savez_compressed(a.output/'initial_state.npz',particle_position=np.asarray(particles),body_pose=np.asarray(bodies),physical_time=np.asarray(times))
    tets=e.model.tet_indices.numpy()
    def determinant(x):
        p=x[tets].astype(np.float64);return np.linalg.det(p[:,1:]-p[:,:1])
    reference=determinant(particles[0]);min_ratio=1.;max_ratio=1.
    checkpoint_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    provenance=source_record()
    branch=None;source_tape=None
    if a.resume_grasp:
        resume_frame=a.resume_frame
        if a.release or a.frames<=resume_frame:raise ValueError('Continuation must extend beyond the restored frame')
        source=a.resume_grasp.expanduser().resolve();parent=json.loads((source/'run.json').read_text())
        if parent.get('style','overhead')!=a.style:raise ValueError('Grasp style changed across continuation')
        expected={'mode':'shared_network_inference','objects':a.objects,'friction':a.friction,
                  'checkpoint_sha256':checkpoint_sha,'asset_sha256':hashlib.sha256(a.asset.read_bytes()).hexdigest(),
                  'physical_dt':e.sim_dt,'device':a.device,'newton_source':provenance,
                  'teacher_trajectory_reads':0,'optimizer_updates':0}
        if parent.get('soft_lame_lambda',5000.)!=e.soft_lambda:raise ValueError('Soft material changed across continuation')
        if parent.get('separate_coarse_ray_step',False)!=scene.metadata['separate_coarse_ray_step']:
            raise ValueError('Subspace step policy changed across continuation')
        for key,value in expected.items():
            if parent.get(key)!=value:raise ValueError('Incompatible grasp prefix: '+key)
        if any(parent['forbidden_solver_calls'].values()) or len(scene.fixed_particle_ids):
            raise ValueError('Continuation requires independent inference and no prescribed particles')
        source_tape=dict(np.load(source/'trajectory.npz'))
        if len(source_tape['physical_time'])<resume_frame+2 or abs(float(source_tape['physical_time'][resume_frame])-resume_frame*e.frame_dt)>1e-9:
            raise ValueError('Complete grasp and a following frame are required for replay verification')
        for i in range(resume_frame+1):
            if i:
                controller(e,(i-1)*e.frame_dt,wp)
                if not np.array_equal(e.control.joint_target_q.numpy(),source_tape['motor_target'][i]):
                    raise ValueError('Saved prefix does not match the holding controller')
            e.state_0.body_q.assign(source_tape['body_pose'][i])
            current=solver.read(e.state_0)
            angle=scene.joint_coordinates(current,solver.history['joint_angle_prev']).axis[...,3]
            solver.history['joint_angle_prev']=torch.where(scene.joint_spec.angular_mode==1,angle,0.).detach()
        for field,key in (('particle_q','particle_position'),('particle_qd','particle_velocity'),
                          ('body_q','body_pose'),('body_qd','body_velocity')):
            getattr(e.state_0,field).assign(source_tape[key][resume_frame])
        solver.history['pin_initialized'].fill_(True);solver.validate_start(solver.read(e.state_0))
        targets=[particles,bodies,times,particle_velocity,body_velocity,tool_positions,motor_targets]
        keys=['particle_position','body_pose','physical_time','particle_velocity','body_velocity','tool_position','motor_target']
        for target,key in zip(targets,keys):target[:]=list(source_tape[key][:resume_frame+1])
        if table_audit is not None:
            table_measurements[:]=[{'time':float(t),**table_audit.measure(pose)} for t,pose in zip(times,bodies)]
        phases[:]=list(source_tape['phase'][:resume_frame]);frames=resume_frame;steps=resume_frame*a.substeps;solver.steps=steps
        prefix_ratios=np.asarray([determinant(x)/reference for x in particles])
        min_ratio=float(prefix_ratios.min());max_ratio=float(prefix_ratios.max())
        branch={'source':str(source),'source_trajectory_sha256':hashlib.sha256((source/'trajectory.npz').read_bytes()).hexdigest(),
                'reused_frames':resume_frame,'history':'empty pin history; joint winding restored from own saved poses',
                'first_continuation_frame_verified':False}
        previous_support=parent.get('rigid_tangent_support_scale',1.)
        previous_transition=parent.get('grasp_continuation') or {}
        previous_transition=previous_transition.get('numerical_friction_support_transition')
        if previous_transition and resume_frame<previous_transition['after_verified_frame']:
            previous_support=previous_transition['from']
        if previous_support!=requested_rigid_support:
            scene.metadata['rigid_tangent_support_scale']=previous_support
            branch['numerical_friction_support_transition']={'from':previous_support,'to':requested_rigid_support,
                'after_verified_frame':resume_frame+1,'physical_friction_coefficient_unchanged':True}
    def report(status):
        r={'status':status,'mode':'shared_network_inference','scene':'research.w1_dual_grasp','asset':str(a.asset.resolve()),
          'asset_sha256':hashlib.sha256(a.asset.read_bytes()).hexdigest(),'checkpoint':str(a.checkpoint.resolve()),
          'checkpoint_sha256':checkpoint_sha,'checkpoint_step':stored['step'],'newton_source':provenance,
          'device':a.device,'physics_device':str(e.model.device),'physical_dt':e.sim_dt,'substeps':a.substeps,
          'physical_seconds':steps*e.sim_dt,'frames_completed':frames,'frames_requested':a.frames,'nn_backbone_calls':nn_calls,
          'physical_substeps':solver.steps,'adaptive_retries':solver.adaptive_retries,
          'adaptive_min_accepted_dt_s':solver.adaptive_min_accepted_dt,
          'analytic_predictor_rounds':config.get('analytic_predictor_rounds',0),'model_unroll':config.get('unroll',1),
          'detection_motion_budget_m':a.motion_budget,'surface_tangent_history':a.surface_tangent_history,
          'current_phase':phases[-1] if phases else 'initial',
          'grasp_continuation':branch,
          'forbidden_solver_calls':calls,'teacher_trajectory_reads':0,'optimizer_updates':0,
          'robot_body_count':e.model.body_count-1,'particle_count':e.model.particle_count,'tet_count':e.model.tet_count,
          'payload_body_id':e.payload_id,'rigid_lift_m':float(bodies[-1][e.payload_id,2]-bodies[0][e.payload_id,2]),
          'soft_center_lift_m':float((particles[-1]-particles[0]).mean(0)[2]),
          'tool_lift_m':(tool_positions[-1][:,2]-tool_positions[0][:,2]).tolist(),
          'minimum_recorded_tet_ratio':min_ratio,'maximum_recorded_tet_ratio':max_ratio,
          'motion':f'Close both grippers, command {LIFT*100:g} cm lift, hold.'+(' Relax fingers at 1.4 s and settle.' if a.release else ''),
          'style':a.style,'table_top_m':layout['table_top'],'support_top_m':layout['support_top'],
          'joint_constraint_stiffness':options['rigid_joint_linear_ke'],
          'line_search':network.config.get('line_search'),
          'native_roundoff_projection_evaluations':solver.native_roundoff_projection_count,
          'maximum_native_position_correction_m':solver.maximum_native_position_correction,
          'soft_tangent_support_scale':scene.metadata['soft_tangent_support_scale'],
          'rigid_tangent_support_scale':requested_rigid_support,
          'active_rigid_tangent_support_scale':scene.metadata['rigid_tangent_support_scale'],
          'separate_coarse_ray_step':scene.metadata['separate_coarse_ray_step'],
          'repair_metric_ray_roundoff':scene.metadata['repair_metric_ray_roundoff'],
          'metric_ray_roundoff_repairs':ray_roundoff_repairs,
          'objects':a.objects,'friction':a.friction,'release':a.release,
          'soft_lame_mu':5000.,'soft_lame_lambda':e.soft_lambda,
          'initial_finger_opening_m':OPENING,'requested_tree_columns_per_interaction':8,
          'scope':'Original W1 visuals and inertias; fixed torso/head; local-box collision proxies. Dynamic arms/fingers and both payloads. Joint targets only; no payload attachment or pose script.',
          'full_authored_timeline_recorded':frames==a.frames and steps*e.sim_dt>=(2.4 if a.release else layout['duration'])-1e-9,
          'wall_seconds':time.perf_counter()-begin,'failure':failure}
        if a.style in ('front','side'):r.update(motion='Reach horizontally, close, lift 9 cm, draw back 10 cm, hold.',initial_finger_opening_m=.028,
                                    finger_targets_m={'capture':.024,'hold':.0265,'transition_seconds':[.8,1.2]})
        if a.style=='side':r.update(initial_finger_opening_m=.028,
            finger_targets_m={'capture':.022,'hold':.022},support_blocks=False,gripper_closing_axis='horizontal lateral',
            table_bounds_xy=[.445,.985,-.6,.6],motion='Reach horizontally 4 cm, side pinch, lift 9 cm, draw back 10 cm, hold.',
            minimum_robot_visual_table_clearance_m=min(r['minimum_robot_visual_table_clearance_m'] for r in table_measurements
                                                       if r['minimum_robot_visual_table_clearance_m'] is not None))
        tmp=a.output/'run.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(a.output/'run.json')
    report('RUNNING')
    try:
        with torch.no_grad():
            for frame in range(frames,a.frames):
                phase=controller(e,frame*e.frame_dt,wp)
                for substep in range(a.substeps):
                    e.state_0.clear_forces();e.state_1.clear_forces()
                    solver.step(e.state_0,e.state_1,e.control,None,e.sim_dt)
                    e.state_0,e.state_1=e.state_1,e.state_0;steps+=1
                    diagnostic=solver.last_diagnostic
                    ray_rows=list(diagnostic.get('contact_metric_ray_scales',[]))
                    for split in diagnostic.get('separate_subspace_ray_scales',[]):ray_rows.extend(split.values())
                    ray_roundoff_repairs+=sum(sum(row.get('roundoff_null_repaired',[])) for rows in ray_rows for row in rows)
                frames+=1;phases.append(phase);record()
                if branch is not None and frames==a.resume_frame+1:
                    delta={key:float(np.max(np.abs(value-source_tape[key][frames]))) for key,value in
                           (('particle_position',particles[-1]),('body_pose',bodies[-1]),
                            ('particle_velocity',particle_velocity[-1]),('body_velocity',body_velocity[-1]))}
                    if max(delta['particle_position'],delta['body_pose'])>2e-7 or max(delta['particle_velocity'],delta['body_velocity'])>1e-5:
                        raise ValueError('Continuation differs from the matching recorded frame: '+str(delta))
                    branch.update(first_continuation_frame_verified=True,maximum_errors=delta)
                    print(json.dumps({'continuation_verified':delta}),flush=True)
                    scene.metadata['rigid_tangent_support_scale']=requested_rigid_support
                if frames%12==0:
                    with (a.output/'progress.tmp').open('wb') as tape:
                        np.savez_compressed(tape,particle_position=np.asarray(particles),body_pose=np.asarray(bodies),physical_time=np.asarray(times))
                    (a.output/'progress.tmp').replace(a.output/'progress.npz')
                ratio=determinant(particles[-1])/reference;min_ratio=min(min_ratio,float(ratio.min()));max_ratio=max(max_ratio,float(ratio.max()))
                budget.observe();report('RUNNING')
                if frames%6==0 or frames==1:print(json.dumps({'frame':frames,'seconds':steps*e.sim_dt,'phase':phase,'wall':time.perf_counter()-begin}),flush=True)
                if viewer:
                    viewer.begin_frame(steps*e.sim_dt);viewer.log_state(e.state_0);viewer.end_frame()
                    if not viewer.is_running():break
    except Exception:
        failure=traceback.format_exc();print(failure,flush=True)
        if solver.last_diagnostic:
            (a.output/'failure_diagnostic.json').write_text(json.dumps(solver.last_diagnostic,
                default=lambda v:v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else str(v),indent=2)+'\n')
    finally:
        np.savez_compressed(a.output/'trajectory.npz',particle_position=np.asarray(particles),body_pose=np.asarray(bodies),
            physical_time=np.asarray(times),phase=np.asarray(phases),particle_velocity=np.asarray(particle_velocity),
            body_velocity=np.asarray(body_velocity),tool_position=np.asarray(tool_positions),motor_target=np.asarray(motor_targets),
            tets=tets,faces=e.model.tri_indices.numpy())
        report('FAILED' if failure else 'COMPLETE' if frames==a.frames else 'INTERRUPTED')
        quality=assess_recording(a.output,json.loads((a.output/'run.json').read_text()))
        if table_audit is not None:
            minimum=min(r['minimum_robot_visual_table_clearance_m'] for r in table_measurements
                        if r['minimum_robot_visual_table_clearance_m'] is not None)
            quality.update(minimum_robot_visual_table_clearance_m=minimum,no_large_robot_table_penetration=minimum>=-.001)
            quality['accepted']&=quality['no_large_robot_table_penetration']
            (a.output/'table_visual_clearance.json').write_text(json.dumps(table_measurements,indent=2)+'\n')
        (a.output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
        if viewer:viewer.close()
    if failure:sys.exit(1)
    print(a.output/'run.json',flush=True)


if __name__=='__main__':main()
