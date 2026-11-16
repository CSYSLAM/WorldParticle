"""Native experimental general-joint inference with the shared response NN."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--demo',default='basic.example_basic_joints');p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/shared_joints_v26/shared_strict.pt');p.add_argument('--device',default='cpu');p.add_argument('--headless',action='store_true');p.add_argument('--frames',type=int,default=120);p.add_argument('--test-final',action='store_true');p.add_argument('--report',type=Path,required=True);p.add_argument('--screenshot',type=Path);p.add_argument('--disable-learning',action='store_true')
    p.add_argument('--component-batch',action='store_true',help='Experimental shared component operator override; does not change NN weights or training coverage')
    for name in ('joint-linear-ke','joint-angular-ke','joint-linear-kd','joint-angular-kd','friction','pitch'):p.add_argument('--'+name,type=float)
    args=p.parse_args()
    implementation={path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'src/wp_vbd').glob('*.py')}
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.general_joint_scene import export_model
    from wp_vbd.general_joint_runtime import JointExampleSolver
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(args.device)
    raw=args.checkpoint.read_bytes();stored=torch.load(io.BytesIO(raw),weights_only=True,map_location=args.device)
    model_configuration=copy.deepcopy(stored['config']['model'])
    if args.component_batch:model_configuration['rigid_component_batch']=True
    model=MixedSolver(model_configuration).to(args.device).eval();model.load_state_dict(stored['model'])
    viewer=newton.viewer.ViewerNull(num_frames=args.frames) if args.headless else newton.viewer.ViewerGL(width=1600,height=900,vsync=False)
    if not args.headless:
        viewer.renderer.set_title('WorldParticle | shared joint inference | '+args.demo)
        viewer.gui.ui.io.set_ini_filename(str(ROOT/'.local/general_joint_v26/viewer.ini'))
    arguments=('--pitch',str(args.pitch)) if args.pitch is not None else ()
    example,configurations,_=make_example(args.demo,reference=False,viewer=viewer,arguments=arguments,host_scalar_compat=args.demo=='basic.example_basic_conveyor',preserve_defaults=args.demo=='basic.example_basic_urdf')
    configuration=configurations[0][1]
    for name in ('joint_linear_ke','joint_angular_ke','joint_linear_kd','joint_angular_kd'):
        value=getattr(args,name)
        if value is not None:
            if not np.isfinite(value) or value<0:p.error('Joint material coefficients must be finite and nonnegative')
            configuration['rigid_'+name]=value
    if args.friction is not None:
        if not np.isfinite(args.friction) or args.friction<0:p.error('Friction must be finite and nonnegative')
        values=example.model.shape_material_mu.numpy();values[:]=args.friction;example.model.shape_material_mu.assign(values)
    def forbidden(*unused,**kwargs):raise AssertionError('Full SolverVBD forbidden in joint inference')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden
    native_observables=getattr(example,'solver_observables',None)
    scene=export_model(example.model,configuration,args.demo,newton,example.sim_dt,args.device,native_contact_export=native_observables is not None,prescribed_motion=bool(np.any(example.model.body_flags.numpy()==int(newton.BodyFlags.KINEMATIC))),packed_worlds=len(example.model.gravity.numpy().reshape(-1,3))>1)
    solver=JointExampleSolver(model,scene,scene.initial_state(example.model,wp),example.sim_dt,wp,newton,example.model,learned=not args.disable_learning);example.solver=solver
    if native_observables is not None:example.solver_observables=solver.observables(contacts=example.contacts,kinds=native_observables.kinds)
    stream=wp.stream_from_torch(torch.cuda.current_stream()) if args.device.startswith('cuda') else None
    def scope():return wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice('cpu')
    begin=time.perf_counter();frames=0;failure=None;final='NOT_RUN';trajectory=[]
    def record(state):
        trajectory.append((solver.steps,state.body_q.numpy().copy(),solver.history['joint_angle_prev'].cpu().numpy().copy()))
        if not args.headless and viewer.is_running():
            previous=example.state_0;clock=example.sim_time;example.state_0=state;example.sim_time=solver.steps*example.sim_dt
            try:example.render()
            finally:example.state_0=previous;example.sim_time=clock
    record(example.state_0);solver.on_substep=record
    try:
        while viewer.is_running() and frames<args.frames:
            try:
                with scope():
                    if viewer.should_step():
                        example.step();frames+=1
                        if (args.test_final or args.demo=='basic.example_basic_conveyor_forces') and hasattr(example,'test_post_step'):example.test_post_step()
                    example.render()
                if frames%30==0:print(f'frames={frames} steps={solver.steps} t={example.sim_time:.3f}',flush=True)
            except (RuntimeError,ValueError) as error:failure=str(error);break
        if args.test_final and failure is None:
            try:example.test_final();final='PASS'
            except Exception as error:final='FAIL: '+str(error)
        if args.screenshot and not args.headless:
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(args.screenshot)
    finally:viewer.close()
    result={'status':'INFERENCE_EXECUTED' if failure is None else 'MODEL_FAILED','demo':args.demo,'frames':frames,'requested_frames':args.frames,'physical_substeps':solver.steps,'physical_seconds':solver.steps*example.sim_dt,'seconds':time.perf_counter()-begin,'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),'failure':failure,'original_test_final':final,'full_vbd_prediction_calls':0,'teacher_trajectory_reads':0,'learning_enabled':not args.disable_learning,'new_training_updates':0,'single_axis_history':solver.history['joint_angle_prev'].cpu().tolist(),'collision_queries':solver.oracle.queries,'source_controls_and_render_preserved':True,'implementation_sha256':{path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'src/wp_vbd').glob('*.py')},'scope':'Untrained joint-contract transfer unless separately documented. Finite objective joint law differs from the source linear projection approximation; no all-demo or realtime claim.'}
    label='urdf_train_default' if args.demo=='basic.example_basic_urdf' else 'conveyor_train_default' if args.demo=='basic.example_basic_conveyor' else 'conveyor_forces_train_default' if args.demo=='basic.example_basic_conveyor_forces' else 'newton_cradle_train_default' if args.demo=='contacts.example_newton_cradle' else 'mimic_default.npz_train_default' if scene.mimic_spec is not None else 'basic_joints_train_default'
    result['current_trial_joint_training_samples']=stored.get('sampled_updates',{}).get(label,0)
    result['active_weight_joint_training_samples']=stored.get('active_weight_optimizer_samples',stored.get('sampled_updates',{})).get(label,0)
    result['joint_training_status']='SAMPLED_IN_ACTIVE_WEIGHT_ANCESTRY' if result['active_weight_joint_training_samples'] else 'UNTRAINED_JOINT_TRANSFER'
    result.pop('new_training_updates');result['optimizer_updates_during_inference']=0;result['checkpoint_selected_trial_updates']=stored.get('step',0) if stored.get('new_optimizer_updates')!=0 else 0
    result['applied_parameters']={name:configuration['rigid_'+name] for name in ('joint_linear_ke','joint_angular_ke','joint_linear_kd','joint_angular_kd')};result['applied_parameters']['friction_override']=args.friction
    result['scope']='One shared checkpoint, objective finite joint/contact law and original native asset/control/render. Current training samples and actual native quality are separate; no all-demo or realtime claim.'
    result['implementation_sha256']=implementation;result['implementation_snapshot_timing']='Before asset construction; captured once at startup'
    result['runtime_model_config_overrides']={'rigid_component_batch':True} if args.component_batch else {}
    if scene.metadata.get('world_gravity_contract'):
        result['world_gravity_contract']=scene.metadata['world_gravity_contract'];result['packed_world_count']=scene.metadata['packed_world_count'];result['native_body_count']=example.model.body_count
        if args.demo=='basic.example_basic_urdf':
            roots=np.arange(example.world_count)*(example.model.body_count//example.world_count)
            result['urdf_task']={'root_heights_m':example.state_0.body_q.numpy()[roots,2].tolist(),'maximum_absolute_velocity_component':float(np.abs(example.state_0.body_qd.numpy()).max())}
    if scene.metadata.get('prescribed_pose_history'):
        result['prescribed_pose_history_contract']=scene.metadata['prescribed_pose_history'];result['complete_history_keys']=sorted(solver.history)
        result['prescribed_task']={'belt_angular_speed_rad_per_s':example.belt_angular_speed,'physical_belt_angle_rad':float(solver.history['joint_angle_prev'][0,0]),'bag_radii_m':np.linalg.norm(example.state_0.body_q.numpy()[example.bag_bodies,:2],axis=-1).tolist()}
    if native_observables is not None:
        result['native_contact_export_contract']=scene.metadata['native_contact_export'];result['contact_exports']=solver.contact_exports
        result['conveyor_task']={'minimum_maximum_travel_m':float(np.min(example.max_travel)),'maximum_travel_by_body_m':example.max_travel.tolist(),'mean_final_speed_m_per_s':float(np.linalg.norm(example.state_0.body_qd.numpy()[example.tracked_bodies,:3],axis=1).mean()),'final_contact_force_norm':float(np.linalg.norm(example.solver_observables.contact_f.numpy()))}
    if scene.mimic_spec is not None:
        result['mimic_training_status']='SAMPLED_IN_ACTIVE_WEIGHT_ANCESTRY' if result['active_weight_joint_training_samples'] else 'UNTRAINED_MIMIC_TRANSFER'
        result['ideal_mimic_contract']=scene.metadata['ideal_mimic_contract']
        result['mimic_task']={name:getattr(example,name) for name in ('pitch','coupling_ratio','max_abs_screw_angle','max_abs_coupling_error','screw_angle','nut_travel')}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.report.with_suffix('.npz'),physical_substep=np.asarray([r[0] for r in trajectory]),body_pose=np.stack([r[1] for r in trajectory]),joint_angle_prev=np.stack([r[2] for r in trajectory]))
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({key:result[key] for key in ('status','frames','physical_substeps','failure','original_test_final','seconds')},indent=2),flush=True)
    if failure is not None:sys.exit(1)


if __name__=='__main__':main()
