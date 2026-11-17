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
    p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/mixed_course/last.pt')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--frames',type=int,default=84)
    p.add_argument('--substeps',type=int,default=10)
    p.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0')
    p.add_argument('--window',action='store_true')
    a=p.parse_args()
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
    from wp_vbd.w1_dual_grasp_scene import build,controller,DURATION,LIFT,OPENING,assess_recording
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    budget=GpuBudget(a.device,12.)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(a.device)
    if a.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    options=dict(iterations=10,integrate_with_external_rigid_solver=False,particle_enable_self_contact=False,
                 rigid_joint_linear_ke=1e5,rigid_joint_angular_ke=1e5,rigid_joint_linear_kd=0.,rigid_joint_angular_kd=0.)
    bound=inspect.signature(newton.solvers.SolverVBD).bind(None,**options);bound.apply_defaults()
    configuration={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    calls={'vbd_construct':0,'vbd_step':0,'featherstone_construct':0,'featherstone_step':0}
    def forbidden(key):
        def fail(*args,**kwargs):calls[key]+=1;raise AssertionError('Complete solver forbidden: '+key)
        return fail
    for cls,prefix in ((newton.solvers.SolverVBD,'vbd'),(newton.solvers.SolverFeatherstone,'featherstone')):
        cls.__init__=forbidden(prefix+'_construct');cls.step=forbidden(prefix+'_step')
    e=build(wp,newton,asset=a.asset,output=a.output,device=a.device,substeps=a.substeps)
    stored=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=stored.get('model_config') or stored['config']['model']
    network=MixedSolver(config).to(a.device).eval();network.load_state_dict(stored['model'])
    scene=export_model(e.model,configuration,'research.w1_dual_grasp',newton,e.sim_dt,a.device)
    scene.metadata.update(detection_motion_budget=.003,bounded_motion_forecast_target_only=True,
        joint_tree_coarse_modes='scalar_tree_twelve_columns_v307',independent_object_motion_modes=True,joint_tree_extra_columns=8,
        separate_coarse_ray_step=False,soft_tangent_support_scale=8.,force_aware_material_anticipation=True,
        compact_numerical_diagnostics=True)
    solver=SurfaceJointExampleSolver(network,scene,e.sim_dt,wp,newton,e.model,e.state_0)
    nn_calls=0
    def count(*args):
        nonlocal nn_calls
        nn_calls+=1
    network.protected_response.hierarchy.register_forward_hook(count)
    viewer=None
    if a.window:
        import newton.viewer
        viewer=newton.viewer.ViewerGL(width=1440,height=900,vsync=False)
        viewer.set_model(e.model);viewer.set_camera(wp.vec3(2.,-1.5,1.6));viewer.camera.look_at(np.array([.1,0.,.9]))
    particles=[];bodies=[];times=[];phases=[];particle_velocity=[];body_velocity=[];tool_positions=[];motor_targets=[]
    frames=0;steps=0;failure=None;begin=time.perf_counter()
    def record():
        particles.append(e.state_0.particle_q.numpy().copy());bodies.append(e.state_0.body_q.numpy().copy());times.append(steps*e.sim_dt)
        particle_velocity.append(e.state_0.particle_qd.numpy().copy());body_velocity.append(e.state_0.body_qd.numpy().copy())
        from scipy.spatial.transform import Rotation
        tool_positions.append(np.asarray([bodies[-1][i,:3]+Rotation.from_quat(bodies[-1][i,3:]).apply(offset)
                                          for i,offset in e.tools]))
        motor_targets.append(e.control.joint_target_q.numpy().copy())
    record()
    np.savez_compressed(a.output/'initial_state.npz',particle_position=np.asarray(particles),body_pose=np.asarray(bodies),physical_time=np.asarray(times))
    tets=e.model.tet_indices.numpy()
    def determinant(x):
        p=x[tets].astype(np.float64);return np.linalg.det(p[:,1:]-p[:,:1])
    reference=determinant(particles[0]);min_ratio=1.;max_ratio=1.
    checkpoint_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    provenance=source_record()
    def report(status):
        r={'status':status,'mode':'shared_network_inference','scene':'research.w1_dual_grasp','asset':str(a.asset.resolve()),
          'asset_sha256':hashlib.sha256(a.asset.read_bytes()).hexdigest(),'checkpoint':str(a.checkpoint.resolve()),
          'checkpoint_sha256':checkpoint_sha,'checkpoint_step':stored['step'],'newton_source':provenance,
          'device':a.device,'physics_device':str(e.model.device),'physical_dt':e.sim_dt,'substeps':a.substeps,
          'physical_seconds':steps*e.sim_dt,'frames_completed':frames,'frames_requested':a.frames,'nn_backbone_calls':nn_calls,
          'forbidden_solver_calls':calls,'teacher_trajectory_reads':0,'optimizer_updates':0,
          'robot_body_count':e.model.body_count-1,'particle_count':e.model.particle_count,'tet_count':e.model.tet_count,
          'payload_body_id':e.payload_id,'rigid_lift_m':float(bodies[-1][e.payload_id,2]-bodies[0][e.payload_id,2]),
          'soft_center_lift_m':float((particles[-1]-particles[0]).mean(0)[2]),
          'tool_lift_m':(tool_positions[-1][:,2]-tool_positions[0][:,2]).tolist(),
          'minimum_recorded_tet_ratio':min_ratio,'maximum_recorded_tet_ratio':max_ratio,
          'motion':f'Close both grippers, command {LIFT*100:g} cm lift, hold. Near-grasp initial configuration.',
          'initial_finger_opening_m':OPENING,'requested_tree_columns_per_interaction':8,
          'scope':'Original W1 visuals and inertias; fixed torso/head; local-box collision proxies. Dynamic arms/fingers and both payloads. Joint targets only; no payload attachment or pose script.',
          'full_authored_timeline_recorded':frames==a.frames and steps*e.sim_dt>=DURATION-1e-9,
          'wall_seconds':time.perf_counter()-begin,'failure':failure}
        tmp=a.output/'run.tmp';tmp.write_text(json.dumps(r,indent=2)+'\n');tmp.replace(a.output/'run.json')
    report('RUNNING')
    try:
        with torch.no_grad():
            for frame in range(a.frames):
                phase=controller(e,frame*e.frame_dt,wp)
                for substep in range(a.substeps):
                    e.state_0.clear_forces();e.state_1.clear_forces()
                    solver.step(e.state_0,e.state_1,e.control,None,e.sim_dt)
                    e.state_0,e.state_1=e.state_1,e.state_0;steps+=1
                frames+=1;phases.append(phase);record()
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
        (a.output/'quality.json').write_text(json.dumps(quality,indent=2)+'\n')
        if viewer:viewer.close()
    if failure:sys.exit(1)
    print(a.output/'run.json',flush=True)


if __name__=='__main__':main()
