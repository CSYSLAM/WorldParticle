"""Experimental native rod/contact inference with the single shared model."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), '/home/oem/code/engine/newton']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', default='cable.example_cable_initial_rest_pose')
    parser.add_argument('--checkpoint', type=Path, default=ROOT/'checkpoints/shared_surface_joint_v39/shared_strict.pt')
    parser.add_argument('--device', default='cpu'); parser.add_argument('--headless', action='store_true')
    parser.add_argument('--gpu-memory-gib',type=float,default=12.)
    parser.add_argument('--bounded-analytic-contact-buffers',action='store_true',help='Count the complete current primitive AABB witness bound before allocating contact buffers')
    parser.add_argument('--sdf-mesh-contacts',action='store_true',help='Explicit experimental native SDF mesh witnesses in the composed rod/finite-joint scene')
    parser.add_argument('--massless-prescribed',action='store_true',help='Include native zero-mass/zero-inverse-mass bodies in accepted boundary-pose history')
    parser.add_argument('--frames', type=int, default=360); parser.add_argument('--test-final', action='store_true')
    parser.add_argument('--report', type=Path, required=True); parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--stiffness-scale', type=float, default=1.); parser.add_argument('--damping-scale', type=float, default=1.)
    parser.add_argument('--friction', type=float); parser.add_argument('--disable-learning', action='store_true')
    parser.add_argument('--force-shaped-material-ports', action='store_true', help='Experimental six material fields shaped by the current constitutive stress/positive weight')
    parser.add_argument('--fresh-rod-inputs', action='store_true', help='Explicit prescribed-pose contract and always-fresh geometry override of source cache-reuse requests')
    parser.add_argument('--native-float64-state', action='store_true', help='Preserve accepted state precision when native buffers equal its float32 projection; serialize velocity caches for restart')
    parser.add_argument('--bundle-release-events',action='store_true',help='Explicit authored bundle load/hold/removal adapter; no swept impulse at the release teleport')
    parser.add_argument('--dahl-history-features',action='store_true',help='Expose normalized Dahl stress and prior strain increment to the shared relation encoder')
    parser.add_argument('--motion-budget',type=float,help='Explicit numerical surface-travel envelope in metres; smaller values tighten motion admission and the matching expanded query')
    parser.add_argument('--gravity-material-ports',action='store_true',help='Experimental six additional gravity-moment material columns, explicit total budget 18')
    parser.add_argument('--retain-rod-iterate',action='store_true',help='Explicit rod policy: retain accepted regular iterates if later rounds stall; wholly stalled or invalid proposals still reject')
    parser.add_argument('--contact-activation-ray-step',action='store_true',help='Bound new normal activation on prospective correction groups; keep original acceptance and fixed iteration budget')
    parser.add_argument('--contact-motion-ray-step',action='store_true',help='Bound proposals using the remaining original collision-query motion envelope')
    parser.add_argument('--contact-structural-support',action='store_true',help='Condition contact weights with current enabled joint material and effective body mobility')
    parser.add_argument('--caller-pose-history',action='store_true',help='Measure all output velocities from raw accepted pose history across caller pose edits')
    args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    import math
    for value in (args.stiffness_scale, args.damping_scale, args.friction):
        if value is not None and (not math.isfinite(value) or value < 0): parser.error('Finite nonnegative physical parameters required')
    if args.motion_budget is not None and (not math.isfinite(args.motion_budget) or args.motion_budget<=0):parser.error('Finite positive motion budget required')
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.rod_contact_scene import export_model
    from wp_vbd.rod_contact_runtime import RodContactExampleSolver
    from wp_vbd import math3d as so3
    from wp_vbd.gpu_budget import GpuBudget
    budget=GpuBudget(args.device,args.gpu_memory_gib)
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir = str(ROOT/'.local/p0/warp-cache'); wp.init(); wp.set_device(args.device)
    if args.device.startswith('cuda'): wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    raw = args.checkpoint.read_bytes(); stored = torch.load(args.checkpoint, weights_only=True, map_location=args.device)
    configuration=dict(stored['config']['model'])
    if args.force_shaped_material_ports:configuration['rod_contact_force_ports']=True
    if args.fresh_rod_inputs:configuration['rod_prescribed_fresh_inputs']=True
    if args.native_float64_state:configuration['rod_native_float64_state']=True
    if args.bundle_release_events:configuration['rod_prescribed_removal_events']=True
    if args.dahl_history_features:configuration['rod_dahl_history_features']=True
    if args.motion_budget is not None:configuration['native_contact_motion_budget']=args.motion_budget
    if args.gravity_material_ports:configuration['rod_gravity_material_ports']=True
    if args.retain_rod_iterate:configuration['rod_retain_accepted_primal_iterate']=True
    if args.contact_activation_ray_step:configuration['contact_activation_ray_step']=True
    if args.contact_motion_ray_step:configuration['contact_motion_ray_step']=True
    if args.contact_structural_support:configuration['contact_structural_support']=True
    if args.caller_pose_history:configuration['rod_caller_pose_history']=True
    network = MixedSolver(configuration).to(args.device).eval(); network.load_state_dict(stored['model'])
    viewer = newton.viewer.ViewerNull(num_frames=args.frames) if args.headless else newton.viewer.ViewerGL(width=1600, height=900, vsync=False)
    if not args.headless:
        viewer.renderer.set_title('WorldParticle | '+('physical ablation' if args.disable_learning else 'shared inference')+' | '+args.demo)
    example, solvers, _ = make_example(args.demo, reference=False, preserve_defaults=True, viewer=viewer)
    if len(solvers) != 1: raise ValueError('One owned rod/contact VBD subsystem required')
    model = example.model
    model.joint_target_ke.assign(model.joint_target_ke.numpy()*args.stiffness_scale)
    model.joint_target_kd.assign(model.joint_target_kd.numpy()*args.damping_scale)
    if args.friction is not None: model.shape_material_mu.assign(np.full(model.shape_material_mu.shape, args.friction, np.float32))
    def forbidden(*unused, **kwargs): raise AssertionError('Full VBD forbidden during rod/contact prediction')
    newton.solvers.SolverVBD.__init__ = forbidden; newton.solvers.SolverVBD.step = forbidden
    contact_policy=None
    if solvers[0][1].get('rigid_contact_history'):
        from wp_vbd.rigid_contact_memory import authored_policy
        contact_policy=authored_policy(example)
    finite=~np.isin(model.joint_type.numpy(),[int(newton.JointType.FREE),int(newton.JointType.ROD)])
    if np.any(finite):
        from wp_vbd.rod_joint_scene import export_model
        from wp_vbd.rod_joint_runtime import RodJointExampleSolver as RodContactExampleSolver
    export_options={'contact_policy':contact_policy}
    if args.bounded_analytic_contact_buffers:export_options['bounded_analytic_buffers']=True
    if args.sdf_mesh_contacts or args.massless_prescribed:
        if not np.any(finite):raise ValueError('Explicit mesh/massless boundary path currently requires the composed finite-joint adapter')
        export_options.update(sdf_mesh_contacts=args.sdf_mesh_contacts,massless_prescribed=args.massless_prescribed)
    scene = export_model(model, solvers[0][1], args.demo, newton, example.sim_dt, args.device,**export_options)
    solver = RodContactExampleSolver(network, scene, example.state_0, example.sim_dt, wp, newton, model, learned=not args.disable_learning)
    if configuration.get('rod_prescribed_removal_events') and args.demo=='cable.example_cable_bundle_hysteresis':
        from wp_vbd.rod_prescribed_events import BundleReleaseEvents
        solver.prescribed_event_provider=BundleReleaseEvents(example,scene,wp)
    elif args.bundle_release_events:raise ValueError('Bundle release events require the authored bundle scene')
    example.solver = solver; trajectory = []
    def record(state):
        trajectory.append((solver.steps, state.body_q.numpy().copy(), state.body_qd.numpy().copy()))
        if not args.headless and viewer.is_running():
            original, clock = example.state_0, example.sim_time
            example.state_0 = state; example.sim_time = solver.steps*example.sim_dt
            try: example.render()
            finally: example.state_0 = original; example.sim_time = clock
    record(example.state_0); solver.on_substep = record
    begin = time.perf_counter(); frames = 0; failure = None; final = 'NOT_RUN'
    try:
        while viewer.is_running() and frames < args.frames:
            try:
                if viewer.should_step(): example.step(); frames += 1;budget.observe()
                example.render()
                if frames % 10 == 0: print(f'frames={frames} steps={solver.steps} t={example.sim_time:.3f}', flush=True)
            except (ValueError, RuntimeError) as error: failure = str(error); break
        if args.test_final and failure is None:
            try: example.test_final(); final = 'PASS'
            except Exception as error: final = 'FAIL: '+str(error)
        if args.screenshot and not args.headless:
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(args.screenshot)
    finally: viewer.close()
    try:budget.observe(force=True)
    except RuntimeError as error:
        if failure is None:failure=str(error)
    label = args.demo.split('.')[-1].removeprefix('example_')+'_train_default'
    samples = stored.get('active_weight_optimizer_samples', stored.get('sampled_updates', {})).get(label, 0)
    result = {'status': 'INFERENCE_EXECUTED' if failure is None else 'MODEL_FAILED', 'failure': failure,
              'demo': args.demo, 'frames': frames, 'requested_frames': args.frames, 'physical_substeps': solver.steps,
              'physical_seconds': solver.steps*example.sim_dt, 'seconds': time.perf_counter()-begin,
              'original_test_final': final, 'checkpoint_sha256': hashlib.sha256(raw).hexdigest(),
              'body_count': model.body_count, 'rod_count': len(scene.rod.joints), 'learning_enabled': not args.disable_learning,
              'scene_contract':scene.metadata['contract'],
              'finite_joint_count':len(scene.joint_spec.owners) if hasattr(scene,'joint_spec') else 0,
              'prescribed_body_ids':scene.metadata.get('prescribed_body_ids',[]),
              'rigid_mesh_contact_contract':scene.metadata.get('rigid_mesh_contact_contract'),
              'mesh_shape_ids':scene.metadata.get('mesh_shape_ids',[]),
              'prescribed_body_policy':scene.metadata.get('prescribed_body_policy'),
              'cross_step_contact_history':scene.metadata.get('cross_step_contact_history',False),
              'contact_matching_policy':scene.metadata.get('contact_matching_policy'),
              'gpu_resource_budget':budget.report(),
              'active_weight_rod_contact_training_samples': samples, 'training_status': 'SAMPLED_IN_ACTIVE_WEIGHT_ANCESTRY' if samples else 'UNTRAINED_ROD_CONTACT_TRANSFER',
              'full_vbd_prediction_calls': 0, 'teacher_trajectory_reads': 0, 'new_optimizer_updates': 0,
              'history_keys': sorted(solver.history), 'collision_queries': solver.oracle.queries,
              'collision_capacity_policy':solver.oracle.capacity_policy,
              'collision_capacity':len(solver.oracle.contacts.rigid_contact_shape0),
              'maximum_collision_candidates':solver.oracle.maximum_candidate_count,
              'maximum_allocated_contact_capacity':solver.oracle.maximum_allocated_contact_capacity,
              'bounded_analytic_buffer_reallocations':solver.oracle.capacity_reallocations,
              'surface_motion_budget_m':solver.oracle.motion_budget,
              'source_cache_reuse_requests_overridden':solver.source_cache_reuse_requests,
              'native_fixed_pose_input_contract':solver.prescribed_fresh_inputs,
              'native_float64_state_contract':solver.native_float64_state,
              'geometry_removal_event_steps':solver.geometry_removal_event_steps,
              'physical_scene_contract': scene.metadata['contract'],
              'applied_parameters': {'stiffness_scale': args.stiffness_scale, 'damping_scale': args.damping_scale, 'friction': args.friction},
              'implementation_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/wp_vbd').glob('*.py')},
              'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'runtime_model_config_overrides':{**({'rod_contact_force_ports':True} if args.force_shaped_material_ports else {}),
                                                **({'rod_prescribed_fresh_inputs':True} if args.fresh_rod_inputs else {}),
                                                **({'rod_native_float64_state':True} if args.native_float64_state else {}),
                                                **({'rod_prescribed_removal_events':True} if args.bundle_release_events else {}),
                                                **({'rod_dahl_history_features':True} if args.dahl_history_features else {}),
                                                **({'native_contact_motion_budget':args.motion_budget} if args.motion_budget is not None else {}),
                                                **({'rod_gravity_material_ports':True} if args.gravity_material_ports else {}),
                                                **({'rod_retain_accepted_primal_iterate':True} if args.retain_rod_iterate else {}),
                                                **({'contact_activation_ray_step':True} if args.contact_activation_ray_step else {}),
                                                **({'contact_motion_ray_step':True} if args.contact_motion_ray_step else {}),
                                                **({'contact_structural_support':True} if args.contact_structural_support else {}),
                                                **({'rod_caller_pose_history':True} if args.caller_pose_history else {})},
              'scope': 'Experimental owned FREE/ROD and fresh analytic contacts. Actual selected training, complete native quality, cached contacts and realtime are separate checks.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    n=model.body_count;hq=solver.history['rod_previous_q'][0,:n];hx=solver.history['rod_previous_x'][0,:n]
    accepted={'final_accepted_body_pose_float64':torch.cat((hx-so3.rotate(hq,scene.body_com[:n]),hq),-1).cpu().numpy()}
    accepted.update({'final_history__'+key:value.cpu().numpy() for key,value in solver.history.items()})
    if solver.native_float64_state:
        accepted['final_accepted_body_velocity_float64']=torch.cat((solver.history['rod_native_v'][0,:n],solver.history['rod_native_omega'][0,:n]),-1).cpu().numpy()
    np.savez_compressed(args.report.with_suffix('.npz'), physical_substep=np.asarray([r[0] for r in trajectory]),
                        body_pose=np.stack([r[1] for r in trajectory]), body_velocity=np.stack([r[2] for r in trajectory]),**accepted)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key: result[key] for key in ('status', 'failure', 'physical_substeps', 'original_test_final', 'seconds')}, indent=2), flush=True)
    if failure is not None or final.startswith('FAIL'): raise SystemExit(1)


if __name__ == '__main__': main()
