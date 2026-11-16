"""Experimental native mixed-contact inference using one shared checkpoint."""
import argparse
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), '/home/oem/code/engine/newton']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', default='vbd.example_vbd_soft_rigid_contact')
    parser.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/shared_urdf_contact_v32/shared_coverage.pt')
    parser.add_argument('--device', default='cpu'); parser.add_argument('--headless', action='store_true')
    parser.add_argument('--gpu-memory-gib',type=float,default=12.,help='Process GPU budget; reserve two GiB outside the Torch allocator')
    parser.add_argument('--frames', type=int, default=300); parser.add_argument('--test-final', action='store_true')
    parser.add_argument('--report', type=Path, required=True); parser.add_argument('--screenshot', type=Path)
    parser.add_argument('--disable-learning', action='store_true')
    parser.add_argument('--neural-graphs',action='store_true',help='Experimental replay of pure attention blocks; physical/contact steps remain eager')
    parser.add_argument('--component-batch',action='store_true',help='Experimental equivalent component operator; does not change neural weights or training counts')
    parser.add_argument('--predictive-contact-metric',action='store_true',help='Experimental inverse-secant metric for newly activating contacts; no new physical force or iterations')
    parser.add_argument('--allow-internal-stall',action='store_true',help='Keep a rejected internal proposal unchanged; the last fixed-budget iteration must still pass')
    parser.add_argument('--surface-joints', action='store_true', help='Experimental analytic face/edge contacts and finite joints in the same shared response')
    parser.add_argument('--particle-self-contact', action='store_true', help='Experimental reciprocal particle VF/EE plus elastic/surface/joint assembly')
    parser.add_argument('--prescribed-massless-bodies',action='store_true',help='Explicit authored massless body pose commands with accepted COM history')
    parser.add_argument('--responsive-self-only',action='store_true',help='Self pairs must contain a responsive primitive vertex; prescribed meshes remain colliders, but their mutual separation is not asserted')
    parser.add_argument('--retain-self-iterate',action='store_true',help='Experimental fixed-budget retention of already accepted regular self-contact iterates')
    parser.add_argument('--compact-frozen-contact-metric',action='store_true',help='Experimental omission of zero solve-metric blocks; full physical candidates/forces/path tests remain')
    parser.add_argument('--self-motion-budget', type=float, default=.04, help='Self-contact particle motion envelope in metres')
    parser.add_argument('--self-max-rows',type=int,default=200000,help='Complete VF/EE candidate capacity; overflow rejects without truncation')
    parser.add_argument('--self-broad-phase',choices=('chunked','kdtree'),default='chunked',help='Experimental host sphere-tree prefilter with original F64 postfilter on the owning device')
    for name in ('friction', 'contact-stiffness', 'contact-damping', 'elastic-scale'):
        parser.add_argument('--' + name, type=float)
    args = parser.parse_args()
    if args.neural_graphs and not args.device.startswith('cuda'):parser.error('--neural-graphs requires CUDA')
    if args.surface_joints and args.particle_self_contact:parser.error('Select one complete experimental assembly')
    if args.retain_self_iterate and not args.particle_self_contact:parser.error('Self iterate retention requires the complete self-contact assembly')
    if args.responsive_self_only and not args.particle_self_contact:parser.error('Responsive pair eligibility requires particle self-contact')
    if args.prescribed_massless_bodies and not args.particle_self_contact:parser.error('This prescribed-body contract requires the complete self-contact assembly')
    if not __import__('math').isfinite(args.self_motion_budget) or args.self_motion_budget<=0:parser.error('Finite positive self motion budget required')
    if args.self_max_rows<=0:parser.error('Positive self-contact capacity required')
    if args.report.exists(): raise FileExistsError(args.report)
    for name in ('friction', 'contact_stiffness', 'contact_damping', 'elastic_scale'):
        value = getattr(args, name)
        if value is not None and (not __import__('math').isfinite(value) or value < 0): parser.error(name + ' must be finite and nonnegative')
    implementation = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT / 'src/wp_vbd').glob('*.py')}
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.mixed_contact_scene import export_model
    from wp_vbd.mixed_contact_runtime import MixedContactExampleSolver
    from wp_vbd.gpu_budget import GpuBudget
    budget=GpuBudget(args.device,args.gpu_memory_gib)
    if args.surface_joints:
        from wp_vbd.surface_joint_scene import export_model
        from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    if args.particle_self_contact:
        from wp_vbd.self_contact_scene import export_model
        from wp_vbd.self_contact_runtime import SelfContactExampleSolver
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir = str(ROOT / '.local/p0/warp-cache'); wp.init(); wp.set_device(args.device)
    if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    raw = args.checkpoint.read_bytes(); stored = torch.load(io.BytesIO(raw), weights_only=True, map_location=args.device)
    model_config=copy.deepcopy(stored['config']['model'])
    if args.component_batch:model_config['mixed_component_batch']=True
    if args.predictive_contact_metric:model_config['predictive_contact_metric']=True
    if args.allow_internal_stall:model_config['allow_internal_stall']=True
    if args.particle_self_contact:model_config['particle_self_contact_v58']=True
    if args.retain_self_iterate:model_config['self_retain_accepted_primal_iterate']=True
    if args.compact_frozen_contact_metric:model_config['compact_frozen_contact_metric']=True
    network = MixedSolver(model_config).to(args.device).eval(); network.load_state_dict(stored['model'])
    from wp_vbd.neural_cuda_graph import enable_attention_replay,attention_replay_report
    if args.neural_graphs:enable_attention_replay(network)
    viewer = newton.viewer.ViewerNull(num_frames=args.frames) if args.headless else newton.viewer.ViewerGL(width=1600, height=900, vsync=False)
    if not args.headless:
        viewer.renderer.set_title('WorldParticle | shared elastic contact | ' + args.demo)
        viewer.gui.ui.io.set_ini_filename(str(ROOT / '.local/mixed_contact_v33/viewer.ini'))
    example, configurations, _ = make_example(args.demo, reference=False, viewer=viewer, preserve_defaults=True)
    if len(configurations) != 1: raise ValueError('A single owned mixed subsystem is required')
    full_surface = bool(getattr(getattr(example,'contacts',None), '_enable_rigid_soft_full_surface_contact', False))
    if full_surface and not (args.surface_joints or args.particle_self_contact): raise ValueError('Source full-surface contact requires a barycentric operator; particle-only replacement is not admitted')
    if args.surface_joints and not full_surface: raise ValueError('Keep the authored full-surface collision contract for this experimental route')
    model = example.model; configuration = configurations[0][1]
    for argument, soft_name, shape_name in (('friction', 'soft_contact_mu', 'shape_material_mu'), ('contact_stiffness', 'soft_contact_ke', 'shape_material_ke'), ('contact_damping', 'soft_contact_kd', 'shape_material_kd')):
        value = getattr(args, argument)
        if value is not None:
            setattr(model, soft_name, value); array = getattr(model, shape_name); array.assign(np.full(array.shape, value, np.float32))
    if args.elastic_scale is not None:
        for name in ('tet_materials', 'tri_materials'):
            array = getattr(model, name, None)
            if array is not None:
                values = array.numpy(); values[:, :2] *= args.elastic_scale; array.assign(values)
        array = model.edge_bending_properties
        if array is not None:
            values = array.numpy(); values[:, 0] *= args.elastic_scale; array.assign(values)
    def forbidden(*unused, **kwargs): raise AssertionError('Full VBD forbidden in mixed prediction')
    newton.solvers.SolverVBD.__init__ = forbidden; newton.solvers.SolverVBD.step = forbidden
    extra={}
    if args.particle_self_contact:
        from wp_vbd.self_contact_scene import authored_full_surface
        extra={'motion_budget':args.self_motion_budget,'full_surface':authored_full_surface(example),'broad_phase':args.self_broad_phase,'max_rows':args.self_max_rows}
        if args.prescribed_massless_bodies:extra['prescribed_massless_bodies']=True
    scene = export_model(model, configuration, args.demo, newton, example.sim_dt, args.device, **extra)
    if args.responsive_self_only:scene.metadata['self_contact_responsive_pairs_only']=True
    self_camera=None
    if args.particle_self_contact and not args.headless:
        # The standalone viewer starts at ground level, while some official
        # cloth assets start ten metres up. Frame the asset and ground together
        # so the live fall remains visible throughout the original task.
        points=scene.rest_x[:-1].detach().cpu().numpy()
        lower,upper=points.min(0),points.max(0)
        up=int(viewer.camera.up_axis)
        planes=model.shape_type.numpy()==int(newton.GeoType.PLANE)
        if planes.any():lower[up]=min(lower[up],float(model.shape_transform.numpy()[planes,up].min()))
        center=(lower+upper)/2;distance=max(1.5*float(np.linalg.norm(upper-lower)),4.)
        viewer.set_camera(wp.vec3(0.,0.,0.),pitch=-18.,yaw=90.)
        direction=np.asarray(tuple(viewer.camera.get_front()))
        eye=center-distance*direction
        viewer.set_camera(wp.vec3(*eye))
        self_camera={'position':eye.tolist(),'target':center.tolist(),'pitch_deg':-18.,'yaw_deg':90.}
    if args.particle_self_contact:
        solver = SelfContactExampleSolver(network, scene, example.sim_dt, wp, newton, model, example.state_0, learned=not args.disable_learning)
    elif args.surface_joints:
        solver = SurfaceJointExampleSolver(network, scene, example.sim_dt, wp, newton, model, example.state_0, learned=not args.disable_learning)
    else:
        solver = MixedContactExampleSolver(network, scene, example.sim_dt, wp, newton, model, learned=not args.disable_learning)
    example.solver = solver
    stream = wp.stream_from_torch(torch.cuda.current_stream()) if args.device.startswith('cuda') else None
    def scope(): return wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice('cpu')
    trajectory = []; contact_counts = []; feature_counts = []; self_counts=[]; diagnostic = None
    def record(state):
        body_pose=state.body_q.numpy().copy() if state.body_q is not None else np.empty((0,7),np.float32)
        body_velocity=state.body_qd.numpy().copy() if state.body_qd is not None else np.empty((0,6),np.float32)
        trajectory.append((solver.steps, state.particle_q.numpy().copy(), body_pose,
                           state.particle_qd.numpy().copy(), body_velocity))
        if solver.last_diagnostic is not None: contact_counts.append(solver.last_diagnostic['contacts'])
        if solver.last_diagnostic is not None and 'active_soft_feature_rows' in solver.last_diagnostic:
            feature_counts.append(solver.last_diagnostic['active_soft_feature_rows'])
        if solver.last_diagnostic is not None and 'active_self_rows_by_kind' in solver.last_diagnostic:
            self_counts.append(solver.last_diagnostic['active_self_rows_by_kind'])
        if not args.headless and viewer.is_running():
            previous = example.state_0; clock = example.sim_time; example.state_0 = state; example.sim_time = solver.steps * example.sim_dt
            try: example.render()
            finally: example.state_0 = previous; example.sim_time = clock
    record(example.state_0); solver.on_substep = record
    begin = time.perf_counter(); frames = 0; failure = None; final = 'NOT_RUN';post_step_checks=0
    try:
        while viewer.is_running() and frames < args.frames:
            try:
                with scope():
                    if viewer.should_step():
                        example.step(); frames += 1
                        if args.test_final and callable(getattr(example,'test_post_step',None)):
                            example.test_post_step();post_step_checks+=1
                        budget.observe()
                    example.render()
                if frames % 10 == 0: print(f'frames={frames} steps={solver.steps} t={example.sim_time:.3f}', flush=True)
            except (RuntimeError, ValueError, AssertionError) as error:
                failure = str(error); break
        if args.test_final and failure is None:
            try: example.test_final(); final = 'PASS'
            except Exception as error: final = 'FAIL: ' + str(error)
        if args.screenshot and not args.headless:
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(args.screenshot)
    finally: viewer.close()
    samples = stored.get('active_weight_optimizer_samples', stored.get('sampled_updates', {}))
    if args.device.startswith('cuda'):torch.cuda.synchronize()
    budget.observe(force=True)
    result = {'status': 'INFERENCE_EXECUTED' if failure is None and not final.startswith('FAIL') else 'MODEL_FAILED', 'demo': args.demo, 'frames': frames, 'requested_frames': args.frames, 'physical_substeps': solver.steps, 'physical_seconds': solver.steps * example.sim_dt, 'seconds': time.perf_counter() - begin, 'failure': failure, 'original_test_final': final, 'checkpoint_sha256': hashlib.sha256(raw).hexdigest(), 'learning_enabled': not args.disable_learning, 'optimizer_updates_during_inference': 0, 'current_mixed_training_samples': 0, 'training_status': 'UNTRAINED_MIXED_CONTACT_TRANSFER', 'inherited_dataset_labels': sorted(samples), 'particle_count': model.particle_count, 'body_count': model.body_count, 'pinned_particle_count': len(scene.fixed_particle_ids), 'collision_queries': solver.oracle.queries, 'maximum_contact_rows': max(contact_counts, default=0), 'history_keys': sorted(solver.history), 'full_vbd_prediction_calls': 0, 'teacher_trajectory_reads': 0, 'implementation_sha256': implementation, 'applied_parameters': {name: getattr(args, name) for name in ('friction', 'contact_stiffness', 'contact_damping', 'elastic_scale')}, 'source_controls_and_render_preserved': True, 'scope': 'New elastic/particle-shape physics path with the existing shared weights. This is not selected mixed-contact training, all-demo quality or realtime verification.'}
    result['original_post_step_checks']=post_step_checks
    if args.test_final and hasattr(example,'max_penetration'):result['maximum_authored_penetration_m']=float(example.max_penetration)
    result['gpu_resource_budget']=budget.report()
    result['neural_graph_replay']=attention_replay_report(network)
    result['execution_mode']='eager_physics_with_attention_replay' if result['neural_graph_replay']['captures'] else 'eager'
    label = {'vbd.example_vbd_soft_rigid_contact': 'soft_rigid_train_default',
             'vbd.example_vbd_soft_rigid_mix_contact': 'soft_rigid_mix_train_default',
             'vbd.example_vbd_gripper_soft_triangle': 'gripper_triangle_train_default',
             'vbd.example_vbd_gripper_soft_grid': 'gripper_grid_train_default',
             'cloth.example_cloth_bending': 'cloth_bending_self_train_default',
             'cloth.example_cloth_hanging': 'cloth_hanging_self_train_default',
             'cloth.example_cloth_twist': 'cloth_twist_self_train_default'}.get(args.demo)
    if args.demo=='multiphysics.example_softbody_dropping_to_cloth':label='soft_cloth_self_train_default'
    result['current_mixed_training_samples'] = stored.get('sampled_updates', {}).get(label, 0)
    result['active_weight_mixed_training_samples'] = samples.get(label, 0)
    result['training_status'] = 'SAMPLED_IN_ACTIVE_WEIGHT_ANCESTRY' if result['active_weight_mixed_training_samples'] else 'UNTRAINED_MIXED_CONTACT_TRANSFER'
    if 'active_weight_official_optimizer_samples' in stored:
        result['current_mixed_training_samples']=stored['sampled_official_updates'].get(args.demo,0)
        result['active_weight_mixed_training_samples']=stored['active_weight_official_optimizer_samples'].get(args.demo,0)
        result['training_status']='SAMPLED_IN_ACTIVE_WEIGHT_ANCESTRY' if result['active_weight_mixed_training_samples'] else 'UNTRAINED_MIXED_CONTACT_TRANSFER'
        result['training_count_scope']='Official demo counts embedded from the selected optimizer prefix, independent of dataset naming'
    elif 'trained_official_demos' in stored:
        trained=args.demo in stored['trained_official_demos']
        result['training_status']='SAMPLED_IN_ACTIVE_WEIGHT_ANCESTRY' if trained else 'UNTRAINED_MIXED_CONTACT_TRANSFER'
        if not trained:result['current_mixed_training_samples']=result['active_weight_mixed_training_samples']=0
        elif not result['active_weight_mixed_training_samples']:
            result['current_mixed_training_samples']=result['active_weight_mixed_training_samples']=None
        result['training_count_scope']='Selected artifact declares official demo ancestry; per-demo numeric counts unavailable when legacy dataset naming does not match'
    result['scope'] = 'One shared checkpoint and native asset/control/render. Actual selected training, complete mixed task behavior and physical realtime are separate checks.'
    result['physical_scene_contract']=scene.metadata['contract']
    result['weighted_surface_joint_path']=args.surface_joints
    if args.surface_joints:
        result['maximum_active_soft_feature_rows']={str(size):max((row[str(size)] for row in feature_counts), default=0) for size in (1,2,3)}
    result['runtime_model_config_overrides']={'mixed_component_batch':True} if args.component_batch else {}
    if args.predictive_contact_metric:result['runtime_model_config_overrides']['predictive_contact_metric']=True
    if args.allow_internal_stall:result['runtime_model_config_overrides']['allow_internal_stall']=True
    if args.compact_frozen_contact_metric:result['runtime_model_config_overrides']['compact_frozen_contact_metric']=True
    if args.particle_self_contact:
        result['runtime_model_config_overrides']['particle_self_contact_v58']=True
        if args.retain_self_iterate:result['runtime_model_config_overrides']['self_retain_accepted_primal_iterate']=True
        result['self_accepted_primal_iterate_policy']=bool(model_config.get('self_retain_accepted_primal_iterate',False))
        result['self_motion_budget_m']=args.self_motion_budget
        result['self_candidate_capacity']=args.self_max_rows
        result['maximum_active_self_rows_by_kind']={str(kind):max((row[str(kind)] for row in self_counts),default=0) for kind in (9,10)}
        result['self_EE_reciprocal_topology_rule']=scene.metadata['self_contact_EE_topology_rule']
        result['self_shape_full_surface']=scene.metadata['self_shape_full_surface']
        result['self_contact_broad_phase']=scene.metadata['self_contact_broad_phase']
        result['self_contact_responsive_pairs_only']=args.responsive_self_only
        result['self_contact_pair_scope']='at_least_one_responsive_primitive_vertex' if args.responsive_self_only else 'all_authored_topology_pairs'
        result['source_bvh_rebuild_requests']=solver.source_bvh_rebuild_requests
        result['self_geometry_query_policy']='Fresh expanded geometry every physical step; authored frame BVH rebuild requests acknowledged without physical advancement'
        result['self_scene_camera']=self_camera
        result['authored_shape_contact_feature_eligibility_preserved']=True
        result['selected_official_training_admissions_added']=0
        result['experimental_self_contact_transfer']=True
    args.report.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.report.with_suffix('.npz'), physical_substep=np.asarray([r[0] for r in trajectory]), particle_position=np.stack([r[1] for r in trajectory]), body_pose=np.stack([r[2] for r in trajectory]), particle_velocity=np.stack([r[3] for r in trajectory]), body_velocity=np.stack([r[4] for r in trajectory]))
    result['native_recording_includes_velocities']=True
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('status', 'frames', 'physical_substeps', 'failure', 'original_test_final', 'seconds')}, indent=2), flush=True)
    if failure or final.startswith('FAIL'): sys.exit(1)


if __name__ == '__main__': main()
