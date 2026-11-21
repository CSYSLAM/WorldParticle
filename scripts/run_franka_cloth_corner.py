"""Frozen shared-model Franka pinches and lifts one free cloth corner."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wp_vbd.inference_policy import CLI_POLICIES,resolve_policy
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'model_releases/shared_vbd_v24_cloth40/model.pt')
    p.add_argument('--execution-policy',choices=CLI_POLICIES,default='auto')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    p.add_argument('--frames', type=int, help='Defaults to the full control timeline')
    p.add_argument('--grasp-style',choices=('overhead','side'),default='overhead')
    p.add_argument('--fold-profile',choices=('corner','edge'),default='edge')
    p.add_argument('--substeps', type=int, default=3)
    p.add_argument('--friction', type=float, default=2.)
    p.add_argument('--geometry-only', action='store_true')
    p.add_argument('--window',action='store_true',help='Display accepted states while inference advances')
    p.add_argument('--disable-learning',action='store_true',
                   help='Matched physical-round ablation; never labeled or rendered as neural inference')
    p.add_argument('--tangent-support-scale', type=float, default=64.)
    p.add_argument('--terminal-velocity', action=argparse.BooleanOptionalAction, default=True,
                   help='Use the frozen endpoint-velocity readout; disable for matched diagnosis')
    a = p.parse_args()
    a.execution_policy=resolve_policy(a.execution_policy,scene='cloth',device='cpu' if a.geometry_only else a.device)
    if (a.frames is not None and a.frames < 1) or a.substeps < 1:
        p.error('Positive frames and substeps required')
    a.output = a.output.expanduser().resolve()
    if (a.output / 'run.json').exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True, exist_ok=True)
    from wp_vbd.newton_source import activate_newton_source, source_record
    activate_newton_source()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership = claim_cuda('cpu' if a.geometry_only else a.device, ROOT)
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd.franka_cloth_corner_scene import build, controller, command, DURATION, TABLE_TOP, TABLE_BOUNDS, TOOL_OFFSET
    from wp_vbd.w1_table_clearance import TableClearance
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir = str(ROOT / '.local/p0/warp-cache')
    wp.init()
    device = 'cpu' if a.geometry_only else a.device
    wp.set_device(device)
    if device.startswith('cuda'):
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    options = dict(iterations=10, integrate_with_external_rigid_solver=False,
        particle_enable_self_contact=False, rigid_joint_linear_ke=1e5,
        rigid_joint_angular_ke=1e5, rigid_joint_linear_kd=0., rigid_joint_angular_kd=0.,
        rigid_body_contact_buffer_size=64, rigid_body_particle_contact_buffer_size=256)
    signature = inspect.signature(newton.solvers.SolverVBD).bind(None, **options)
    signature.apply_defaults()
    configuration = {k: v for k, v in signature.arguments.items() if k not in ('model', 'collision_pipeline')}
    forbidden_calls = dict(vbd_construct=0, vbd_step=0, featherstone_construct=0, featherstone_step=0)
    def forbid(name):
        def fail(*args, **kwargs):
            forbidden_calls[name] += 1
            raise AssertionError('Native dynamics forbidden in model demo: ' + name)
        return fail
    for cls, prefix in [(newton.solvers.SolverVBD, 'vbd'), (newton.solvers.SolverFeatherstone, 'featherstone')]:
        cls.__init__ = forbid(prefix + '_construct')
        cls.step = forbid(prefix + '_step')
    e = build(wp, newton, device=device, substeps=a.substeps, friction=a.friction,grasp_style=a.grasp_style,fold_profile=a.fold_profile)
    if a.frames is None:a.frames=round(e.duration*60)
    clearance = TableClearance(e.model, payload_id=-999, bounds=TABLE_BOUNDS, top=TABLE_TOP)
    positions, velocities, bodies, body_velocities, times, targets, phases, gaps = [], [], [], [], [], [], [], []
    nn_calls = 0
    network = solver = scene = None
    completed = 0
    begin = time.perf_counter()
    def record(now):
        positions.append(e.state_0.particle_q.numpy().copy())
        velocities.append(e.state_0.particle_qd.numpy().copy())
        bodies.append(e.state_0.body_q.numpy().copy())
        body_velocities.append(e.state_0.body_qd.numpy().copy())
        times.append(float(now))
        targets.append(e.control.joint_target_q.numpy().copy())
        gaps.append(dict(time=float(now), **clearance.measure(bodies[-1])))
    def save():
        tmp = a.output / 'trajectory.tmp'
        with tmp.open('wb') as file:
            np.savez_compressed(file, particle_position=positions, particle_velocity=velocities,
                body_pose=bodies, body_velocity=body_velocities, physical_time=times,
                motor_target=targets, phase=phases, faces=e.model.tri_indices.numpy(),
                particle_rest=e.model.particle_q.numpy(), corner_id=e.corner_id, patch_ids=e.patch_ids)
        tmp.replace(a.output / 'trajectory.npz')
    def plain(value):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
        if isinstance(value, np.generic):
            return value.item()
        return str(value)
    def report(status, failure=None):
        result = dict(execution_policy=a.execution_policy,status=status, scene='research.franka_cloth_corner',
            mode='asset_geometry_preview' if a.geometry_only else ('physical_only_diagnostic' if a.disable_learning else 'shared_network_inference'),
            learning_enabled=bool(not a.disable_learning and not a.geometry_only),
            robot_asset='franka_emika_panda/fr3_franka_hand.urdf',
            grasp_style=a.grasp_style,authored_duration_s=e.duration,
            fold_profile=a.fold_profile,
            terminal_velocity_enabled=bool(a.terminal_velocity and not a.geometry_only and not a.disable_learning),
            newton_source=source_record(), device=device,
            checkpoint=None if a.geometry_only else str(a.checkpoint.resolve()),
            checkpoint_sha256=None if a.geometry_only else checkpoint_sha,
            checkpoint_step=None if a.geometry_only else stored['step'],
            model_architecture=None if network is None else network.architecture_id,
            frames_completed=completed, frames_requested=a.frames, physical_seconds=times[-1] if times else 0.,
            physical_dt=e.sim_dt, substeps=a.substeps,
            physical_substeps=0 if solver is None else solver.steps, nn_backbone_calls=nn_calls,
            wall_seconds=time.perf_counter()-begin, forbidden_solver_calls=forbidden_calls,
            teacher_trajectory_reads=0, new_optimizer_updates=0,
            analytic_predictor_rounds=None if network is None else network.config.get('analytic_predictor_rounds', 0),
            model_unroll=None if network is None else network.config.get('unroll'),
            analytic_elastic_postsolve=False if network is None else network.config.get('analytic_elastic_postsolve', False),
            adaptive_retries=0 if solver is None else solver.adaptive_retries,
            adaptive_recovery_modes={} if solver is None else solver.adaptive_recovery_counts,
            adaptive_response_rounds=0 if solver is None else solver.adaptive_network_rounds,
            adaptive_min_accepted_dt_s=None if solver is None else solver.adaptive_min_accepted_dt,
            full_authored_timeline_recorded=bool(not a.geometry_only and times and times[-1]>=e.duration-1e-8),
            particle_count=e.model.particle_count, triangle_count=e.model.tri_count,
            corner_id=e.corner_id, grasp_patch_ids=e.patch_ids.tolist(),
            minimum_robot_visual_table_clearance_m=min((g['minimum_robot_visual_table_clearance_m'] for g in gaps if g['minimum_robot_visual_table_clearance_m'] is not None), default=None),
            physical_parameters=dict(friction=a.friction, membrane_stiffness=300., bending_stiffness=.0015,
                soft_tangent_support_scale=a.tangent_support_scale, cloth_fixed_particles=0,
                initial_edge_fold_width_m=.0225 if a.grasp_style=='overhead' else 0.,
                robot_motor_targets_only=True, cloth_attachments=0, cloth_self_contact=False, table_height_m=TABLE_TOP),
            solver_configuration=configuration,
            scope='Dynamic FR3, free square cloth, frictional single-corner pinch. No pins, attachments, native dynamics, teacher trajectory or cloth pose override. Self-contact is not enabled in this simple drape.',
            failure=failure)
        tmp = a.output / 'run.tmp'
        cache=getattr(scene,'_response_index_workspace',None)
        if cache is not None:result['response_index_cache']=cache.report()
        from wp_vbd.diagnostic_io import to_plain
        tmp.write_text(json.dumps(to_plain(result), ensure_ascii=False, indent=2, default=plain)+'\n')
        tmp.replace(a.output / 'run.json')
    record(0.)
    if a.geometry_only:
        for now in np.linspace(0., e.duration, 105)[1:]:
            controller(e, float(now), wp)
            q=e.model.joint_q.numpy()
            q[:9]=e.control.joint_target_q.numpy()[:9]
            newton.eval_fk(e.model, wp.array(q, dtype=wp.float32, device='cpu'), e.model.joint_qd, e.state_0)
            record(now)
        completed=len(times)-1
        save()
        report('COMPLETE')
        (a.output/'table_visual_clearance.json').write_text(json.dumps(gaps, indent=2)+'\n')
        print('Geometry planned; minimum visual table clearance:',min(g['minimum_robot_visual_table_clearance_m'] for g in gaps if g['minimum_robot_visual_table_clearance_m'] is not None))
        return
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd.adaptive_step import AdaptiveStepConfig
    from wp_vbd.gpu_budget import GpuBudget
    budget=GpuBudget(device, 16.)
    checkpoint_sha=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
    stored=torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    config=dict(stored.get('model_config') or stored['config']['model'])
    config.update(mimic_metric_tangent=True, contact_aware_warm_start=True, periodic_bending=True)
    from wp_vbd.inference_policy import configure
    config=configure(config,a.execution_policy)
    network=MixedSolver(config).to(device).eval()
    network.load_state_dict(stored['model'], strict=True)
    if not a.terminal_velocity or a.disable_learning:network.terminal_velocity=None
    scene=export_model(e.model, configuration, 'research.franka_cloth_corner', newton, e.sim_dt, device)
    scene.metadata.update(detection_motion_budget=.012, bounded_motion_forecast_target_only=True,
        joint_tree_coarse_modes='scalar_tree_twelve_columns_v307',
        articulation_material_motion_modes=True, independent_object_motion_modes=True,
        soft_tangent_support_scale=a.tangent_support_scale, force_aware_material_anticipation=True,
        compact_numerical_diagnostics=True, repair_native_mimic_roundoff=True,
        surface_tangent_memory=True, surface_tangent_history_capacity=8192, record_grasp_forces=True)
    solver=SurfaceJointExampleSolver(network, scene, e.sim_dt, wp, newton, e.model, e.state_0)
    solver.learned=not a.disable_learning
    solver.enable_adaptive_steps(AdaptiveStepConfig(maximum_dt=e.sim_dt, minimum_dt=1/600,
        maximum_attempts=256, maximum_substeps=128, motion_safety=.8, relative_update_limit=.15))
    def count(module, args, result):
        nonlocal nn_calls
        nn_calls+=1
    hook=network.protected_response.hierarchy.register_forward_hook(count)
    source_hashes={}
    for source in [Path(__file__), *sorted((ROOT/'src/wp_vbd').glob('*.py'))]:
        relative=source.relative_to(ROOT)
        payload=source.read_bytes()
        source_hashes[str(relative)]=hashlib.sha256(payload).hexdigest()
        target=a.output/'implementation_snapshot'/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
    (a.output/'implementation_snapshot_manifest.json').write_text(json.dumps(source_hashes, indent=2)+'\n')
    begin=time.perf_counter()
    from wp_vbd.live_view import LiveView
    live=LiveView(e.model,enabled=a.window,position=(.65,-1.0,.8),target=(-.07,-.57,.32))
    report('RUNNING')
    failure=None
    try:
        with torch.no_grad():
            while completed<a.frames:
                name=controller(e, completed*e.frame_dt, wp)
                for _ in range(e.sim_substeps):
                    e.state_0.clear_forces()
                    e.state_1.clear_forces()
                    solver.step(e.state_0, e.state_1, e.control, None, e.sim_dt)
                    e.state_0, e.state_1=e.state_1, e.state_0
                completed+=1
                phases.append(name)
                record(solver.physical_time)
                if not live.present(e.state_0,solver.physical_time):break
                if completed%10==0 or completed==1:
                    report('RUNNING')
                    print(json.dumps(dict(frame=completed,time=times[-1],phase=name,
                        cloth_com=np.mean(positions[-1],axis=0).tolist(),
                        corner=positions[-1][e.corner_id].tolist(),
                        minimum_cloth_z=float(positions[-1][:,2].min()),nn=nn_calls,
                        wall_seconds=time.perf_counter()-begin)),flush=True)
                    budget.observe()
                if completed%30==0:
                    save()
    except Exception:
        failure=traceback.format_exc()
        print(failure, flush=True)
        (a.output/'failure_diagnostic.json').write_text(json.dumps(solver.last_diagnostic, default=plain, indent=2)+'\n')
    finally:
        live.close()
        save()
        report('FAILED' if failure else 'COMPLETE' if completed==a.frames else 'INTERRUPTED', failure)
        final=solver.read(e.state_0)
        torch.save({'state':{k:getattr(final,k).detach().cpu() for k in ('x','q','v','omega')},
            'history':{k:v.detach().cpu() for k,v in (final.history or {}).items()},
            'physical_time':solver.physical_time,'model_config':dict(network.config)}, a.output/'final_runtime_state.pt')
        (a.output/'table_visual_clearance.json').write_text(json.dumps(gaps, indent=2)+'\n')
        hook.remove()
    if failure:
        raise RuntimeError('Franka cloth model run did not complete; see failure_diagnostic.json')
    print('Complete:',a.output,flush=True)


if __name__=='__main__':
    main()
