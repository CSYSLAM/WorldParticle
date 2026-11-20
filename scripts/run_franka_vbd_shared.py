"""Run a dynamic, pure-VBD Franka/soft-body scene with the shared network.

--reference is a separate offline VBD baseline. Model mode forbids both
VBD and Featherstone execution and consumes no reference trajectory.
"""
import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import source_root, source_record
NEWTON_ROOT = source_root()
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT/'model_releases/shared_vbd_v24_cloth40/model.pt')
    p.add_argument('--device', choices=('cuda:0','cpu'), default='cuda:0')
    p.add_argument('--gpu-memory-gib',type=float,default=12.,help='CUDA process budget including a two GiB native allocation reserve')
    p.add_argument('--frames', type=int, default=240)
    p.add_argument('--substeps', type=int, help='Defaults to 3 for model inference, 10 for the offline VBD reference')
    p.add_argument('--request-dt',type=float,help='Model-only single physical step per output frame, e.g. 1/60 or 1/30 as a decimal')
    p.add_argument('--disable-learning',action='store_true',help='Use the same physical-round budget with learned responses disabled')
    p.add_argument('--contact-aware-warm-start',action=argparse.BooleanOptionalAction,default=True,
                   help='Keep the initial free-flight candidate from closing static particle contacts; inertial targets are unchanged')
    p.add_argument('--analytic-elastic-postsolve',action=argparse.BooleanOptionalAction,default=None,
                   help='Explicit experiment: one reduced material correction after each response')
    p.add_argument('--analytic-predictor-rounds',type=int,choices=(0,1,2,3),
                   help='Explicit analytic budget override; neural response count is unchanged')
    p.add_argument('--auto-retry-rejected-steps',action=argparse.BooleanOptionalAction,default=True,
                   help='Rollback a rejected physical proposal, subdivide the requested interval and retry')
    p.add_argument('--start-time', type=float, default=0., help='Independently initialize at an authored command pose; never restore teacher state')
    p.add_argument('--release-after',type=float,help='Smoothly open the hand after this authored time; preserve approach/pinch/lift controls before it')
    p.add_argument('--release-duration',type=float,default=.15,help='Smooth gripper opening duration for a release teacher')
    p.add_argument('--resume-recording',type=Path,help='Continue an accepted rigid/cloth model recording, retaining its complete saved frame prefix')
    p.add_argument('--place-height',type=float,default=.29,help='Rigid/cloth tool drive target during release; actual pose remains dynamic')
    p.add_argument('--motion-budget',type=float,default=.003,help='Detection envelope in meters; exceeding it rejects the step')
    p.add_argument('--material-mu',type=float);p.add_argument('--material-lambda',type=float)
    p.add_argument('--arm-drive-ke',type=float,default=2000.)
    p.add_argument('--friction',type=float,help='Override both shape and soft friction; zero is a diagnostic ablation')
    p.add_argument('--joint-linear-ke',type=float,default=1e5)
    p.add_argument('--joint-angular-ke',type=float,default=1e5)
    p.add_argument('--reference-iterations',type=int,default=10,help='Offline VBD baseline only; model still uses its checkpoint unroll')
    p.add_argument('--reference-body-particle-buffer',type=int,default=256)
    p.add_argument('--reference-rigid-buffer',type=int,default=64,help='Offline VBD per-body rigid contact adjacency capacity')
    p.add_argument('--reference-mimic-relaxation',type=float,default=1.,help='Explicit offline native-VBD mimic projection relaxation; 1 uses stock Newton')
    p.add_argument('--reference-mimic-final-only',action='store_true',help='Offline numerical experiment: native mimic projection once at the end of each VBD iteration budget')
    p.add_argument('--reference-audit-substeps',action='store_true',help='Inspect material orientation and mimic error at every native physical step')
    p.add_argument('--reference-record-substeps',action='store_true',help='Save native velocities, actual controls and source assets at every physical step; enables the substep audit')
    p.add_argument('--contact-mode',choices=('full','rigid_only','none'),default='full',help='Diagnostic isolation; only full is a grasp test')
    p.add_argument('--articulation-modes',action='store_true',help='Experimental bounded joint-tree physical response with unchanged network weights')
    p.add_argument('--mimic-metric-tangent',action=argparse.BooleanOptionalAction,default=True,help='Match both mimic response projections to the frozen local inverse')
    p.add_argument('--terminal-velocity',action=argparse.BooleanOptionalAction,default=True,help='Use the saved learned terminal velocity correction')
    p.add_argument('--native-mimic-roundoff-repair',action=argparse.BooleanOptionalAction,default=True,
                   help='Repair only bounded Float32 native-state mimic quantization, preserving strict joint tolerances')
    p.add_argument('--separate-coarse-step',action='store_true',help='Experimental separate curvature scaling for protected and complementary proposals')
    p.add_argument('--material-motion-modes',action='store_true',help='Experimental independent material motion in a single-material anchored-tree interaction')
    p.add_argument('--tangent-support-scale',type=float,default=1.,help='Numerical tangential response scale; friction coefficient and Coulomb force cap stay physical')
    p.add_argument('--rigid-tangent-support-scale',type=float,default=1.,help='Numerical rigid-contact tangential support; Coulomb cap is unchanged')
    p.add_argument('--force-aware-anticipation',action='store_true',help='Use total material force when forecasting contact opening/activation')
    p.add_argument('--learned-contact-memory',action=argparse.BooleanOptionalAction,default=False,help='Use accepted weighted-surface tangent memory and a learned Coulomb-bounded tangent seed')
    p.add_argument('--surface-tangent-history',action=argparse.BooleanOptionalAction,default=True,
                   help='Carry accepted matched surface tangential multipliers without enabling the learned seed')
    p.add_argument('--contact-seed-scale',type=float,default=.1,
                   help='Bounded learned tangent seed fraction of physical Coulomb capacity (0 to .25)')
    p.add_argument('--elastic-local-heads',type=int,default=0,choices=(0,1,2),help='Learned spatially localized elastic response columns')
    p.add_argument('--grasp-x',type=float,default=-.005)
    p.add_argument('--grasp-height',type=float,default=.21,help='Tool height during pinch; align the finger pads to the material')
    p.add_argument('--grasp-yaw',type=float,default=0.,help='Tool yaw in radians')
    p.add_argument('--grip-open',type=float,default=.04);p.add_argument('--grip-closed',type=float,default=.02)
    p.add_argument('--soft-asset', choices=('block','duck','rigid_cloth'), default='block')
    p.add_argument('--reference', action='store_true')
    p.add_argument('--window', action='store_true')
    p.add_argument('--diagnose', action='store_true', help='Record the first physical-step line searches without changing them')
    p.add_argument('--neural-graphs',action='store_true',help='Replay pure attention blocks on CUDA; physical steps and geometry remain eager')
    p.add_argument('--neural-precision',choices=('fp32','bf16'),default='fp32',help='Optional inference arithmetic inside the neural response only')
    p.add_argument('--compact-diagnostics',action='store_true',help='Omit per-component ray-scaling lists; preserve all numerical checks and acceptance rules')
    p.add_argument('--profile',action='store_true',help='Bounded CPU call statistics and neural/solver stream intervals; not a speed benchmark')
    p.add_argument('--exhaustive-line-search',action='store_true',help='Diagnostic reference: evaluate smaller rays even after all groups accept')
    a = p.parse_args()
    if a.substeps is None:a.substeps=10 if a.reference else 3
    if a.soft_asset=='rigid_cloth' and (a.material_mu is not None or a.material_lambda is not None or a.grasp_yaw!=0.):
        p.error('Rigid/cloth scenes use authored cloth coefficients and a fixed tool orientation')
    if not math.isfinite(a.place_height) or not .2<=a.place_height<=.335:p.error('Place height must be in [0.2, 0.335] m')
    if not math.isfinite(a.contact_seed_scale) or not 0<=a.contact_seed_scale<=.25:
        p.error('--contact-seed-scale must be in [0, .25]')
    if a.resume_recording and (a.soft_asset!='rigid_cloth' or a.reference or a.start_time!=0.):
        p.error('Model-frame continuation requires rigid/cloth model mode with the original zero command origin')
    newton_provenance = source_record()
    if a.frames<1 or a.substeps<1: p.error('Positive frames/substeps required')
    if a.request_dt is not None and (a.reference or not math.isfinite(a.request_dt) or a.request_dt<=0):
        p.error('Positive finite --request-dt is available in model mode')
    if a.release_after is not None and (not math.isfinite(a.release_after)
            or not math.isfinite(a.release_duration) or a.release_after<a.start_time or a.release_duration<=0):
        p.error('--release-after requires a finite event after start_time and a positive duration')
    if a.release_after is None and a.release_duration!=.15:p.error('--release-duration requires --release-after')
    if a.disable_learning and a.reference:p.error('--disable-learning is available in model mode')
    if a.request_dt is not None and a.resume_recording:p.error('Macro-step recordings require a fresh independent run')
    if not math.isfinite(a.gpu_memory_gib) or a.gpu_memory_gib<=2:
        p.error('GPU budget must be finite and exceed the two GiB native reserve')
    if not 0<=a.start_time<=4 or not 0<a.motion_budget<float('inf'):
        p.error('Start time must be between zero and four seconds, with a finite positive motion budget')
    if any(value is not None and not 0<value<float('inf') for value in (a.material_mu,a.material_lambda,a.arm_drive_ke,a.joint_linear_ke,a.joint_angular_ke)):
        p.error('Material and drive stiffness must be finite and positive')
    if a.friction is not None and not 0<=a.friction<float('inf'):p.error('Friction must be finite and nonnegative')
    if min(a.reference_iterations,a.reference_body_particle_buffer,a.reference_rigid_buffer)<1:p.error('Positive reference iteration/buffer budget required')
    if not math.isfinite(a.reference_mimic_relaxation) or not 0<a.reference_mimic_relaxation<=1:
        p.error('Reference mimic relaxation must be in (0, 1]')
    if a.reference and (a.articulation_modes or a.force_aware_anticipation):
        p.error('Experimental response/anticipation flags apply only to shared model inference')
    if a.reference and (a.neural_graphs or a.neural_precision!='fp32' or a.compact_diagnostics):
        p.error('Neural arithmetic options apply only to model inference')
    if a.neural_graphs and not a.device.startswith('cuda'):
        p.error('Neural graph replay requires CUDA')
    if a.separate_coarse_step and (a.reference or not a.articulation_modes):p.error('Separate subspace scaling requires model mode with --articulation-modes')
    if a.material_motion_modes and (a.reference or not a.articulation_modes):p.error('Material motion modes require model mode with --articulation-modes')
    if not 0<a.tangent_support_scale<float('inf') or a.reference and a.tangent_support_scale!=1.:
        p.error('Finite positive tangential scale is a model-only numerical setting')
    if not 0<a.rigid_tangent_support_scale<float('inf') or a.reference and a.rigid_tangent_support_scale!=1.:
        p.error('Finite positive rigid tangential scale is a model-only numerical setting')
    if not all(math.isfinite(value) for value in (a.grasp_x,a.grasp_yaw,a.grasp_height,a.grip_open,a.grip_closed)) or not 0<=a.grip_closed<=a.grip_open<=.04:
        p.error('Finite grasp targets and 0 <= closed <= open <= 0.04 m are required')
    if not a.reference and (a.reference_iterations!=10 or a.reference_body_particle_buffer!=256 or a.reference_rigid_buffer!=64 or a.reference_mimic_relaxation!=1. or a.reference_mimic_final_only or a.reference_audit_substeps or a.reference_record_substeps):
        p.error('Reference-only solver budgets are unavailable in model mode')
    if a.reference_record_substeps:a.reference_audit_substeps=True
    a.output.mkdir(parents=True, exist_ok=True)
    report_path = a.output/'run.json'
    if report_path.exists(): raise FileExistsError(report_path)
    from wp_vbd.gpu_ownership import claim_cuda
    ownership = claim_cuda(a.device, ROOT)
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    # GitPython is needed even for cached Newton assets. Keep root binary
    # packages first, appending only the existing Newton environment.
    sys.path.append(str(NEWTON_ROOT/'.venv/lib/python3.12/site-packages'))
    from wp_vbd.franka_vbd_scene import build, controller
    if a.soft_asset=='rigid_cloth':
        if a.reference:p.error('The rigid/cloth recording entry is model-only')
        from wp_vbd.franka_rigid_cloth_scene import build, controller
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.gpu_budget import GpuBudget
    torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
    budget = GpuBudget(a.device, a.gpu_memory_gib)
    wp.config.kernel_cache_dir = str(ROOT/'.local/p0/warp-cache')
    wp.init(); wp.set_device(a.device)
    if a.device.startswith('cuda'): wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    original_vbd = newton.solvers.SolverVBD
    overrides = dict(iterations=a.reference_iterations, integrate_with_external_rigid_solver=False, particle_enable_self_contact=False,
                     rigid_body_contact_buffer_size=a.reference_rigid_buffer,
                     rigid_body_particle_contact_buffer_size=a.reference_body_particle_buffer,
                     rigid_joint_linear_ke=a.joint_linear_ke, rigid_joint_angular_ke=a.joint_angular_ke,
                     rigid_joint_linear_kd=0., rigid_joint_angular_kd=0.)
    bound = inspect.signature(original_vbd).bind(None, **overrides); bound.apply_defaults()
    configuration = {k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    calls = {'vbd_construct':0, 'vbd_step':0, 'featherstone_construct':0, 'featherstone_step':0}
    def forbidden(kind):
        def fail(*unused, **kwargs):
            calls[kind] += 1
            raise AssertionError('Forbidden solver in shared-network inference: '+kind)
        return fail
    newton.solvers.SolverFeatherstone.__init__ = forbidden('featherstone_construct')
    newton.solvers.SolverFeatherstone.step = forbidden('featherstone_step')
    if not a.reference:
        original_vbd.__init__ = forbidden('vbd_construct'); original_vbd.step = forbidden('vbd_step')
    begin = time.perf_counter(); e = build(wp, newton, device=a.device, soft_asset=a.soft_asset, substeps=a.substeps, start_time=a.start_time,
                                           material_mu=a.material_mu,material_lambda=a.material_lambda,arm_drive_ke=a.arm_drive_ke,
                                           grasp_x=a.grasp_x,grasp_yaw=a.grasp_yaw,grip_open=a.grip_open,grip_closed=a.grip_closed,friction=a.friction,
                                           grasp_height=a.grasp_height,
                                           **({'place_height':a.place_height} if a.soft_asset=='rigid_cloth' else {}))
    if a.request_dt is not None:
        e.sim_dt=e.frame_dt=a.request_dt;e.sim_substeps=1
    if a.contact_mode!='full':
        flags=e.model.shape_flags.numpy()
        mask=int(newton.ShapeFlags.COLLIDE_PARTICLES)
        if a.contact_mode=='none':mask|=int(newton.ShapeFlags.COLLIDE_SHAPES)
        e.model.shape_flags.assign(flags & ~mask)
    source_files=sorted((ROOT/'src/wp_vbd').glob('*.py'))
    implementation={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),*source_files]}
    for relative,expected in implementation.items():
        payload=(ROOT/relative).read_bytes()
        if hashlib.sha256(payload).hexdigest()!=expected:raise RuntimeError('Implementation changed while freezing this run')
        target=a.output/'implementation_snapshot'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(payload)
    (a.output/'implementation_snapshot_manifest.json').write_text(json.dumps(implementation,indent=2)+'\n')
    checkpoint_sha = None; checkpoint_step = None; reference_adaptation = None; precision = None; resume = None
    from wp_vbd.neural_cuda_graph import enable_attention_replay,attention_replay_report
    if a.reference:
        if a.reference_mimic_relaxation!=1. or a.reference_mimic_final_only:
            from wp_vbd.reference_mimic_relaxation import ReferenceMimicRelaxation
            reference_adaptation=ReferenceMimicRelaxation(a.reference_mimic_relaxation,
                projection_interval=a.reference_iterations if a.reference_mimic_final_only else 1).install()
        pipeline = newton.CollisionPipeline(e.model, deterministic=True, enable_rigid_soft_full_surface_contact=True)
        contacts = pipeline.contacts(); solver = original_vbd(e.model, **overrides)
        scene = None; network = None
    else:
        checkpoint_sha = hashlib.sha256(a.checkpoint.read_bytes()).hexdigest()
        stored = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
        checkpoint_step = stored['step']
        model_config = stored['model_config'] if 'model_config' in stored else stored['config']['model']
        saved_local_heads=model_config.get('elastic_shape_local_heads',0)
        if a.elastic_local_heads>max(2,saved_local_heads):
            p.error('Runtime local head count may not exceed the saved two-head architecture')
        runtime_config={**model_config,'elastic_shape_local_heads':max(saved_local_heads,a.elastic_local_heads),
                        'learned_contact_tangent_seed':bool(model_config.get('learned_contact_tangent_seed',False)
                            or a.learned_contact_memory)}
        if a.analytic_elastic_postsolve is not None:runtime_config['analytic_elastic_postsolve']=a.analytic_elastic_postsolve
        if a.analytic_predictor_rounds is not None:runtime_config['analytic_predictor_rounds']=a.analytic_predictor_rounds
        if a.mimic_metric_tangent is not None:runtime_config['mimic_metric_tangent']=a.mimic_metric_tangent
        runtime_config['contact_aware_warm_start']=a.contact_aware_warm_start
        network = MixedSolver(runtime_config).to(a.device).eval()
        missing,unexpected=network.load_state_dict(stored['model'],strict=False)
        allowed_missing=set()
        for prefix,module in (('elastic_shape_head',network.elastic_shape_head),
                              ('contact_tangent_head',network.contact_tangent_head)):
            keys={prefix+'.'+name for name in module.state_dict()} if module is not None else set()
            absent=keys & set(missing)
            if prefix=='elastic_shape_head' and absent:
                new={key for key in keys if '.local_network.' in key}
                if absent==new:allowed_missing|=new
                elif keys.issubset(missing):allowed_missing|=keys
                else:raise ValueError('Incomplete checkpoint elastic shape head')
            elif prefix=='contact_tangent_head' and absent:
                if keys.issubset(missing):allowed_missing|=keys
                else:raise ValueError('Incomplete checkpoint contact tangent head')
        if set(missing)!=allowed_missing or unexpected:
            raise ValueError(f'Unexpected runtime checkpoint migration: missing={missing}, unexpected={unexpected}')
        if not a.terminal_velocity:network.terminal_velocity=None
        if network.elastic_shape_head is not None:network.elastic_shape_head.local_heads=a.elastic_local_heads
        network.config['elastic_shape_local_heads']=a.elastic_local_heads
        network.config['learned_contact_tangent_seed']=bool(a.learned_contact_memory)
        network.config['contact_tangent_seed_scale']=a.contact_seed_scale
        if a.resume_recording:
            previous=json.loads((a.resume_recording/'run.json').read_text())
            if previous['status']=='RUNNING':raise ValueError('A live model recording cannot be restored')
            if previous['mode']!='shared_network_inference' or previous['scene']!='research.franka_vbd_rigid_cloth' or previous['initial_command_time']!=0.:
                raise ValueError('A model-produced rigid/cloth prefix with the same command origin is required')
            if previous['checkpoint_sha256']!=checkpoint_sha or previous['newton_source']!=newton_provenance or previous['physical_dt']!=e.sim_dt:
                raise ValueError('Model weights, Newton source and physical timestep must match the prefix')
            if any(previous['forbidden_solver_calls'].values()) or previous['teacher_trajectory_reads']:
                raise ValueError('Teacher/native-solver recordings are not model continuation inputs')
            if (previous.get('mimic_metric_tangent',False)!=runtime_config.get('mimic_metric_tangent',False) or
                    previous.get('contact_aware_warm_start',False)!=a.contact_aware_warm_start or
                    previous.get('surface_tangent_history',False)!=bool(a.surface_tangent_history or a.learned_contact_memory) or
                    previous.get('native_mimic_roundoff_repair',False)!=a.native_mimic_roundoff_repair):
                raise ValueError('Continuation must preserve the recorded projection/contact policy; use explicit legacy flags or start a fresh recording')
            expected={'arm_drive_ke':a.arm_drive_ke,'joint_linear_ke':a.joint_linear_ke,'joint_angular_ke':a.joint_angular_ke,
                      'friction_override':a.friction,'grip_open':a.grip_open,'grip_closed':a.grip_closed,'grasp_height':a.grasp_height}
            if any(previous['physical_parameters'][key]!=value for key,value in expected.items()):
                raise ValueError('Model continuation cannot change physical parameters')
            if (previous['articulation_modes']!=a.articulation_modes or previous['separate_coarse_step']!=a.separate_coarse_step or
                previous['solver_parameters']['soft_tangent_support_scale']!=a.tangent_support_scale or
                previous['solver_parameters'].get('rigid_tangent_support_scale',1.)!=a.rigid_tangent_support_scale or
                previous['solver_parameters']['force_aware_material_anticipation']!=a.force_aware_anticipation):
                raise ValueError('Model continuation requires the same physical response settings')
            with np.load(a.resume_recording/'trajectory.npz') as archive:
                tape={key:archive[key].copy() for key in archive.files}
            count=previous['frames_completed']
            if not 0<count<a.frames or len(tape['physical_time'])<count+1 or not np.allclose(tape['physical_time'][:count+1],np.arange(count+1)*e.frame_dt,rtol=0,atol=1e-10):
                raise ValueError('A nonempty, frame-aligned accepted prefix shorter than the requested timeline is required')
            # Failure diagnostics may append an accepted partial frame.
            # Restore the last complete frame and replay the incomplete
            # interval from its own model state, never from teacher data.
            for key in ('physical_time','particle_position','particle_velocity','body_pose','body_velocity','joint_position','joint_target'):
                tape[key]=tape[key][:count+1]
            if not np.array_equal(tape['particle_rest'],e.model.particle_q.numpy()) or not np.array_equal(tape['faces'],e.model.tri_indices.numpy()):
                raise ValueError('Rest assets/topology must match the model recording')
            for field,key in (('particle_q','particle_position'),('particle_qd','particle_velocity'),('body_q','body_pose'),('body_qd','body_velocity')):
                value=tape[key][-1]
                if not np.isfinite(value).all():raise ValueError('Nonfinite model continuation state')
                getattr(e.state_0,field).assign(value)
            wp.to_torch(e.helper.ik_joint_q)[0,:9].copy_(torch.as_tensor(tape['joint_target'][-1,:9],device=a.device))
            e.control.joint_target_q.assign(tape['joint_target'][-1])
            resume={'report':previous,'tape':tape,'frames':count,
                    'provenance':{'recording':str(a.resume_recording.resolve()),
                      'trajectory_sha256':hashlib.sha256((a.resume_recording/'trajectory.npz').read_bytes()).hexdigest(),
                      'source_manifest_sha256':hashlib.sha256((a.resume_recording/'implementation_snapshot_manifest.json').read_bytes()).hexdigest(),
                      'accepted_frame_prefix':count,'accepted_physical_seconds':count*e.frame_dt,
                      'ignored_incomplete_substeps':previous['physical_substeps']-count*a.substeps,
                      'teacher_trajectory_reads':0,
                      'history_policy':'Static pins retained; joint winding reinitialized from this accepted model pose within the original branch. No contact dual history persists across physical steps.',
                      'controller_place_height_m':a.place_height}}
        if a.neural_graphs:enable_attention_replay(network)
        from wp_vbd.neural_precision import NeuralPrecision
        precision=NeuralPrecision(network.protected_response,a.neural_precision,a.device)
        scene = export_model(e.model, configuration, 'research.franka_vbd_'+a.soft_asset, newton, e.sim_dt, a.device)
        scene.metadata['detection_motion_budget']=a.motion_budget
        scene.metadata['evaluate_all_line_search_rays']=a.exhaustive_line_search
        if a.compact_diagnostics:scene.metadata['compact_numerical_diagnostics']=True
        scene.metadata['repair_native_mimic_roundoff']=a.native_mimic_roundoff_repair
        scene.metadata['bounded_motion_forecast_target_only']=True
        if a.articulation_modes:scene.metadata['joint_tree_coarse_modes']='scalar_tree_twelve_columns_v307'
        if a.separate_coarse_step:scene.metadata['separate_coarse_ray_step']=True
        if a.material_motion_modes:scene.metadata['articulation_material_motion_modes']=True
        if a.soft_asset=='rigid_cloth':
            if not a.articulation_modes:p.error('Rigid/cloth scenes require --articulation-modes')
            scene.metadata['independent_object_motion_modes']=True
        scene.metadata['soft_tangent_support_scale']=a.tangent_support_scale
        scene.metadata['rigid_tangent_support_scale']=a.rigid_tangent_support_scale
        scene.metadata['record_grasp_forces']=True
        if a.force_aware_anticipation:scene.metadata['force_aware_material_anticipation']=True
        if a.learned_contact_memory or a.surface_tangent_history:
            scene.metadata['surface_tangent_memory']=True
            scene.metadata['surface_tangent_history_capacity']=8192
        network.config['learned_contact_tangent_seed']=bool(a.learned_contact_memory)
        network.config['elastic_shape_local_heads']=a.elastic_local_heads
        solver = SurfaceJointExampleSolver(network, scene, e.sim_dt, wp, newton, e.model, e.state_0)
        solver.learned=not a.disable_learning
        if a.auto_retry_rejected_steps and e.sim_dt>=1/600-1e-12:
            from wp_vbd.adaptive_step import AdaptiveStepConfig
            solver.enable_adaptive_steps(AdaptiveStepConfig(maximum_dt=e.sim_dt,
                minimum_dt=1/600,maximum_attempts=256,
                maximum_substeps=128,motion_safety=.8,relative_update_limit=.15))
        elif a.auto_retry_rejected_steps:
            scene.metadata['adaptive_retry_bypassed_below_floor']=True
        if resume:
            if bool((scene.rest_x[scene.fixed_particle_ids]-solver.read(e.state_0).x[0,scene.fixed_particle_ids]).abs().max()>1e-8):
                raise ValueError('Model-frame continuation admits unchanged static cloth clamps only')
            solver.history['pin_initialized'].fill_(True)
            solver.steps=resume['frames']*a.substeps
            if solver.adaptive_stepper is not None:
                solver.physical_time=resume['frames']*e.frame_dt
    if a.diagnose and network:
        from wp_vbd import rigid_contact_solver as algorithm, surface_joint_system as physics
        from wp_vbd.mimic_constraint import rows as mimic_rows
        choose_original=algorithm.choose;retract_original=physics.retract;merit_original=physics.merit
        probes=[]
        def inspected_choose(*args,**kwargs):
            if solver.steps:return choose_original(*args,**kwargs)
            row={'search':len(probes),'evaluations':[],'retractions':[]}
            if len(probes)==1:
                from wp_vbd.data import State
                from wp_vbd import math3d as so3
                s,current,start,update,tx,tq,context,frozen=args[:8]
                with torch.enable_grad():
                    delta=torch.zeros_like(update,requires_grad=True)
                    trial=State(current.x+delta[...,:3],so3.multiply(so3.exp(delta[...,3:]),current.q),current.v,current.omega)
                    gradient,=torch.autograd.grad(merit_original(s,trial,start,tx,tq,context,frozen).sum(),delta)
                assembled,metric,_=physics.system(s,current,start,tx,tq,context)
                row['merit_gradient_residual_max_error']=float((gradient*s.dof-assembled).abs().max())
                row['merit_gradient_dot_update']=float((gradient*update).sum())
                row['assembled_dot_update']=float((assembled*update).sum())
                from wp_vbd.protected_response import metric_action
                row['actual_direction_metric_curvature']=float((update*metric_action(metric,update)).sum())
            def inspected_merit(*arguments,**options):
                value=merit_original(*arguments,**options)
                n=e.model.particle_count
                evaluated=physics.joints.joint_rows(arguments[0],arguments[1],arguments[5])
                row['evaluations'].append({'robot':float(value[:,n:n+e.model.body_count].sum()),'soft':float(value[:,:n].sum()),
                                          'joint_energy':evaluated['energy'].tolist()})
                return value
            def inspected_retract(s,state):
                before=mimic_rows(s,state).value
                result,valid=retract_original(s,state)
                row['retractions'].append({'valid':valid.tolist(),'before_mimic':before.tolist(),'after_mimic':mimic_rows(s,result).value.tolist()})
                return result,valid
            physics.merit=inspected_merit;physics.retract=inspected_retract
            try:return choose_original(*args,**kwargs)
            finally:
                physics.merit=merit_original;physics.retract=retract_original;probes.append(row)
                (a.output/'line_search_diagnostic.json').write_text(json.dumps(probes,indent=2)+'\n')
        algorithm.choose=inspected_choose
    print(json.dumps({'event':'initialized','particle_count':e.model.particle_count, 'tet_count':e.model.tet_count,
                      'body_count':e.model.body_count,'joint_count':e.model.joint_count,
                      'mimic_count':int((e.model.joint_mimic_joint.numpy()>=0).sum()),
                      'sdf_shapes':int((e.model._shape_sdf_index.numpy()>=0).sum()),
                      'device':a.device,'physics_device':str(e.model.device),'dt':e.sim_dt,
                      'initialization_seconds':time.perf_counter()-begin}), flush=True)
    viewer = newton.viewer.ViewerGL(width=1600,height=900,vsync=False) if a.window else None
    if viewer:
        viewer.set_model(e.model); viewer.set_camera(wp.vec3(.65,-1.15,.80),-22.,140.)
    particle=[]; body=[]; times=[]; phases=[]; joint_positions=[];joint_targets=[];steps=0; frames=0; failure=None
    particle_velocities=[];body_velocities=[];input_targets=[];input_rates=[];input_efforts=[]
    static_sha=None
    if a.reference_record_substeps:
        from wp_vbd.mixed_contact_data import STATIC_FIELDS
        from wp_vbd.surface_joint_data import JOINT_FIELDS
        static={}
        for name in dict.fromkeys((*STATIC_FIELDS,*JOINT_FIELDS,'joint_mimic_coeffs','joint_target_mode')):
            value=getattr(e.model,name,None)
            if value is not None:static['source__'+name]=value.numpy().copy()
        for index,item in enumerate(e.model.shape_source):
            for name in ('vertices','indices','points'):
                value=getattr(item,name,None)
                if value is None:continue
                value=value.numpy() if hasattr(value,'numpy') else np.asarray(value)
                if not value.dtype.hasobject:static[f'source_geometry{index}__{name}']=value.copy()
        np.savez_compressed(a.output/'source_static.npz',**static)
        static_sha=hashlib.sha256((a.output/'source_static.npz').read_bytes()).hexdigest()
    native_audit={'enabled':a.reference_audit_substeps,'steps':0,'min_tet_volume_ratio':None,
                  'max_tet_volume_ratio':None,'nonpositive_tet_steps':0,'first_nonpositive_step':None,
                  'max_mimic_coordinate_error':0.}
    if a.reference_audit_substeps:
        tet_ids=wp.to_torch(e.model.tet_indices).long()
        rest=wp.to_torch(e.model.particle_q).double()[tet_ids]
        rest_determinant=torch.linalg.det(rest[:,1:]-rest[:,:1])
        followers=torch.where(wp.to_torch(e.model.joint_mimic_joint)>=0)[0]
        references=wp.to_torch(e.model.joint_mimic_joint)[followers].long()
        qstarts=wp.to_torch(e.model.joint_q_start).long()
        mimic_coefficients=wp.to_torch(e.model.joint_mimic_coeffs)[followers].double()
    q_readout=wp.empty_like(e.model.joint_q);qd_readout=wp.empty_like(e.model.joint_qd)
    force_peaks={'upward_friction_n':0.,'coulomb_capacity_n':0.,'coulomb_ratio':0.}
    max_contacts=resume['report']['max_contact_counts'].copy() if resume else {'rigid':0,'soft':0}
    nn_calls=resume['report']['nn_backbone_calls'] if resume else 0
    if network:
        def count_nn(*unused):
            nonlocal nn_calls
            nn_calls+=1
        hook=network.protected_response.hierarchy.register_forward_hook(count_nn)
    def record():
        particle.append(e.state_0.particle_q.numpy().copy())
        physical_time=(solver.physical_time if network and solver.adaptive_stepper is not None else steps*e.sim_dt)
        body.append(e.state_0.body_q.numpy().copy()); times.append(physical_time)
        newton.eval_ik(e.model,e.state_0,q_readout,qd_readout)
        joint_positions.append(q_readout.numpy().copy());joint_targets.append(e.control.joint_target_q.numpy().copy())
        particle_velocities.append(e.state_0.particle_qd.numpy().copy())
        body_velocities.append(e.state_0.body_qd.numpy().copy())
    if resume:
        tape=resume['tape'];frames=resume['frames'];steps=frames*a.substeps
        for target,key in ((particle,'particle_position'),(body,'body_pose'),(times,'physical_time'),
                           (particle_velocities,'particle_velocity'),(body_velocities,'body_velocity'),
                           (joint_positions,'joint_position'),(joint_targets,'joint_target')):
            target.extend(tape[key])
        phases.extend(tape['phase'][:frames])
    else:record()
    initial_particle=particle[0].copy(); initial_body=body[0].copy()
    run_start=time.perf_counter()
    def write(status):
        out={'status':status,'mode':'offline_vbd_reference' if a.reference else 'shared_network_inference',
             'scene':'research.franka_vbd_'+a.soft_asset,'source_robot':'fr3_franka_hand.urdf',
             'official_unchanged_demo':False,'robot_dynamics':'single VBD baseline' if a.reference else 'shared neural/physical response',
             'contact_mode':a.contact_mode,'is_grasp_test':a.contact_mode=='full',
             'articulation_modes':a.articulation_modes if network else False,
             'mimic_metric_tangent':network.config.get('mimic_metric_tangent',False) if network else False,
             'contact_aware_warm_start':network.config.get('contact_aware_warm_start',False) if network else False,
             'terminal_velocity_enabled':network.terminal_velocity is not None if network else False,
             'native_mimic_roundoff_repair':a.native_mimic_roundoff_repair if network else False,
             'surface_tangent_history':bool(scene.metadata.get('surface_tangent_memory',False)) if network else False,
             'native_roundoff_projection_count':getattr(solver,'native_roundoff_projection_count',0),
             'maximum_native_position_correction_m':getattr(solver,'maximum_native_position_correction',0.),
             'separate_coarse_step':a.separate_coarse_step,
             'material_motion_modes':a.material_motion_modes,
             'solver_parameters':{'soft_tangent_support_scale':a.tangent_support_scale,
                  'rigid_tangent_support_scale':a.rigid_tangent_support_scale,
                  'contact_tangent_seed_scale':a.contact_seed_scale if a.learned_contact_memory else 0.,
                  'automatic_rejected_step_recovery':bool(network and solver.adaptive_stepper is not None),
                  'adaptive_max_dt_s':e.sim_dt if network and solver.adaptive_stepper is not None else None,
                  'adaptive_min_dt_s':solver.adaptive_stepper.config.minimum_dt if network and solver.adaptive_stepper is not None else None,
                  'adaptive_max_substeps_per_requested_interval':solver.adaptive_stepper.config.maximum_substeps if network and solver.adaptive_stepper is not None else None,
                  'evaluate_all_line_search_rays':a.exhaustive_line_search,
                  'force_aware_material_anticipation':a.force_aware_anticipation,
                  'model_unroll':network.config.get('unroll') if network else None,
                  'offline_reference_iterations':a.reference_iterations if a.reference else None,
                  'offline_reference_mimic_relaxation':a.reference_mimic_relaxation if a.reference else None,
                  'offline_reference_mimic_schedule':('final_iteration' if a.reference_mimic_final_only else 'every_iteration') if a.reference else None,
                  'reference_numerical_adaptation':'explicit_native_mimic_projection_policy' if reference_adaptation else None,
                  'offline_reference_buffers':{'rigid_per_body':a.reference_rigid_buffer,
                      'soft_per_body':a.reference_body_particle_buffer} if a.reference else None},
             'contact_force_peaks':force_peaks,
             'neural_graph_replay':attention_replay_report(network) if network else None,
             'neural_precision':precision.report() if precision else None,
             'compact_numerical_diagnostics':a.compact_diagnostics,
             'checkpoint':str(a.checkpoint) if network else None,'checkpoint_sha256':checkpoint_sha,'checkpoint_step':checkpoint_step,
             'model_architecture':network.architecture_id if network else None,
             'analytic_predictor_rounds':network.config.get('analytic_predictor_rounds',0) if network else 0,
             'analytic_elastic_postsolve':bool(network.config.get('analytic_elastic_postsolve',False)) if network else False,
             'device':a.device,'physics_device':str(e.model.device),'physical_dt':e.sim_dt,'frame_dt':e.frame_dt,
             'learning_enabled':not a.disable_learning,'requested_single_step_dt':a.request_dt,
             'recording_stride_substeps':1 if a.reference_record_substeps else e.sim_substeps,
             'source_static_sha256':static_sha,'native_substep_audit':native_audit if a.reference else None,
             'solver_configuration':configuration,
             'initial_command_time':a.start_time,'independent_undeformed_soft_initialization':True,
             'teacher_release':None if a.release_after is None else {'start_time_s':a.release_after,'duration_s':a.release_duration,
                 'opening':'smoothstep from authored closed to open drive target'},
             'model_recording_continuation':resume['provenance'] if resume else None,
             'requested_window_completed':frames==a.frames,
             'full_authored_timeline_recorded':frames==a.frames and a.start_time==0 and steps*e.sim_dt>=(6.4 if a.soft_asset=='rigid_cloth' else 4.)-1e-9,
             'payload_position':body[-1][e.payload_id,:3].tolist() if a.soft_asset=='rigid_cloth' and body else None,
             'detection_motion_budget':a.motion_budget if network else None,
             'frames_completed':frames,'frames_requested':a.frames,
             'physical_substeps':solver.steps if network and solver.adaptive_stepper is not None else steps,
             'physical_seconds':solver.physical_time if network and solver.adaptive_stepper is not None else steps*e.sim_dt,
             'adaptive_requested_intervals':getattr(solver,'adaptive_requested_intervals',0) if network else 0,
             'adaptive_accepted_substeps':getattr(solver,'adaptive_accepted_substeps',0) if network else 0,
             'adaptive_retries':getattr(solver,'adaptive_retries',0) if network else 0,
             'adaptive_recovery_modes':getattr(solver,'adaptive_recovery_counts',{}) if network else {},
             'adaptive_solver_attempts':getattr(solver,'adaptive_solver_attempts',0) if network else 0,
             'adaptive_recovery_attempts':getattr(solver,'adaptive_recovery_attempts',0) if network else 0,
             'adaptive_response_rounds':getattr(solver,'adaptive_network_rounds',0) if network else 0,
             'adaptive_min_accepted_dt_s':getattr(solver,'adaptive_min_accepted_dt',None) if network else None,
             'adaptive_max_accepted_dt_s':getattr(solver,'adaptive_max_accepted_dt',None) if network else None,
             'adaptive_counter_scope':'Committed requested intervals; failed-interval work is in last_diagnostic.adaptive.attempts. nn_backbone_calls and wall_seconds include all work.',
             'particle_count':e.model.particle_count,'tet_count':e.model.tet_count,'body_count':e.model.body_count,'joint_count':e.model.joint_count,
             'ideal_mimic_count':int((e.model.joint_mimic_joint.numpy()>=0).sum()),
             'forbidden_solver_calls':calls,'teacher_trajectory_reads':0,'new_optimizer_updates':0,
             'reference_vbd_steps':steps if a.reference else 0,
             'adapted_mimic_projection_calls':reference_adaptation.calls if reference_adaptation else 0,
             'adapted_mimic_projection_opportunities':reference_adaptation.opportunities if reference_adaptation else 0,
             'initial_joint_positions':e.model.joint_q.numpy().tolist(),
             'physical_parameters':{'tet_material':e.model.tet_materials.numpy()[0].tolist() if e.model.tet_count else None,
                  'cloth_triangle_material':e.model.tri_materials.numpy()[0].tolist() if a.soft_asset=='rigid_cloth' else None,
                  'payload_body_id':getattr(e,'payload_id',None),
                  'arm_drive_ke':a.arm_drive_ke,'soft_contact_ke':e.model.soft_contact_ke,'soft_contact_kd':e.model.soft_contact_kd,
                  'joint_linear_ke':a.joint_linear_ke,'joint_angular_ke':a.joint_angular_ke,
                  'grasp_x':a.grasp_x,'grasp_yaw':a.grasp_yaw,'grasp_height':a.grasp_height,'grip_open':a.grip_open,'grip_closed':a.grip_closed,
                  'friction_override':a.friction,
                  'soft_contact_mu':e.model.soft_contact_mu,'shape_friction':e.model.shape_material_mu.numpy().tolist()},
             'implementation_sha256':implementation,'newton_source':newton_provenance,
             'max_contact_counts':max_contacts,'nn_backbone_calls':nn_calls,'failure':failure,
             'wall_seconds':time.perf_counter()-run_start+(resume['report']['wall_seconds'] if resume else 0.),'gpu_budget':budget.report(),
             'scope':'New dynamic grasp test; original URDF/inertia/collision geometry, native per-configuration mesh force filtering, low-resolution elastic block or original duck. IK produces drive targets only. No trained-demo or realtime claim.'}
        if network and solver.last_diagnostic:
            out['last_diagnostic']={k:(v.detach().cpu().tolist() if isinstance(v,torch.Tensor) else v) for k,v in solver.last_diagnostic.items() if k not in ('iterations','rejected_physical_step','contact_repair_supervision')}
            out['articulation_coarse']=getattr(scene,'articulation_coarse_diagnostic',None)
        if particle:
            out['soft_center_lift_m']=float((particle[-1]-initial_particle).mean(0)[2])
            out['body_origin_max_motion_m']=float(np.linalg.norm(body[-1][:,:3]-initial_body[:,:3],axis=-1).max())
            tet=e.model.tet_indices.numpy() if e.model.tet_count else np.empty((0,4),dtype=np.int32)
            if len(tet):
                def determinants(position):
                    points=position[tet]
                    return np.linalg.det((points[:,1:]-points[:,:1]).astype(np.float64))
                jacobian=determinants(particle[-1])/determinants(initial_particle)
                out['final_tet_volume_ratio_min']=float(jacobian.min())
                out['final_tet_volume_ratio_max']=float(jacobian.max())
            from wp_vbd.franka_vbd_scene import command
            tool=body[-1][e.helper.endeffector_id].astype(np.float64)
            q=tool[3:]/np.linalg.norm(tool[3:]);offset=np.array([0.,0.,.22])
            point=tool[:3]+offset+2*np.cross(q[:3],np.cross(q[:3],offset)+q[3]*offset)
            target_z,_,_=command(a.start_time+max(0,frames-1)*e.frame_dt,grasp_height=a.grasp_height)
            target=np.array([a.grasp_x,-.5,target_z])
            if a.soft_asset=='rigid_cloth':
                from wp_vbd.franka_rigid_cloth_scene import command as place_command
                target,_,_=place_command(a.start_time+max(0,frames-1)*e.frame_dt,
                                         grasp_height=a.grasp_height,grip_open=a.grip_open,grip_closed=a.grip_closed,place_height=a.place_height)
                payload=np.asarray(body)[:,e.payload_id,:3]
                out['payload_initial_position']=payload[0].tolist()
                out['payload_position']=payload[-1].tolist()
                out['payload_max_lift_m']=float(payload[:,2].max()-payload[0,2])
                out['payload_lateral_travel_m']=float(payload[-1,0]-payload[0,0])
                out['physical_parameters']['grasp_x']=-.12
                out['physical_parameters']['cloth_clamps']='four boundary edges'
                out['physical_parameters']['cloth_height_m']=.235
                out['physical_parameters']['place_x_m']=.14
                out['physical_parameters']['place_height_m']=a.place_height
                out['physical_parameters']['payload_size_m']=[.05,.05,.05]
                out['physical_parameters']['payload_mass_kg']=float(e.model.body_mass.numpy()[e.payload_id])
                out['scope']='Authored FR3/free rigid/cloth scene. Shared network and physical response on the selected device; IK controls robot drives only. No payload attachment or teacher trajectory.'
            out.update(actual_tool_position=point.tolist(),commanded_tool_position=target.tolist(),
                       tool_position_error_m=float(np.linalg.norm(point-target)))
        temporary=report_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(out,indent=2,default=str)+'\n')
        temporary.replace(report_path)
    write('RUNNING')
    profiler=None
    if a.profile:
        from wp_vbd.execution_profile import ExecutionProfile
        profiler=ExecutionProfile(network.protected_response if network else None,a.device,a.output/'operator_profile.json')
        profiler.start()
    try:
        with torch.no_grad():
            for frame in range(frames,a.frames):
                phase=controller(e,a.start_time+frame*e.frame_dt+(.5*e.frame_dt if a.request_dt else 0.),wp)
                if a.release_after is not None and a.start_time+frame*e.frame_dt>=a.release_after:
                    now=a.start_time+frame*e.frame_dt
                    blend=float(np.clip((now-a.release_after)/a.release_duration,0.,1.))
                    blend=blend*blend*(3.-2.*blend)
                    target=float(a.grip_closed+(a.grip_open-a.grip_closed)*blend)
                    wp.to_torch(e.control.joint_target_q)[-2:]=target
                    if now>=a.release_after:phase='unload' if blend<1 else 'recovery'
                if not a.reference_record_substeps:phases.append(phase)
                for sub in range(e.sim_substeps):
                    e.state_0.clear_forces(); e.state_1.clear_forces()
                    if a.reference_record_substeps:
                        input_targets.append(e.control.joint_target_q.numpy().copy())
                        input_rates.append(e.control.joint_target_qd.numpy().copy())
                        input_efforts.append(e.control.joint_f.numpy().copy())
                    if a.reference:
                        pipeline.collide(e.state_0,contacts)
                        solver.step(e.state_0,e.state_1,e.control,contacts,e.sim_dt)
                        for name in ('particle_q','particle_qd','body_q','body_qd'):
                            if not bool(torch.isfinite(wp.to_torch(getattr(e.state_1,name))).all()):
                                raise FloatingPointError(f'Offline VBD baseline produced nonfinite {name} at substep {steps}')
                    else:
                        with profiler.measure('wp.solver_step') if profiler else nullcontext():
                            solver.step(e.state_0,e.state_1,e.control,None,e.sim_dt)
                        precision.verify_physical_boundary()
                        c=scene.candidates
                        d=solver.last_diagnostic
                        if 'dynamic_soft_tangent_force_world' in d:
                            force_peaks['upward_friction_n']=max(force_peaks['upward_friction_n'],d['dynamic_soft_tangent_force_world'][0][2])
                            force_peaks['coulomb_capacity_n']=max(force_peaks['coulomb_capacity_n'],d['dynamic_soft_coulomb_capacity_sum'][0])
                            force_peaks['coulomb_ratio']=max(force_peaks['coulomb_ratio'],d['maximum_dynamic_soft_coulomb_ratio'])
                        max_contacts['rigid']=max(max_contacts['rigid'],c.rigid_count)
                        max_contacts['soft']=max(max_contacts['soft'],len(c.soft.shape))
                    e.state_0,e.state_1=e.state_1,e.state_0; steps+=1
                    if a.reference_record_substeps:phases.append(phase);record()
                    if a.reference_audit_substeps:
                        position=wp.to_torch(e.state_0.particle_q).double()[tet_ids]
                        ratio=torch.linalg.det(position[:,1:]-position[:,:1])/rest_determinant
                        lower=float(ratio.min());upper=float(ratio.max())
                        native_audit['steps']+=1
                        native_audit['min_tet_volume_ratio']=lower if native_audit['min_tet_volume_ratio'] is None else min(lower,native_audit['min_tet_volume_ratio'])
                        native_audit['max_tet_volume_ratio']=upper if native_audit['max_tet_volume_ratio'] is None else max(upper,native_audit['max_tet_volume_ratio'])
                        if lower<=0:
                            native_audit['nonpositive_tet_steps']+=1
                            if native_audit['first_nonpositive_step'] is None:native_audit['first_nonpositive_step']=steps
                        if len(followers):
                            if not a.reference_record_substeps:newton.eval_ik(e.model,e.state_0,q_readout,qd_readout)
                            qvalue=wp.to_torch(q_readout).double()
                            error=qvalue[qstarts[followers]]-mimic_coefficients[:,0]-mimic_coefficients[:,1]*qvalue[qstarts[references]]
                            native_audit['max_mimic_coordinate_error']=max(native_audit['max_mimic_coordinate_error'],float(error.abs().max()))
                frames+=1
                if not a.reference_record_substeps:record()
                budget.observe()
                write('RUNNING')
                if viewer:
                    viewer.begin_frame(steps*e.sim_dt);viewer.log_state(e.state_0);viewer.end_frame()
                    if not viewer.is_running():break
                if frames%10==0 or frames==1:
                    write('RUNNING');print(json.dumps({'event':'progress','frames':frames,'physical_seconds':steps*e.sim_dt,
                                                     'phase':phase,
                                                     'adaptive_substeps':getattr(solver,'steps',steps) if network and solver.adaptive_stepper is not None else steps,
                                                     'adaptive_retries':getattr(solver,'adaptive_retries',0) if network and solver.adaptive_stepper is not None else 0,
                                                     'wall_seconds':time.perf_counter()-run_start}),flush=True)
    except Exception:
        failure=traceback.format_exc(); print(failure,flush=True)
        if scene is not None and hasattr(scene,'candidates'):
            max_contacts['rigid']=max(max_contacts['rigid'],scene.candidates.rigid_count)
            max_contacts['soft']=max(max_contacts['soft'],len(scene.candidates.soft.shape))
        if steps*e.sim_dt>times[-1]+1e-12: record()
    finally:
        if profiler:profiler.finish()
        budget.observe(force=True)
        if viewer:viewer.close()
        np.savez_compressed(a.output/'trajectory.npz',particle_position=np.asarray(particle),body_pose=np.asarray(body),
                            physical_time=np.asarray(times),phase=np.asarray(phases),faces=e.model.tri_indices.numpy(),
                            joint_position=np.asarray(joint_positions),joint_target=np.asarray(joint_targets),
                            particle_velocity=np.asarray(particle_velocities),body_velocity=np.asarray(body_velocities),
                            input_joint_target_raw=np.asarray(input_targets),input_joint_rate_raw=np.asarray(input_rates),
                            input_joint_effort_raw=np.asarray(input_efforts),
                            tets=e.model.tet_indices.numpy() if e.model.tet_count else np.empty((0,4),dtype=np.int32),particle_rest=initial_particle)
        if network:
            final={name:getattr(e.state_0,name).numpy().copy() for name in ('particle_q','particle_qd','body_q','body_qd')}
            final.update({'history__'+name:value.detach().cpu().numpy() for name,value in solver.history.items()})
            final['ik_joint_q']=e.helper.ik_joint_q.numpy().copy()
            final['joint_target_q']=e.control.joint_target_q.numpy().copy()
            np.savez_compressed(a.output/'final_runtime_state.npz',**final)
        write('FAILED' if failure else 'COMPLETE' if frames==a.frames else 'INTERRUPTED')
        if precision:precision.close()
        if reference_adaptation:reference_adaptation.restore()
        print(report_path,flush=True)
    if failure:sys.exit(1)


if __name__=='__main__':main()
