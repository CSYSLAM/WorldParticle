"""Interleaved no-retraining physics-step comparisons from an accepted own-model state.

By default this measures solver latency. --frame-loop includes the original
controller and recording work during a held-state window. Neither measures
whole-task rendered FPS. Input retains its exact accepted own-model history.
"""
import argparse
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=('franka','cloth','w1'),default='franka')
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'model_releases/shared_vbd_v24_cloth40/model.pt')
    p.add_argument('--baseline-source',type=Path,help='Pre-change source snapshot for the legacy planner ablation')
    p.add_argument('--repeats',type=int,default=4)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--frame-loop',action='store_true',help='Franka held frames: include the original IK controller, joint readout and in-memory frame recording')
    p.add_argument('--material-ablation',action='store_true',help='Algorithm comparison: reference=A2 learned, all variants use fused_graph; numerical equivalence is diagnostic only')
    p.add_argument('--variants',nargs='+',choices=('reference','cache','assembly','cache_assembly','compiled','fused','fused_search','reference_atomic','fused_atomic','reference_no_fill','fused_no_fill','fused_fast','fused_fast_legacy_plan','fused_device','fused_device_uncached_topology','fused_device_native_products','fused_graph','fused_graph_legacy','material_analytic','material_omit','prefix_one_learned'),
                   default=['reference','assembly','cache_assembly'])
    p.add_argument('--profile',action='store_true')
    p.add_argument('--scalar-profile',action='store_true',help='Extra CPU operator/stack pass to locate scalar reads and dynamic selections; never used for timing claims')
    p.add_argument('--nvtx',action='store_true',help='Extra annotated pass for Nsight; excludes Python call profiling unless --profile is also set')
    p.add_argument('--nvtx-all-variants',action='store_true',help='Annotate every requested variant in the extra diagnostic pass')
    p.add_argument('--cuda-profiler',action='store_true',help='Start/stop CUDA profiler around the extra NVTX pass; use Nsight capture-range=cudaProfilerApi')
    a=p.parse_args()
    if a.cuda_profiler:a.nvtx=True
    if a.nvtx_all_variants:a.nvtx=True
    if a.repeats<1 or a.steps<1:p.error('Positive repeats and steps required')
    if a.frame_loop and (a.case!='franka' or a.steps%3):
        p.error('--frame-loop requires Franka and a multiple of three physical steps')
    if a.variants[0]!='reference' or len(set(a.variants))!=len(a.variants):
        p.error('Unique variants starting with reference required')
    material_variants={'reference','material_analytic','material_omit','prefix_one_learned'}
    if a.material_ablation and not set(a.variants)<=material_variants:
        p.error('Material ablation requires explicit material/prefix variants')
    if not a.material_ablation and set(a.variants)&(material_variants-{'reference'}):
        p.error('Material/prefix variants require --material-ablation')
    if 'fused_fast_legacy_plan' in a.variants and a.baseline_source is None:p.error('Legacy planner requires --baseline-source')
    a.output.mkdir(parents=True,exist_ok=False)
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda('cuda:0',ROOT)
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd.adaptive_step import AdaptiveStepConfig
    from wp_vbd.data import State
    from wp_vbd import hierarchical_response
    current_plan=hierarchical_response.build_plan;legacy_plan=None
    if a.baseline_source is not None:
        import importlib.util
        path=a.baseline_source/'src/wp_vbd/heterogeneous_backbone.py'
        spec=importlib.util.spec_from_file_location('wp_vbd._benchmark_legacy_planner',path)
        module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
        legacy_plan=module.build_plan
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cuda:0')
    wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    run=json.loads((a.recording/'run.json').read_text())
    if run['status']!='COMPLETE' or run['mode']!='shared_network_inference' or any(run['forbidden_solver_calls'].values()):
        raise ValueError('Complete independently inferred model state required')
    import hashlib
    implementation={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(__file__).resolve(),*sorted((ROOT/'src/wp_vbd').glob('*.py'))]}
    environment={'torch':torch.__version__,'torch_cuda':torch.version.cuda,
        'device':torch.cuda.get_device_name(0),'newton':run.get('newton_source')}
    if a.baseline_source is not None:
        environment['legacy_planner_source_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    digest=hashlib.file_digest(a.checkpoint.open('rb'),'sha256').hexdigest()
    if digest!=run['checkpoint_sha256']:raise ValueError('Weights must match accepted-state source')
    cfg=run.get('solver_configuration')
    if cfg is None and a.case=='w1':
        bound=inspect.signature(newton.solvers.SolverVBD).bind(None,iterations=10,
            integrate_with_external_rigid_solver=False,particle_enable_self_contact=False,
            rigid_joint_linear_ke=1e6,rigid_joint_angular_ke=1e6,
            rigid_joint_linear_kd=0.,rigid_joint_angular_kd=0.)
        bound.apply_defaults()
        cfg={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    if cfg is None:raise ValueError('Recording lacks source export configuration')
    # Asset reconstruction may inspect the original constructor signature.
    def forbidden(*args,**kwargs):raise AssertionError('No native dynamics in benchmark')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        original=cls.__init__
        import functools
        cls.__init__=functools.wraps(original)(forbidden);cls.step=forbidden
    if a.case=='franka':
        from wp_vbd.franka_vbd_scene import build
        p=run['physical_parameters']
        e=build(wp,newton,device='cuda:0',substeps=3,material_mu=p['tet_material'][0],
            material_lambda=p['tet_material'][1],arm_drive_ke=p['arm_drive_ke'],grasp_x=p['grasp_x'],
            grasp_yaw=p['grasp_yaw'],grip_closed=p['grip_closed'],grip_open=p['grip_open'],
            grasp_height=p['grasp_height'])
    elif a.case=='cloth':
        from wp_vbd.franka_cloth_corner_scene import build
        e=build(wp,newton,device='cuda:0',substeps=3,fold_profile=run['fold_profile'])
    else:
        from wp_vbd.w1_bag_scene import build
        e=build(wp,newton,device='cuda:0',asset=Path(run['asset']),output=a.output/'asset',
                substeps=3,friction=run['friction'],bag_profile=run['bag_profile'],bag_grip=run['bag_grip'])
    stored=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=dict(stored['model_config']);config.update(mimic_metric_tangent=True,periodic_bending=True)
    config['contact_aware_warm_start']=False if a.case=='w1' else True
    if a.case=='w1':config['line_search']=[2.**(-i) for i in range(13)]
    model=MixedSolver(config).cuda().eval();model.load_state_dict(stored['model'],strict=True);del stored
    if a.material_ablation and model.shared_material_head is None:raise ValueError('Trained material head required for algorithm comparison')
    scene=export_model(e.model,cfg,run['scene'],newton,1/180,'cuda:0')
    scene.metadata.update(detection_motion_budget=.012,bounded_motion_forecast_target_only=True,
        joint_tree_coarse_modes='scalar_tree_twelve_columns_v307',articulation_material_motion_modes=a.case!='w1',
        independent_object_motion_modes=a.case!='franka',soft_tangent_support_scale=512. if a.case=='w1' else 64. if a.case=='cloth' else 8.,
        force_aware_material_anticipation=True,compact_numerical_diagnostics=True,
        repair_native_mimic_roundoff=True,surface_tangent_memory=True,surface_tangent_history_capacity=8192,
        record_grasp_forces=a.case!='w1')
    if a.case=='w1':scene.metadata.update(joint_tree_extra_columns=8,separate_coarse_ray_step=True,
        rigid_tangent_support_scale=64.,repair_metric_ray_roundoff=True,elastic_shape_response=True)
    solver=SurfaceJointExampleSolver(model,scene,1/180,wp,newton,e.model,e.state_0)
    solver.enable_adaptive_steps(AdaptiveStepConfig(maximum_dt=1/180,minimum_dt=1/600,
        maximum_attempts=256,maximum_substeps=128))
    if a.case=='cloth':
        saved=torch.load(a.recording/'final_runtime_state.pt',map_location='cuda:0',weights_only=True)
        state=State(*(saved['state'][k] for k in ('x','q','v','omega')),saved['history'])
        solver._write_native_output(e.state_0,state);initial_history=state.history
        with np.load(a.recording/'trajectory.npz') as z:e.control.joint_target_q.assign(z['motor_target'][-1])
    else:
        with np.load(a.recording/'final_runtime_state.npz') as z:
            for name in ('particle_q','particle_qd','body_q','body_qd'):getattr(e.state_0,name).assign(z[name])
            e.control.joint_target_q.assign(z['joint_target_q'])
            initial_history={k[len('history__'):]:torch.as_tensor(z[k].copy(),device='cuda:0') for k in z.files if k.startswith('history__')}
            if a.frame_loop:
                initial_ik=z['ik_joint_q'].copy()
                initial_targets=z['joint_target_q'].copy()
    if a.frame_loop:
        from wp_vbd.franka_vbd_scene import controller
        q_readout=wp.empty_like(e.model.joint_q);qd_readout=wp.empty_like(e.model.joint_qd)
    initial_native={k:wp.to_torch(getattr(e.state_0,k)).clone() for k in ('particle_q','particle_qd','body_q','body_qd')}
    initial_history={k:v.clone() for k,v in initial_history.items()}
    command_targets=[];controller_calls=[0]
    def reset(variant):
        if a.material_ablation:
            model.config.update(analytic_predictor_rounds=1 if variant=='prefix_one_learned' else 2,
                shared_material_use_learned=variant!='material_analytic',
                omit_final_material_correction=variant=='material_omit')
            model._shared_material_calls=0;model._analytic_material_total_solves=0;model._omitted_material_calls=0
            variant='fused_graph'
        kinematics_requested=variant=='fused_graph'
        model.config['fused_physical_checks']=variant=='fused_graph'
        graph_requested=variant in ('fused_graph','fused_graph_legacy')
        if graph_requested:variant='fused_device'
        model.config['capture_cuda_graphs']=graph_requested
        hierarchical_response.build_plan=legacy_plan if variant=='fused_fast_legacy_plan' else current_plan
        if variant=='fused_fast_legacy_plan':variant='fused_fast'
        spatial_products=variant!='fused_device_native_products'
        if variant=='fused_device_native_products':variant='fused_device'
        cache_topology=variant=='fused_device'
        if variant=='fused_device_uncached_topology':variant='fused_device'
        torch.use_deterministic_algorithms(variant not in ('reference_atomic','fused_atomic'))
        torch.utils.deterministic.fill_uninitialized_memory=variant not in ('reference_no_fill','fused_no_fill')
        for name,value in initial_native.items():wp.to_torch(getattr(e.state_0,name)).copy_(value)
        solver.history={k:v.clone() for k,v in initial_history.items()}
        solver.physical_time=run['physical_seconds'];solver.previous_dt=1/180
        command_targets.clear();controller_calls[0]=0
        if a.frame_loop:
            e.helper.ik_joint_q.assign(initial_ik)
            e.control.joint_target_q.assign(initial_targets)
        model.config['inference_kinematics_cache']=kinematics_requested or variant in ('cache','cache_assembly')
        model.config['fast_inference_assembly']=variant in ('assembly','cache_assembly')
        if variant in ('compiled','fused','fused_search','fused_atomic','fused_no_fill','fused_fast','fused_device'):model.config['fast_inference_assembly']=True
        model.config['compiled_contact_rows']=variant in ('compiled','fused','fused_search','fused_atomic','fused_no_fill','fused_fast','fused_device')
        model.config['compiled_kinematics']=variant in ('fused','fused_search','fused_atomic','fused_no_fill','fused_fast','fused_device')
        model.config['compiled_candidate_kernels']=variant in ('fused_search','fused_atomic','fused_no_fill','fused_fast','fused_device')
        model.config['compiled_response_kernels']=variant=='fused_device'
        model.config['response_topology_cache']=cache_topology
        model.config['compile_small_spatial_products']=spatial_products and variant=='fused_device'
        model.config['inference_fill_uninitialized_memory']=False if variant in ('fused_fast','fused_device') else None
        model.config['inference_deterministic_algorithms']=False if variant in ('fused_fast','fused_device') else None
        from wp_vbd.inference_policy import apply_scene
        apply_scene(scene,model.config)
        solver.adaptive_retries=0;solver.steps=0
    def advance(steps=None):
        for index in range(a.steps if steps is None else steps):
            if a.frame_loop and index%3==0:
                controller(e,solver.physical_time,wp)
                controller_calls[0]+=1
                command_targets.append(wp.to_torch(e.control.joint_target_q).clone())
            e.state_0.clear_forces();e.state_1.clear_forces()
            solver.step(e.state_0,e.state_1,e.control,None,1/180)
            e.state_0,e.state_1=e.state_1,e.state_0
            if a.frame_loop and (index+1)%3==0:
                # Match the Franka recorder, discarding buffers after the
                # frame. Video rendering and final file compression are separate.
                e.state_0.particle_q.numpy().copy();e.state_0.body_q.numpy().copy()
                newton.eval_ik(e.model,e.state_0,q_readout,qd_readout)
                q_readout.numpy().copy();e.control.joint_target_q.numpy().copy()
                e.state_0.particle_qd.numpy().copy();e.state_0.body_qd.numpy().copy()
    rows=[];reference=None;reference_targets=None
    def write():
        summary={v:{'median_seconds':statistics.median(r['seconds'] for r in rows if r['variant']==v),
            'samples':sum(r['variant']==v for r in rows)} for v in a.variants if any(r['variant']==v for r in rows)}
        if 'reference' in summary:
            for v in summary:summary[v]['speedup_vs_reference']=summary['reference']['median_seconds']/summary[v]['median_seconds']
        if a.frame_loop:
            for v in summary:summary[v]['held_frame_fps']=(a.steps/3)/summary[v]['median_seconds']
        (a.output/'benchmark.json').write_text(json.dumps({'status':'RUNNING','case':a.case,'source':str(a.recording.resolve()),
            'weights_sha256':digest,'implementation_sha256':implementation,'environment':environment,
            'steps_per_sample':a.steps,'nominal_substeps':3,'frame_loop':a.frame_loop,
            'algorithmic_material_ablation':a.material_ablation,
            'quality_scope':'Full-task quality requires separate complete rollout audits' if a.material_ablation else 'Same algorithm execution comparison',
            'controller_cadence':'once_per_three_substeps' if a.frame_loop else 'not_included',
            'samples':rows,'summary':summary,
            'scope':('Interleaved held-state frame loop including original IK controller, native joint readout and in-memory recording. No video rendering, file compression or whole-task FPS claim. '
                if a.frame_loop else 'Interleaved solver-only latency at an accepted own-model final state; no controller, rendering or trajectory serialization. ')
                +'Other GPU tasks are not stopped.'},indent=2)+'\n')
    hook_calls=[0]
    def count(*unused):hook_calls[0]+=1
    hook=model.protected_response.hierarchy.register_forward_hook(count)
    warmups=[]
    with torch.no_grad():
        for variant in a.variants:
            reset(variant);torch.cuda.synchronize();begin=time.perf_counter();advance();torch.cuda.synchronize()
            warmups.append({'variant':variant,'seconds_including_lazy_compilation':time.perf_counter()-begin})
            (a.output/'warmup.json').write_text(json.dumps(warmups,indent=2)+'\n')
        for repeat in range(a.repeats):
            order=list(a.variants)
            if repeat%2:order.reverse()
            for variant in order:
                reset(variant);hook_calls[0]=0;torch.cuda.synchronize();begin=time.perf_counter();advance();torch.cuda.synchronize()
                elapsed=time.perf_counter()-begin
                values={k:wp.to_torch(getattr(e.state_0,k)).cpu().numpy().copy() for k in initial_native}
                control_error=None
                if a.frame_loop:
                    controls=torch.stack(command_targets).cpu().numpy().copy()
                    if reference_targets is None:reference_targets=controls
                    control_error=float(np.max(abs(controls-reference_targets),initial=0.))
                if reference is None:
                    if variant!='reference':raise ValueError('Reference must be first')
                    reference=values
                errors={k:float(np.max(abs(v-reference[k]),initial=0.)) for k,v in values.items()}
                rows.append({'repeat':repeat,'variant':variant,'seconds':elapsed,'neural_calls':hook_calls[0],
                    'analytic_predictor_rounds':model.config.get('analytic_predictor_rounds'),
                    'material_omitted':model.config.get('omit_final_material_correction',False),
                    'material_calls':dict(learned=getattr(model,'_shared_material_calls',0),
                        analytic=getattr(model,'_analytic_material_total_solves',0),
                        omitted=getattr(model,'_omitted_material_calls',0)) if a.material_ablation else None,
                    'deterministic_algorithms':torch.are_deterministic_algorithms_enabled(),
                    'fill_uninitialized_memory':torch.utils.deterministic.fill_uninitialized_memory,
                    'physics_fill_uninitialized_memory':model.config.get('inference_fill_uninitialized_memory'),
                    'physics_deterministic_algorithms':model.config.get('inference_deterministic_algorithms'),
                    'controller_target_max_abs_error':control_error,'controller_calls':controller_calls[0],
                    'fused_physical_checks':model.config.get('fused_physical_checks',False),
                    'accepted_physical_steps':solver.steps,'retries':solver.adaptive_retries,'errors_vs_reference':errors,
                    'kinematic_cache':getattr(model,'last_kinematics_cache',None) if model.config.get('inference_kinematics_cache') else None,
                    'response_index_cache':(scene._response_index_workspace.report()
                        if model.config.get('response_topology_cache') and hasattr(scene,'_response_index_workspace') else None)})
                write();print(json.dumps(rows[-1]),flush=True)
    hook.remove()
    result=json.loads((a.output/'benchmark.json').read_text());result['status']='COMPLETE'
    baseline=next(r for r in rows if r['variant']=='reference')
    checks={'finite_timings':all(np.isfinite(r['seconds']) and r['seconds']>0 for r in rows),
        'same_neural_budget':all(r['neural_calls']==baseline['neural_calls'] for r in rows),
        'same_physical_steps':all(r['accepted_physical_steps']==baseline['accepted_physical_steps'] for r in rows),
        'same_controller_targets':all(r['controller_target_max_abs_error'] in (None,0.) for r in rows),
        'correct_controller_cadence':all(r['controller_calls']==(a.steps//3 if a.frame_loop else 0) for r in rows),
        'no_retries':all(r['retries']==0 for r in rows),
        'short_window_numerically_equivalent':all(all(error<=(1e-6 if key.endswith('qd') else 1e-8)
            for key,error in r['errors_vs_reference'].items()) for r in rows)}
    if a.material_ablation:
        budgets={'reference':(1,2,0),'material_analytic':(0,3,0),
                 'material_omit':(0,2,1),'prefix_one_learned':(1,1,0)}
        checks['correct_material_stage_budgets']=all(
            tuple(r['material_calls'][key] for key in ('learned','analytic','omitted'))==
            tuple(a.steps*count for count in budgets[r['variant']]) for r in rows)
    result['checks']=checks
    required={k:v for k,v in checks.items() if not (a.material_ablation and k=='short_window_numerically_equivalent')}
    result['required_checks']=required
    if not all(required.values()):result['status']='FAILED_COMPARISON'
    elif a.material_ablation:result['status']='COMPLETE_ALGORITHMIC_ABLATION'
    if a.profile or a.nvtx:
        from wp_vbd.execution_profile import ExecutionProfile
        if a.cuda_profiler:torch.cuda.profiler.start()
        if a.nvtx:torch.cuda.nvtx.range_push('wp.measurement')
        try:
            for variant in (a.variants if a.nvtx_all_variants else dict.fromkeys(('reference',a.variants[-1]))):
                reset(variant);profile=ExecutionProfile(model.protected_response,'cuda:0',a.output/(variant+'_profile.json'),
                    python_calls=a.profile,nvtx=a.nvtx,prefix=variant+'.' if a.nvtx else '')
                torch.cuda.synchronize()
                if a.nvtx:torch.cuda.nvtx.range_push(variant+'.wp.complete_window')
                try:
                    with torch.no_grad():
                        profile.start()
                        try:
                            with profile.measure('wp.solver_step'):advance()
                        finally:profile.finish()
                finally:
                    if a.nvtx:torch.cuda.nvtx.range_pop()
        finally:
            if a.nvtx:torch.cuda.nvtx.range_pop()
            if a.cuda_profiler:torch.cuda.profiler.stop()
    if a.scalar_profile:
        from collections import defaultdict
        for variant in (a.variants[-1],):
            reset(variant)
            with torch.no_grad(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU],
                    with_stack=True,experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as trace:
                advance(steps=1)
            grouped=defaultdict(lambda:{'calls':0,'self_cpu_microseconds':0.})
            for event in trace.events():
                if event.name not in ('aten::_local_scalar_dense','aten::nonzero','aten::_to_copy'):continue
                stack=tuple(frame for frame in event.stack if 'wp_vbd/' in frame or '/scripts/' in frame)
                key=(event.name,stack)
                grouped[key]['calls']+=1;grouped[key]['self_cpu_microseconds']+=event.self_cpu_time_total
            rows=[dict(operator=name,stack=list(stack),**value) for (name,stack),value in grouped.items()]
            rows.sort(key=lambda row:row['calls'],reverse=True)
            (a.output/(variant+'_scalar_reads.json')).write_text(json.dumps({
                'rows':rows,'physical_steps':1,'scope':'Extra one-physical-step CPU operator/stack diagnostic, bounded to reduce profiler memory. Counts identify eager scalar reads, dynamic selections and copies. Profiler overhead invalidates latency claims; _to_copy also includes dtype/device-local copies.'},indent=2)+'\n')
    (a.output/'benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    if result['status'] not in ('COMPLETE','COMPLETE_ALGORITHMIC_ABLATION'):raise SystemExit(1)


if __name__=='__main__':main()
