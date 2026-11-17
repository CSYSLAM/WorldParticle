"""Train one full648-wide response on qualified ordinary VBD demo data.

The course follows the user's shared-GPU budget and current pure-VBD scope;
training coverage comes from changed parameters in selected real updates.
"""
import argparse
import copy
import functools
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import activate_newton_source
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def write(path,data):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,default=ROOT/'reports/shared_normal_648_v81/data_manifest.json')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--updates',type=int,default=200)
    parser.add_argument('--validation-interval',type=int,default=50)
    parser.add_argument('--learning-rate',type=float,default=1e-5)
    parser.add_argument('--gpu-memory-gib',type=float,default=12.)
    parser.add_argument('--duty-cycle',type=float,default=.25)
    parser.add_argument('--resume-from',type=Path)
    parser.add_argument('--resume-report',type=Path)
    parser.add_argument('--resume-recorded-joint',action='store_true',help='Continue a stopped recorded joint-course checkpoint, preserving its full sampling and validation history')
    parser.add_argument('--optimizer-schedule',type=Path,help='Hash-bound declaration for an isolated learning-rate change during recorded continuation')
    parser.add_argument('--gradient-normalization',choices=('clip','unit'),default='clip')
    parser.add_argument('--diagnostic-plan',type=Path,help='Isolated GPU gradient/update comparison using the identical training window runtime; never writes training weights or counts')
    parser.add_argument('--extend-from',type=Path,help='Continue an eligible selected648 prefix from a stopped course with the same model/Adam')
    parser.add_argument('--extend-report',type=Path)
    parser.add_argument('--checkpoint-neural-encoder',action='store_true')
    parser.add_argument('--host-mixed-trajectories',action='store_true',help='Keep complete mixed reference trajectories on CPU and transfer only the supervised two-step window')
    parser.add_argument('--host-self-trajectories',action='store_true',help='Keep complete self-contact trajectories on CPU and transfer all states, inputs and accepted history for the supervised two-step window')
    parser.add_argument('--compact-host-reference-constants',action='store_true',help='Store only byte-identical complete time rows once for read-only self-contact host references')
    parser.add_argument('--trim-host-allocator',action='store_true',help='Reclaim only unused glibc pages at load/query/forward boundaries')
    parser.add_argument('--host-rod-trajectories',action='store_true',help='Keep complete rod/contact state and accepted matching history on CPU; transfer each complete supervised window')
    parser.add_argument('--host-encoder-inputs',action='store_true',help='Stage exact encoder checkpoint inputs on CPU until recomputation')
    parser.add_argument('--checkpoint-relation-ffn-rows',type=int,default=0,help='Bound pure relation FFN execution with nested row checkpoints')
    parser.add_argument('--checkpoint-attention-blocks',action='store_true',help='Use nested checkpoints for all pure hierarchy attention blocks')
    parser.add_argument('--contact-metric-ray-step',action='store_true')
    parser.add_argument('--elastic-prestress-metric',action='store_true')
    parser.add_argument('--compact-frozen-contact-metric',action='store_true')
    parser.add_argument('--checkpoint-contact-rows',action='store_true')
    parser.add_argument('--contact-activation-ray-step',action='store_true')
    parser.add_argument('--contact-motion-ray-step',action='store_true')
    parser.add_argument('--contact-structural-support',action='store_true')
    parser.add_argument('--caller-pose-history',action='store_true')
    parser.add_argument('--elastic-rod-self-contact',action='store_true')
    parser.add_argument('--refresh-self-geometry',action='store_true')
    parser.add_argument('--elastic-rod-current-geometry',action='store_true',help='Refresh composed elastic/ROD self geometry while preserving legacy self-demo policies')
    parser.add_argument('--elastic-rod-differentiated-residual',action='store_true',help='Differentiate composed material/contact residuals while preserving legacy backward policies')
    parser.add_argument('--host-step-saved-tensors',action='store_true')
    parser.add_argument('--learned-contact-repair',action='store_true')
    parser.add_argument('--development-only',action='store_true',help='Verify the effective runtime on all course development demos without optimizer calls; save a separate pretraining report')
    parser.add_argument('--joint-course',action='store_true',help='Explicit combined course: retain selected baseline, use bound CUDA admissions, and defer full development to the declared final interval')
    parser.add_argument('--loss-monitor-interval',type=int,default=0,help='Monitor fixed authored and representative official windows without gradients or old32 comparisons')
    parser.add_argument('--self-broad-phase',choices=('chunked','kdtree'),help='Complete self-contact candidate query; default kdtree on CPU and chunked on CUDA')
    args=parser.parse_args()
    activate_newton_source()
    if args.updates<1 or args.validation_interval<1 or not 0<args.duty_cycle<=1:parser.error('Positive update budgets and duty cycle in (0,1] required')
    if args.gpu_memory_gib<=2:parser.error('Reserve two GiB for native geometry and device libraries')
    if args.checkpoint_relation_ffn_rows<0:parser.error('Nonnegative relation FFN chunk size required')
    if args.loss_monitor_interval<0:parser.error('Nonnegative loss monitor interval required')
    if args.joint_course and (not (args.extend_from or args.resume_recorded_joint) or not args.loss_monitor_interval):parser.error('A joint course requires a bound original parent or recorded continuation, and fixed-window loss monitoring')
    if args.resume_recorded_joint and (not args.joint_course or not args.resume_from or not args.optimizer_schedule):parser.error('Recorded joint continuation requires its checkpoint/report and declared optimizer schedule')
    if args.optimizer_schedule and not args.resume_recorded_joint:parser.error('Optimizer scheduling requires recorded joint continuation')
    if args.gradient_normalization!='clip' and not args.resume_recorded_joint:parser.error('Sample gradient balancing requires a declared recorded continuation')
    if args.compact_host_reference_constants and not args.host_self_trajectories:parser.error('Constant reference sharing requires --host-self-trajectories')
    if bool(args.resume_from)!=bool(args.resume_report):parser.error('Resume checkpoint and producer report are required together')
    if bool(args.extend_from)!=bool(args.extend_report):parser.error('Selected extension checkpoint and stopped producer report are required together')
    if args.resume_from and args.extend_from:parser.error('Choose last-call recovery or selected-prefix extension')
    directory=args.manifest.parent;directory.mkdir(parents=True,exist_ok=True)
    if (directory/'training.json').exists():raise FileExistsError('Keep earlier trial evidence immutable')
    from wp_vbd.training_scope import load_scope,validate_rows,changed_weight_coverage,CUSTOM_PURE_VBD
    manifest=json.loads(args.manifest.read_text());rows=manifest['scenes']
    from wp_vbd.research_course import sampling_rows
    if manifest.get('research_sampling') and not manifest.get('research_initialization'):
        parser.error('Added-data replay requires an explicit fresh research initializer')
    sampling_schedule=sampling_rows(rows,manifest.get('research_sampling'))
    monitored_research=bool(manifest.get('research_initialization') and args.loss_monitor_interval and not args.joint_course)
    resource_policy=manifest.get('resource_policy',{})
    if resource_policy.get('self_contact_broad_phase') and args.self_broad_phase!=resource_policy['self_contact_broad_phase']:
        parser.error('This qualified course requires its exact --self-broad-phase policy')
    if resource_policy.get('host_self_trajectories',False) and not args.host_self_trajectories:
        parser.error('This qualified course requires --host-self-trajectories to keep the original GPU budget')
    if 'checkpoint_relation_ffn_rows' in resource_policy:
        required=resource_policy['checkpoint_relation_ffn_rows']
        if type(required) is not int or required<=0 or args.checkpoint_relation_ffn_rows!=required:
            parser.error('This qualified course requires its exact positive --checkpoint-relation-ffn-rows bound')
    if 'checkpoint_attention_blocks' in resource_policy:
        required=resource_policy['checkpoint_attention_blocks']
        if type(required) is not bool or args.checkpoint_attention_blocks!=required:
            parser.error('This qualified course requires its exact --checkpoint-attention-blocks policy')
    if manifest.get('resource_policy',{}).get('host_encoder_checkpoint_inputs',False) and not args.host_encoder_inputs:
        parser.error('This qualified course requires --host-encoder-inputs to keep the original GPU budget')
    if manifest.get('resource_policy',{}).get('host_step_saved_tensors',False) and not args.host_step_saved_tensors:
        parser.error('This qualified course requires --host-step-saved-tensors to keep the original GPU budget')
    if manifest.get('resource_policy',{}).get('host_rod_trajectories',False) and not args.host_rod_trajectories:
        parser.error('This qualified course requires --host-rod-trajectories to keep the original GPU budget')
    if manifest.get('resource_policy',{}).get('compact_host_reference_constants',False) and not args.compact_host_reference_constants:
        parser.error('This qualified course requires --compact-host-reference-constants to keep its host storage budget')
    if manifest.get('resource_policy',{}).get('trim_host_allocator',False) and not args.trim_host_allocator:
        parser.error('This qualified course requires --trim-host-allocator to keep its host storage budget')
    if manifest.get('optimizer_learning_rate_migration') and not args.extend_from:
        parser.error('A declared rate migration requires the bound selected-prefix continuation')
    pure_scope=load_scope(ROOT);training_scope=validate_rows(rows,pure_scope)
    from wp_vbd.development_cases import validation_rows,case_key,comparison_metric,object_errors,equal_dataset_score
    development_rows=validation_rows(rows)
    required_custom_datasets={r['dataset_id'] for r in rows if r.get('demo') in CUSTOM_PURE_VBD}
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.mixed_experiment import load_sample,select,external_inputs,controlled_start,loss,clip_gradient_norm_fp64
    from wp_vbd.rigid_contact_data import load_reference as load_rigid,query as query_rigid
    from wp_vbd.general_joint_data import load_reference as load_joint,set_controls,apply_prescribed_input
    from wp_vbd.surface_joint_data import load_reference as load_surface
    from wp_vbd.self_contact_data import load_reference as load_self
    from wp_vbd.elastic_rod_data import load_reference as load_elastic_rod,step as elastic_rod_step
    from wp_vbd.mixed_contact_data import load_reference as load_mixed,apply_pin_input,query as query_mixed
    from wp_vbd.rod_contact_data import load_reference as load_rod,query as query_rod,apply_prescribed_input as apply_rod
    from wp_vbd.training_window import transfer_window
    from wp_vbd.gpu_budget import GpuBudget
    from wp_vbd.development_baseline import extend_policies,validate_policy,validate_runtime_overrides,PHYSICAL_KEYS,selected_parent_record,verify_pretraining_origin
    torch.set_num_threads(1);torch.manual_seed(20261102);torch.use_deterministic_algorithms(True)
    budget=GpuBudget(args.device,args.gpu_memory_gib)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(args.device)
    if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd='/home/oem/code/engine/newton',text=True).strip()==manifest['newton_sha']
    assert len({r['dataset_id'] for r in rows})==len(rows),'Dataset IDs must be unique'
    initial=ROOT/manifest['initial_checkpoint'];assert sha(initial)==manifest['initial_checkpoint_sha256']
    stored=torch.load(initial,weights_only=True,map_location='cpu')
    config={**stored['model_config'],'constitutive_coarse_scope':'all','compact_neural_relations':True,
            'self_retain_accepted_primal_iterate':True,'self_geometric_step_cap':True}
    if args.checkpoint_neural_encoder:config['neural_encoder_checkpointing']=True
    if args.contact_metric_ray_step:config['contact_metric_ray_step']=True
    if args.elastic_prestress_metric:config['elastic_prestress_metric']=True
    if args.compact_frozen_contact_metric:config['compact_frozen_contact_metric']=True
    if args.checkpoint_contact_rows:config['contact_row_checkpointing']=True
    if args.elastic_rod_self_contact:config['elastic_rod_self_contact']=True
    if args.refresh_self_geometry:config['self_contact_refresh_geometry']=True
    if args.elastic_rod_current_geometry:config['elastic_rod_refresh_self_geometry']=True
    if args.elastic_rod_differentiated_residual:config['elastic_rod_differentiated_residual_backward']=True
    if args.contact_activation_ray_step:config['contact_activation_ray_step']=True
    if args.contact_motion_ray_step:config['contact_motion_ray_step']=True
    if args.contact_structural_support:config['contact_structural_support']=True
    if args.caller_pose_history:config['rod_caller_pose_history']=True
    if args.checkpoint_relation_ffn_rows:config['checkpoint_relation_ffn_rows']=args.checkpoint_relation_ffn_rows
    if args.checkpoint_attention_blocks:config['checkpoint_attention_blocks']=True
    if args.learned_contact_repair:config['learned_contact_repair']=True
    if 'effective_model_config' in manifest:assert config==manifest['effective_model_config']
    model=MixedSolver(config).to(args.device);model.load_state_dict(stored['model'])
    if args.host_encoder_inputs and not config.get('neural_encoder_checkpointing'):parser.error('Host encoder inputs require encoder checkpointing')
    model.protected_response.checkpoint_inputs_on_cpu=args.host_encoder_inputs
    model.step_saved_tensors_on_cpu=args.host_step_saved_tensors
    assert model.protected_response.width==648
    for name,p in model.named_parameters():p.requires_grad_(name.startswith('protected_response.'))
    parameters=[p for p in model.parameters() if p.requires_grad]
    assert sum(p.numel() for p in parameters)==46882811
    relation_parameter=dict(model.named_parameters())['protected_response.hierarchy.relation_kind.weight']
    optimizer=torch.optim.AdamW(parameters,lr=args.learning_rate,eps=1e-12,weight_decay=0.,foreach=False)
    baseline_path=ROOT/'checkpoints/shared_particle_hanging_v63/shared_strict.pt'
    baseline_stored=torch.load(baseline_path,weights_only=True,map_location='cpu')
    baseline=MixedSolver(baseline_stored['config']['model']).to(args.device).eval();baseline.load_state_dict(baseline_stored['model'])
    original_init=newton.solvers.SolverVBD.__init__
    @functools.wraps(original_init)
    def forbidden(*unused,**kwargs):raise AssertionError('Complete SolverVBD is forbidden in training predictions')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden
    checkpoints=ROOT/'checkpoints'/directory.name;checkpoints.mkdir(parents=True,exist_ok=True)
    counts={r['dataset_id']:0 for r in rows};samples=[];history=[];selected=0;best=None
    begin=time.perf_counter();rng=random.Random(20261102)
    resumed_steps=0;resume_evidence=None;inherited_baseline_policies=None;course_offset=0;carried_joint=None
    if args.resume_from:
        prior=json.loads(args.resume_report.read_text())
        recovered=torch.load(args.resume_from,weights_only=True,map_location='cpu')
        assert prior['selected_step']==0
        if not args.resume_recorded_joint:assert 'OutOfMemoryError' in prior.get('failure','')
        assert recovered['initial_checkpoint_sha256']==manifest['initial_checkpoint_sha256']
        assert recovered['data_manifest_sha256']==sha(args.manifest)
        resumed_steps=recovered['step'];assert 0<resumed_steps<=prior['completed_updates']<args.updates
        changed=[k for k in config.keys()|recovered['model_config'].keys() if config.get(k)!=recovered['model_config'].get(k)]
        assert changed==([] if args.resume_recorded_joint else ['neural_encoder_checkpointing']),changed
        model.load_state_dict(recovered['model']);optimizer.load_state_dict(recovered['optimizer'])
        if args.resume_recorded_joint:
            from joint_resume import recorded_prefix,verify_extension_arguments
            schedule=json.loads(args.optimizer_schedule.read_text())
            carried_joint=recorded_prefix(recovered,prior,manifest,schedule,checkpoint_sha=sha(args.resume_from),
                report_sha=sha(args.resume_report),manifest_sha=sha(args.manifest))
            verify_extension_arguments(schedule,args)
            from wp_vbd.training_schedule import apply_learning_rate_migration
            migration=apply_learning_rate_migration(optimizer,args.learning_rate,schedule['migration'])
            course_offset=carried_joint['origin'];history=carried_joint['validation_history']
            inherited_baseline_policies=carried_joint['baseline_policies']
        assert all(g['lr']==args.learning_rate and g['eps']==1e-12 for g in optimizer.param_groups)
        counts=recovered['sampled_updates'];assert set(counts)=={r['dataset_id'] for r in rows}
        samples=copy.deepcopy([s for s in prior['optimization_samples'] if s['update']<=resumed_steps])
        assert len(samples)==sum(counts.values())==resumed_steps
        rng.setstate(recovered['rng_state'])
        resume_evidence={'checkpoint':str(args.resume_from),'checkpoint_sha256':sha(args.resume_from),
                         'report':str(args.resume_report),'report_sha256':sha(args.resume_report),
                         'producer_commit':prior['source_git_commit'],'valid_updates_recovered':resumed_steps,
                         'model_and_optimizer_loaded_strictly':True,'allowed_config_changes':changed,
                         'failed_next_sample_will_be_retried_without_skipping':True}
        if args.resume_recorded_joint:
            resume_evidence.update(mode='recorded_joint_optimizer_schedule_continuation',producer_status=prior['status'],
                original_selected_course_origin=course_offset,optimizer_learning_rate_migration=migration,
                optimizer_schedule_sha256=sha(args.optimizer_schedule),all_recorded_optimizer_calls_preserved=True,
                original_sampling_and_fixed_validation_history_preserved=True,
                failed_next_sample_will_be_retried_without_skipping=False,next_unprocessed_sample_continued_without_skipping=True)
            if carried_joint['schedule'].get('producer_exit_recovery'):
                resume_evidence.update(mode='observed_producer_exit_checkpoint_recovery',
                    all_recorded_optimizer_calls_preserved=not carried_joint['replay_samples'],
                    original_recorded_updates_without_complete_checkpoint=len(carried_joint['replay_samples']),
                    original_end_state_bitexactness_certified=False)
        del recovered,prior
    if args.extend_from:
        prior_raw=args.extend_report.read_bytes();prior=json.loads(prior_raw)
        recovered_raw=args.extend_from.read_bytes()
        import io
        recovered=torch.load(io.BytesIO(recovered_raw),weights_only=True,map_location='cpu')
        prior_manifest_path=args.extend_report.parent/'data_manifest.json'
        prior_manifest=json.loads(prior_manifest_path.read_text())
        selected_parent_record(prior)
        resumed_steps=recovered['step']
        assert resumed_steps==prior['selected_step']<=prior['completed_updates'] and resumed_steps<args.updates
        assert any(v['update']==resumed_steps and v['eligible'] for v in prior['validation_history'])
        runtime_overrides=validate_runtime_overrides(manifest.get('model_configuration_overrides',{}))
        assert config=={**recovered['model_config'],**runtime_overrides}
        assert recovered['initial_checkpoint_sha256']==manifest['initial_checkpoint_sha256']
        assert recovered['data_manifest_sha256']==sha(prior_manifest_path)
        assert prior_manifest['newton_sha']==manifest['newton_sha']
        by_id={r['dataset_id']:r for r in rows}
        assert all(by_id.get(r['dataset_id'])==r for r in prior_manifest['scenes']),'Old replay datasets must remain unchanged'
        parent=manifest['selected_parent']
        assert sha(args.extend_from)==parent['checkpoint_sha256'] and sha(args.extend_report)==parent['report_sha256']
        counts.update(recovered['sampled_updates'])
        samples=copy.deepcopy([s for s in prior['optimization_samples'] if s['update']<=resumed_steps])
        assert len(samples)==sum(counts.values())==resumed_steps
        assert [s['update'] for s in samples]==list(range(1,resumed_steps+1))
        assert all(sum(s['dataset']==label for s in samples)==count for label,count in counts.items())
        model.load_state_dict(recovered['model']);optimizer.load_state_dict(recovered['optimizer'])
        from wp_vbd.training_schedule import apply_learning_rate_migration
        optimizer_migration=apply_learning_rate_migration(optimizer,args.learning_rate,manifest.get('optimizer_learning_rate_migration'))
        rng.setstate(recovered['rng_state']);course_offset=resumed_steps
        resume_evidence={'mode':('completed_selected_course_extension' if prior['status']=='TRAINING_COMPLETED' else
                                'interrupted_course_selected_prefix_extension' if prior['status']=='INTERRUPTED_FOR_COURSE_REPLAN' else
                                'failed_course_selected_prefix_extension'),
                         'producer_status':prior['status'],'producer_failure':prior.get('failure'),'checkpoint':str(args.extend_from),
                         'checkpoint_sha256':sha(args.extend_from),'report':str(args.extend_report),
                         'report_sha256':sha(args.extend_report),'producer_commit':prior['source_git_commit'],
                         'valid_updates_recovered':resumed_steps,'model_and_optimizer_loaded_strictly':True,
                         'allowed_config_changes':sorted(runtime_overrides),'unchanged_replay_datasets':len(prior_manifest['scenes']),
                         'optimizer_learning_rate_migration':optimizer_migration,
                         'parent_completed_optimizer_calls':prior['completed_updates'],
                         'unselected_parent_calls_not_inherited':prior['completed_updates']-resumed_steps,
                         'new_dataset_ids':sorted(set(by_id)-set(recovered['sampled_updates'])),
                         'new_course_selection_requires_all_added_demos':True}
        inherited_baseline_policies=extend_policies(prior,rows,config)
        recorded_sha=prior.get('development_comparison_policy',{}).get('old32_checkpoint_sha256')
        if recorded_sha:assert recorded_sha==sha(baseline_path)
        del recovered,prior,recovered_raw,prior_raw
    if args.joint_course:
        from wp_vbd.joint_course import verify_joint_training_policy
        # Keep the original admission/course audit anchored to its parent;
        # completed extensions have a separately bound additional budget.
        budget_origin=resume_evidence if not args.resume_recorded_joint else {'valid_updates_recovered':course_offset}
        extension=carried_joint['schedule'].get('completed_course_extension') if carried_joint else None
        policy_limit=extension['source_completed_updates'] if extension else args.updates
        verify_joint_training_policy(manifest,ROOT,config,budget_origin,policy_limit,args.validation_interval,args.loss_monitor_interval)
    baseline_policies=manifest.get('development_baseline_policies',inherited_baseline_policies)
    if baseline_policies is None:baseline_policies={case_key(r):{} for r in development_rows}
    baseline_policies={key:validate_policy(value) for key,value in baseline_policies.items()}
    assert set(baseline_policies)=={case_key(r) for r in development_rows}
    if inherited_baseline_policies is not None:assert baseline_policies==inherited_baseline_policies
    if manifest.get('development_baseline_checkpoint_sha256'):assert manifest['development_baseline_checkpoint_sha256']==sha(baseline_path)
    comparison_models={():baseline}
    for policy in baseline_policies.values():
        signature=tuple(sorted(policy.items()))
        if signature not in comparison_models:
            comparison=MixedSolver({**baseline.config,**policy}).to(args.device).eval()
            comparison.load_state_dict(baseline.state_dict());comparison_models[signature]=comparison
    baseline_physical_overrides={key:True for key in PHYSICAL_KEYS if config.get(key,False)}
    report={'status':'RUNNING','device':args.device,'completed_updates':0,'selected_step':0,
            'active_neural_parameters':46882811,'initial_checkpoint_sha256':manifest['initial_checkpoint_sha256'],
            'inherited_selected_official_demos':manifest.get('inherited_selected_official_demos',0),'old32_demo_coverage_inherited':0,'newton_sha':manifest['newton_sha'],'full_vbd_prediction_calls':0,
            'config':config,'optimizer':{'learning_rate':args.learning_rate,'epsilon':1e-12,'weight_decay':0.,'foreach':False},
            'gradient_normalization':args.gradient_normalization,
            'planned_maximum_updates':args.updates,
            'training_scope':training_scope,
            'resource_budget':{'process_gpu_gib':args.gpu_memory_gib,'torch_allocator_gib':args.gpu_memory_gib-2,'duty_cycle':args.duty_cycle,'batch_size':1,'horizon':2,'host_mixed_trajectories':args.host_mixed_trajectories,'host_self_trajectories':args.host_self_trajectories,'host_rod_trajectories':args.host_rod_trajectories,'host_encoder_checkpoint_inputs':args.host_encoder_inputs,'contact_row_checkpointing':bool(config.get('contact_row_checkpointing',False)),'self_contact_broad_phase':args.self_broad_phase or ('kdtree' if args.device=='cpu' else 'chunked')},
            'source_git_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'script_sha256':sha(Path(__file__)),'module_sha256':{p.name:sha(p) for p in (ROOT/'src/wp_vbd').glob('*.py')},
            'optimization_samples':samples,'validation_history':history,
            'development_comparison_policy':{'inherited_demos':'unchanged recorded32 physical configuration and existing caps',
                                             'new_demos':'same32 neural weights with explicitly matched physical proposal/retention settings',
                                             'new_demo_baseline_physical_overrides':baseline_physical_overrides,
                                             'per_demo_physical_overrides':baseline_policies,
                                             'old32_checkpoint_sha256':sha(baseline_path)},
            'resume_evidence':resume_evidence,'resumed_valid_updates':resumed_steps,
            'selection_policy':'Every scheduled official demo needs a nonzero-gradient optimizer sample with an actually changed scalar. Every mobile object in development must stay within old32 error +20%*max(old32 error,0.1). Among eligible epochs minimize equal-dataset relative error. Native task and realtime certification remain separate.',
            'scope':'New648-wide full-backbone training, no inherited official counts from the old32-wide model. Current pure-VBD target is42; nine external-solver demos are excluded by user scope. Development is not held-out material/topology or complete native acceptance.'}
    report['resource_budget']['host_step_saved_tensors']=args.host_step_saved_tensors
    report['resource_budget']['compact_host_reference_constants']=args.compact_host_reference_constants
    report['resource_budget']['trim_host_allocator']=args.trim_host_allocator
    report['resource_budget']['checkpoint_relation_ffn_rows']=config.get('checkpoint_relation_ffn_rows',0)
    report['resource_budget']['checkpoint_attention_blocks']=config.get('checkpoint_attention_blocks',False)
    if manifest.get('research_sampling'):
        report['research_sampling']={**manifest['research_sampling'],'updates_per_pass':len(sampling_schedule),
            'unique_dataset_count':len(rows),'repeated_visits_are_not_extra_datasets':True}
    if manifest.get('research_initialization'):
        report['research_initialization']=manifest['research_initialization']
    if monitored_research:
        report['loss_monitor_history']=[]
        report['loss_specification']={'state':'1e4 normalized position MSE +100 SO3 rotation MSE',
            'contact':'1000 normalized raw-proposal / detached local-repair target MSE, mean over repair calls',
            'two_hop_scaling':'Average both autoregressive hops; multiply by ((1/120)/h)^4',
            'scope':'Fixed-window development loss; native long-rollout appearance and realtime are evaluated separately.'}
    if args.joint_course:
        report['joint_course_policy']=manifest['joint_course_policy']
        report['joint_course_origin_updates']=course_offset
        report['selection_policy']='Select the lowest balanced fixed-window validation checkpoint after observed plateau, provided every scheduled dataset learned with changed weights and all final per-object development caps pass. Native long rollouts and realtime remain separate.'
        report['loss_monitor_history']=carried_joint['monitors'] if carried_joint else []
        if carried_joint:
            report['optimizer_schedule']=carried_joint['schedule']
            report['optimizer_schedule_history']=[*carried_joint['prior_schedules'],carried_joint['schedule']]
            if carried_joint['schedule'].get('producer_exit_recovery'):
                report['recovery_replay']={'restored_update':resumed_steps,
                    'original_recorded_update':carried_joint['schedule']['producer_exit_recovery']['producer_recorded_updates'],
                    'expected_replayed_samples':len(carried_joint['replay_samples']),'verified_samples':[],
                    'original_end_state_bitexactness_certified':False}
            from analyze_joint_training_loss import analyze
            report['loss_convergence']=analyze(report['loss_monitor_history'])
            if carried_joint['lowest']:
                report['lowest_fixed_validation']={**carried_joint['lowest'],'checkpoint':str(checkpoints/'lowest_fixed_loss.pt')}
        report['loss_specification']={'state':'1e4 normalized position MSE +100 SO3 rotation MSE',
            'contact':'1000 normalized raw-proposal / detached local-repair target MSE, mean over repair calls',
            'two_hop_scaling':'Average both hops; multiply by ((1/120)/h)^4',
            'architecture_scope':'Existing shared648 inverse-response backbone and four physical rounds retained for this contact-learning iteration. A direct feed-forward dynamics replacement is not implemented by this flag.'}
    def trim_host():
        if args.trim_host_allocator:
            from wp_vbd.host_allocator import release_unused_host_memory
            release_unused_host_memory()
    def persist(status=None):
        report.update(completed_updates=len(samples),current_trial_optimizer_updates=len(samples)-resumed_steps,selected_step=selected,seconds=time.perf_counter()-begin,sampled_updates=counts.copy())
        if status:report['status']=status
        write(directory/'training.json',report)
    def save(name,step):
        from checkpoint_storage import atomic_save
        target=checkpoints/name
        atomic_save({'architecture':model.architecture_id,'model':model.state_dict(),'model_config':config,
                    'optimizer':optimizer.state_dict(),'step':step,'sampled_updates':counts.copy(),
                    'initial_checkpoint_sha256':manifest['initial_checkpoint_sha256'],'data_manifest_sha256':sha(args.manifest),
                    'rng_state':rng.getstate()},target)
    def idle(work_seconds):
        if args.device.startswith('cuda'):
            remaining=work_seconds*(1/args.duty_cycle-1)
            while remaining>0:
                duration=min(remaining,5.);time.sleep(duration);remaining-=duration
    def release():
        optimizer.zero_grad(set_to_none=True);gc.collect()
        if args.device.startswith('cuda'):torch.cuda.empty_cache()
    def load(row,validation=False):
        path=ROOT/(row['validation_path'] if validation and row.get('validation_path') else row['path'])
        expected=row.get('validation_sha256') if validation and row.get('validation_path') else row['sha256']
        assert sha(path)==expected,path
        loader=row['loader']
        scene,states=(load_sample(path,args.device) if loader=='legacy' else
                      load_elastic_rod(path,wp,newton,args.device,state_device='cpu' if args.host_self_trajectories else args.device) if loader=='elastic_rod' else
                      load_mixed(path,wp,newton,args.device,state_device='cpu') if loader=='mixed' and args.host_mixed_trajectories else
                      load_self(path,wp,newton,args.device,state_device='cpu',compact_reference_constants=args.compact_host_reference_constants) if loader=='self' and args.host_self_trajectories else
                      load_rod(path,wp,newton,args.device,state_device='cpu') if loader=='rod' and args.host_rod_trajectories else
                      {'rigid':load_rigid,'joint':load_joint,'rod':load_rod,'surface':load_surface,'self':load_self,'mixed':load_mixed}[loader](path,wp,newton,args.device))
        self_query=args.self_broad_phase or ('kdtree' if args.device=='cpu' else 'chunked')
        response_pairs=row.get('self_contact_responsive_pairs_only',False)
        if type(response_pairs) is not bool:raise ValueError('Explicit Boolean self-contact pair eligibility required')
        motion_budget=row.get('self_contact_motion_budget')
        if motion_budget is not None and (isinstance(motion_budget,bool) or not isinstance(motion_budget,(int,float)) or not math.isfinite(motion_budget) or motion_budget<=0):raise ValueError('Finite positive complete motion envelope required')
        if loader in ('self','elastic_rod') and (self_query!='chunked' or row.get('self_contact_max_rows') is not None or response_pairs or motion_budget is not None):
            from wp_vbd.self_contact_runtime import SelfContactCollisionOracle
            from wp_vbd.elastic_rod_runtime import ElasticRodCollisionOracle
            scene.metadata['self_contact_broad_phase']=self_query
            if response_pairs:scene.metadata['self_contact_responsive_pairs_only']=True
            if motion_budget is not None:scene.metadata['self_contact_motion_budget']=float(motion_budget)
            if row.get('self_contact_max_rows') is not None:
                capacity=row['self_contact_max_rows']
                if type(capacity) is not int or capacity<=0:raise ValueError('Positive complete self-contact capacity required')
                scene.metadata['self_contact_max_rows']=capacity
            oracle_type=ElasticRodCollisionOracle if loader=='elastic_rod' else SelfContactCollisionOracle
            scene.geometry_oracle=oracle_type(scene.geometry_oracle.source_model,scene,wp,newton)
        trim_host()
        return scene,states
    def advance(network,row,scene,states,current,index):
        from wp_vbd.self_contact_repair import configure as configure_repair
        configure_repair(scene,network.config)
        loader=row['loader']
        if loader=='elastic_rod':
            result=elastic_rod_step(network,scene,current,states,index,wp,before_model=trim_host)
            trim_host();return result
        if loader=='rod':
            from wp_vbd.rod_caller_window import enabled as caller_enabled,step as caller_step
            if caller_enabled(scene):
                predicted,diagnostic=caller_step(network,scene,current,states,index,wp,before_model=trim_host)
                trim_host();return predicted,diagnostic
        if loader=='legacy':current=controlled_start(scene,current,states,index)
        elif loader=='rod':current=apply_rod(scene,current,states,index);query_rod(scene,current,wp)
        elif loader in ('surface','self'):
            set_controls(scene,states,index);current=apply_pin_input(scene,current,states,index);query_mixed(scene,current,wp)
        elif loader=='mixed':current=apply_pin_input(scene,current,states,index);query_mixed(scene,current,wp)
        else:
            if loader=='joint':set_controls(scene,states,index);current=apply_prescribed_input(scene,current,states,index)
            query_rigid(scene,current,wp)
        trim_host()
        predicted,diagnostic=network(scene,current,scene.metadata['dt'],**external_inputs(states,slice(index,index+1)))
        trim_host()
        return predicted,diagnostic
    def start_index(row,scene,states,validation=False):
        length=len(states['x'])-1;label=row['dataset_id'];visits=counts[label]
        if scene.metadata.get('data_layout')=='teacher_queried_windows_v1':return int(scene.qualified_starts[0 if validation else rng.randrange(len(scene.qualified_starts))])
        if validation:return row.get('validation_starts',[min(int(.8*length),length-3)])[0]
        if row.get('qualified_train_starts'):
            starts=row['qualified_train_starts']
            if not all(isinstance(i,int) and 0<=i<length-1 for i in starts):raise ValueError('Complete two-step qualified windows required')
            if set(starts)&set(row.get('validation_starts',[])):raise ValueError('Training and development windows must be distinct')
            return starts[visits%len(starts)]
        if label=='cable_bundle_hysteresis_train_default':return (rng.randrange(100,599),rng.randrange(601,1199),1200,rng.randrange(1202,int(.6*length)))[visits%4]
        if row['loader']=='surface':return 20*rng.randrange(5,60) if visits%2==0 else rng.randrange(1400,int(.6*length))
        lo,hi=row.get('train_start_range',[min(20,length-3),max(21,int(.6*length))])
        if hi<=lo:lo,hi=0,length-2
        return rng.randrange(lo,hi)
    def validate():
        model.eval();metrics=[]
        with torch.no_grad():
            for row in development_rows:
                key=case_key(row);case_begin=time.perf_counter();scene,states=load(row,True)
                index=start_index(row,scene,states,True);physical_index=index;outputs=[]
                if (row['loader']=='mixed' and args.host_mixed_trajectories) or (row['loader'] in ('self','elastic_rod') and args.host_self_trajectories) or (row['loader']=='rod' and args.host_rod_trajectories):
                    states=transfer_window(states,index,2,args.device);physical_index=0
                policy=baseline_policies[key];use_matched=bool(policy)
                comparison=comparison_models[tuple(sorted(policy.items()))]
                for network in (comparison,model):
                    current=select(states,slice(physical_index,physical_index+1))
                    for offset in range(2):
                        current,d=advance(network,row,scene,states,current,physical_index+offset)
                        if not bool(d['accepted'].all()):
                            network_label='shared648' if network is model else ('old32_matched_physics' if use_matched else 'old32_original_physics')
                            raise RuntimeError(f'Development prediction rejected {key} {index+offset} network={network_label}')
                    outputs.append(current)
                target=select(states,slice(physical_index+2,physical_index+3))
                by_object=object_errors(scene,outputs,target)
                metrics.append(comparison_metric(row,index,by_object,policy))
                if args.device.startswith('cuda'):metrics[-1]['observed_process_gpu_mib']=budget.observe(force=True)
                print(json.dumps({'development':metrics[-1]}),flush=True)
                if args.development_only:
                    write(directory/'pretraining_development_progress.json',{
                        'status':'RUNNING','completed_windows':len(metrics),'expected_windows':len(development_rows),
                        'metrics':metrics,'new_optimizer_updates':0,'device':args.device,
                        'data_manifest_sha256':sha(args.manifest),
                        'initial_checkpoint_sha256':manifest['initial_checkpoint_sha256'],
                        'script_sha256':report['script_sha256'],'module_sha256':report['module_sha256']})
                del scene,states,current,outputs,target,d;release();idle(time.perf_counter()-case_begin)
        model.train();return metrics
    def monitor_loss(update):
        model.eval();measured=[];by_dataset={}
        representative={'cloth.example_cloth_twist','softbody.example_softbody_hanging',
            'vbd.example_vbd_rigid_rigid_contact','basic.example_basic_joints','cable.example_cable_twist'}
        with torch.no_grad():
            for row in development_rows:
                if row.get('demo') not in CUSTOM_PURE_VBD and row.get('demo') not in representative:continue
                began_case=time.perf_counter();scene,states=load(row,True)
                index=start_index(row,scene,states,True);physical_index=index
                if args.host_self_trajectories:states=transfer_window(states,index,2,args.device);physical_index=0
                current=select(states,slice(physical_index,physical_index+1));objective=current.x.new_zeros(())
                state_objective=current.x.new_zeros(());repair_objective=current.x.new_zeros(())
                scale=(1/120/scene.metadata['dt'])**4
                for offset in range(2):
                    current,diagnostic=advance(model,row,scene,states,current,physical_index+offset)
                    if not bool(diagnostic['accepted'].all()):raise RuntimeError('Fixed loss-monitor prediction rejected '+case_key(row))
                    state_objective=state_objective+loss(scene,current,select(states,slice(physical_index+offset+1,physical_index+offset+2)))*scale/2
                    repair_objective=repair_objective+diagnostic.get('contact_repair_supervision',0.)*scale/2
                    objective=state_objective+repair_objective
                value=float(objective)
                if not math.isfinite(value):raise RuntimeError('Nonfinite fixed-window loss')
                item={'after_update':update,'dataset_id':row['dataset_id'],'index':index,'two_hop_loss_scaled':value,
                    'state_loss_scaled':float(state_objective),'repair_loss_scaled':float(repair_objective)}
                if args.device.startswith('cuda'):item['observed_process_gpu_mib']=budget.observe(force=True)
                measured.append(item);by_dataset.setdefault(row['dataset_id'],[]).append(value)
                print(json.dumps({'loss_monitor':item}),flush=True)
                del scene,states,current,objective,state_objective,repair_objective,diagnostic;release();idle(time.perf_counter()-began_case)
        model.train()
        entry={'after_update':update,'fixed_windows':measured,'dataset_means':{k:sum(v)/len(v) for k,v in by_dataset.items()},
               'scope':'Fixed authored and five official-family development windows; no gradients, optimizer updates, old32 comparisons or all42 quality claim'}
        report['loss_monitor_history'].append(entry);persist()
        from analyze_joint_training_loss import analyze
        report['loss_convergence']=analyze(report['loss_monitor_history'])
        if update>resumed_steps:
            score=report['loss_convergence']['balanced_relative_scores'][-1]
            if score<report.get('lowest_fixed_validation',{}).get('score',float('inf')):
                save('lowest_fixed_loss.pt',update)
                report['lowest_fixed_validation']={'update':update,'score':score,'checkpoint':str(checkpoints/'lowest_fixed_loss.pt')}
        persist()
    if args.diagnostic_plan:
        if not args.resume_recorded_joint:raise ValueError('An isolated diagnostic requires a real recorded source')
        from probe_joint_update_direction import run_diagnostic
        run_diagnostic(args=args,report=report,model=model,optimizer=optimizer,parameters=parameters,
            rows=rows,load=load,advance=advance,release=release,budget=budget)
        return
    if args.development_only:
        path=directory/'pretraining_development.json'
        if path.exists():raise FileExistsError(path)
        try:metrics=validate()
        except Exception as error:
            write(path,{'status':'EFFECTIVE_RUNTIME_DEVELOPMENT_EXCEPTION','failure':repr(error),
                        'data_manifest_sha256':sha(args.manifest),'config':config,
                        'development_comparison_policy':report['development_comparison_policy'],
                        'script_sha256':report['script_sha256'],'module_sha256':report['module_sha256'],
                        'new_optimizer_updates':0,'new_training_admissions':0,'full_vbd_prediction_calls':0})
            raise
        caps=all(o['within_cap'] for r in metrics for o in r['objects'])
        if not all(math.isfinite(o[key]) for r in metrics for o in r['objects'] for key in ('old32','new648')):
            raise RuntimeError('Nonfinite pretraining development metric')
        result={'status':'EFFECTIVE_RUNTIME_DEVELOPMENT_PASS' if caps else 'EFFECTIVE_RUNTIME_DEVELOPMENT_FAILED',
                'device':args.device,'data_manifest_sha256':sha(args.manifest),'config':config,
                'source_git_commit':report['source_git_commit'],'script_sha256':report['script_sha256'],'module_sha256':report['module_sha256'],
                'parent_checkpoint_sha256':resume_evidence['checkpoint_sha256'] if resume_evidence else None,
                'initial_checkpoint_sha256':manifest['initial_checkpoint_sha256'],
                'research_initialization':manifest.get('research_initialization'),
                'parent_selected_step':resumed_steps,'metrics':metrics,'all_mobile_objects_within_existing_caps':caps,
                'development_comparison_policy':report['development_comparison_policy'],
                'new_optimizer_updates':0,'new_training_admissions':0,'full_vbd_prediction_calls':0,
                'optimizer':report['optimizer'],
                'resource_budget':report['resource_budget'],'seconds':time.perf_counter()-begin,
                'gpu_resource_budget':budget.report(),
                'scope':'All scheduled development demos on the effective migrated runtime and unchanged selected NN weights. This check adds no training coverage, full native quality, held-out material/topology or realtime certification.'}
        if manifest.get('research_initialization'):
            result.update(status='RESEARCH_INITIAL_DEVELOPMENT_READY',all_predictions_accepted=True,
                initial_quality_caps_are_final_selection_only=True,
                scope='Exact fresh research initializer; all scheduled GPU predictions accepted with finite metrics and observed resource budgets. Initial old32 quality comparisons are retained, including failures, and do not preselect these research weights. Final training selection rules are unchanged.')
        write(path,result);print(json.dumps(result),flush=True)
        if not caps and not manifest.get('research_initialization'):raise SystemExit(1)
        return
    if not args.joint_course and (manifest.get('model_configuration_overrides') or manifest.get('optimizer_learning_rate_migration')
            or manifest.get('resource_policy',{}).get('host_encoder_checkpoint_inputs')):
        proof=json.loads((directory/'pretraining_development.json').read_text())
        if manifest.get('research_initialization'):
            assert proof['status']=='RESEARCH_INITIAL_DEVELOPMENT_READY' and proof['all_predictions_accepted']
            assert proof['initial_quality_caps_are_final_selection_only']
        else:
            assert proof['status']=='EFFECTIVE_RUNTIME_DEVELOPMENT_PASS' and proof['all_mobile_objects_within_existing_caps']
        assert proof['config']==config and proof['data_manifest_sha256']==sha(args.manifest)
        assert proof['module_sha256']==report['module_sha256'] and proof['script_sha256']==report['script_sha256']
        if manifest.get('resource_policy',{}).get('host_encoder_checkpoint_inputs'):
            assert proof['resource_budget']==report['resource_budget']
        if manifest.get('optimizer_learning_rate_migration'):assert proof['optimizer']==report['optimizer']
        report['pretraining_origin']=verify_pretraining_origin(manifest,proof,resume_evidence)
        assert proof['new_optimizer_updates']==proof['new_training_admissions']==0
        report['pretraining_development_sha256']=sha(directory/'pretraining_development.json')
    persist()
    try:
        if (args.joint_course and not args.resume_recorded_joint) or monitored_research:monitor_loss(resumed_steps)
        for update in range(resumed_steps+1,args.updates+1):
            work_begin=time.perf_counter();row=sampling_schedule[(update-course_offset-1)%len(sampling_schedule)];label=row['dataset_id']
            if args.device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            scene,states=load(row);index=start_index(row,scene,states)
            physical_index=index
            if (row['loader']=='mixed' and args.host_mixed_trajectories) or (row['loader'] in ('self','elastic_rod') and args.host_self_trajectories) or (row['loader']=='rod' and args.host_rod_trajectories):
                states=transfer_window(states,index,2,args.device);physical_index=0
            model.train();optimizer.zero_grad(set_to_none=True);current=select(states,slice(physical_index,physical_index+1));objective=current.x.new_zeros(())
            diagnostic=[];scale=(1/120/scene.metadata['dt'])**4
            state_objective=current.x.new_zeros(());repair_objective=current.x.new_zeros(())
            for offset in range(2):
                current,d=advance(model,row,scene,states,current,physical_index+offset)
                if not bool(d['accepted'].all()):raise RuntimeError(f'Training prediction rejected {label} {index+offset}')
                state_objective=state_objective+loss(scene,current,select(states,slice(physical_index+offset+1,physical_index+offset+2)))*scale/2
                repair_objective=repair_objective+d.get('contact_repair_supervision',0.)*scale/2
                objective=state_objective+repair_objective
                diagnostic.append({k:d[k] for k in ('active_rigid_contact_rows','active_particle_shape_rows','active_dynamic_particle_body_rows','active_self_rows_by_kind','active_soft_cloth_self_rows','rod_rows_with_nonzero_wrench','finite_joint_rows_with_nonzero_wrench') if k in d})
                if 'contact_repair_calls' in d:
                    diagnostic[-1].update(contact_repair_calls=d['contact_repair_calls'],contact_repair_supervision=float(d['contact_repair_supervision'].detach()),
                        maximum_oriented_overlap_after_m=max((r['maximum_oriented_overlap_after_m'] for r in d['contact_repairs']),default=0.))
            if not bool(torch.isfinite(objective)):raise RuntimeError('Nonfinite training loss')
            objective.backward();norm=clip_gradient_norm_fp64(parameters,1.)
            if args.gradient_normalization=='unit':
                from probe_joint_update_direction import normalize_unit_gradient
                gradient_extra_scale=normalize_unit_gradient(parameters,norm)
            else:gradient_extra_scale=1.
            available=[p for p in parameters if p.grad is not None]
            norms=torch.stack([p.grad.abs().amax() for p in available]);chosen=available[int(norms.argmax())]
            scalar_index=int(chosen.grad.abs().reshape(-1).argmax());before=float(chosen.reshape(-1)[scalar_index])
            relation_before=relation_parameter.detach().clone()
            relation_gradient=(relation_parameter.grad.norm(dim=-1) if relation_parameter.grad is not None else relation_parameter.new_zeros(len(relation_parameter))).detach().cpu().tolist()
            optimizer.step();changed=before!=float(chosen.reshape(-1)[scalar_index])
            relation_changed=(relation_before!=relation_parameter.detach()).any(-1).cpu().tolist()
            counts[label]+=1
            sample={'update':update,'dataset':label,'demo':row.get('demo'),'index':index,'two_hop_loss_scaled':float(objective.detach()),
                    'state_loss_scaled':float(state_objective.detach()),'repair_loss_scaled':float(repair_objective.detach()),
                    'gradient_norm_before_clip':float(norm),'actual_max_gradient_scalar_changed':changed,'physical_rows':diagnostic,
                    'gradient_normalization':args.gradient_normalization,'gradient_extra_scale':gradient_extra_scale,
                    'relation_embedding_gradient_norm_after_clip':relation_gradient,'relation_embedding_rows_changed':relation_changed}
            if args.device.startswith('cuda'):
                torch.cuda.synchronize();sample.update(peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
                sample['observed_process_gpu_mib']=budget.observe(force=True)
            samples.append(sample)
            if carried_joint and carried_joint['replay_samples']:
                expected=next((value for value in carried_joint['replay_samples'] if value['update']==update),None)
                if expected:
                    from checkpoint_storage import verify_replayed_sample
                    report['recovery_replay']['verified_samples'].append(verify_replayed_sample(sample,expected))
            persist();print(json.dumps(sample),flush=True)
            if args.device.startswith('cuda') and sample['observed_process_gpu_mib']>args.gpu_memory_gib*1024:raise MemoryError('Shared-GPU process budget exceeded')
            del scene,states,current,objective,state_objective,repair_objective,d,available,norms,chosen,relation_before;release()
            if update%10==0:save('last.pt',update)
            idle(time.perf_counter()-work_begin)
            stop_for_plateau=False
            if (args.joint_course or monitored_research) and ((update-course_offset)%args.loss_monitor_interval==0 or update==args.updates):
                save('last.pt',update);monitor_loss(update)
                stop_for_plateau=bool(args.joint_course and report['loss_convergence']['converged'])
            if (update-course_offset)%args.validation_interval==0 or update==args.updates or stop_for_plateau:
                evaluation_update=update;selection_snapshot=None
                if args.joint_course and report.get('lowest_fixed_validation',{}).get('update',update)!=update:
                    selection_snapshot=torch.load(checkpoints/'lowest_fixed_loss.pt',weights_only=True,map_location='cpu')
                    model.load_state_dict(selection_snapshot['model']);evaluation_update=selection_snapshot['step']
                metrics=validate();caps=all(o['within_cap'] for r in metrics for o in r['objects'])
                coverage=changed_weight_coverage(samples,pure_scope,selected_step=evaluation_update)
                learned=set(coverage['official_demos'])
                covered=(set(manifest['current_course_official_demos'])<=learned and
                         required_custom_datasets<=set(coverage['custom_pure_vbd_datasets']))
                score=equal_dataset_score(metrics)
                eligible=caps and covered and (not args.joint_course or report.get('loss_convergence',{}).get('converged',False))
                history.append({'update':evaluation_update,'score':score,'eligible':eligible,'actually_learned_official_demos':sorted(learned),'metrics':metrics})
                if required_custom_datasets:
                    history[-1].update(actually_learned_custom_pure_vbd_demos=coverage['custom_pure_vbd_demos'],
                                       actually_learned_custom_pure_vbd_datasets=coverage['custom_pure_vbd_datasets'])
                if eligible and (best is None or score<best):
                    best=score;selected=evaluation_update
                    if selection_snapshot is None:save('shared_candidate.pt',update)
                    else:
                        temporary=(checkpoints/'shared_candidate.pt').with_suffix('.tmp')
                        torch.save(selection_snapshot,temporary);temporary.replace(checkpoints/'shared_candidate.pt')
                if selection_snapshot is not None:
                    latest=torch.load(checkpoints/'last.pt',weights_only=True,map_location='cpu');model.load_state_dict(latest['model'])
                    del latest,selection_snapshot
                save('last.pt',update);persist()
            if stop_for_plateau:
                if selected:
                    report['early_stop']={'update':update,'planned_maximum_updates':args.updates,'reason':'Observed fixed-validation plateau after at least four complete courses; full development passed before selection.'}
                    break
                report['plateau_without_eligible_physics']={'update':update,'action':'Continue the declared course rather than stop on loss alone.'}
                persist()
        persist('TRAINING_COMPLETED')
        if selected:
            chosen=torch.load(checkpoints/'shared_candidate.pt',weights_only=True,map_location='cpu')
            selected_coverage=changed_weight_coverage(samples,pure_scope,selected_step=selected)
            demos=selected_coverage['official_demos']
            checkpoint_report={'status':'TRAINED_CANDIDATE_REQUIRES_NATIVE_TASK_CHECKS','checkpoint':str((checkpoints/'shared_candidate.pt').relative_to(ROOT)),
                  'checkpoint_sha256':sha(checkpoints/'shared_candidate.pt'),'selected_updates':selected,'trained_official_demos':demos,'trained_official_demo_count':len(demos),
                  'active_neural_parameters':46882811,'inherited_selected_official_demos':manifest.get('inherited_selected_official_demos',0),'old32_demo_coverage_inherited':0,'full_vbd_prediction_calls':0,'scope':'Actual changed-weight optimizer samples in the selected prefix; ordinary-demo development caps passed. Full native quality, remaining targets and realtime are not certified.'}
            if required_custom_datasets:
                checkpoint_report.update(trained_custom_pure_vbd_demos=selected_coverage['custom_pure_vbd_demos'],
                    trained_custom_pure_vbd_demo_count=len(selected_coverage['custom_pure_vbd_demos']),
                    trained_custom_pure_vbd_datasets=selected_coverage['custom_pure_vbd_datasets'])
            write(directory/'checkpoint.json',checkpoint_report)
    except Exception as error:
        report['failure']=repr(error)
        try:save('interrupted_or_rejected.pt',len(samples))
        except Exception as checkpoint_error:report['failure_checkpoint_error']=repr(checkpoint_error)
        persist('FAILED_WITH_RETAINED_WEIGHTS');raise


if __name__=='__main__':main()
