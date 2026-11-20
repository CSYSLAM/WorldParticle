"""Supervise the shared backbone after a bounded analytic state prediction."""
import argparse
from collections import OrderedDict,Counter
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--updates',type=int,default=200)
    p.add_argument('--device',default='cuda:0');p.add_argument('--gpu-memory-gib',type=float,default=12.)
    p.add_argument('--preflight',action='store_true');p.add_argument('--validation-interval',type=int,default=50)
    p.add_argument('--replay-manifest',type=Path)
    p.add_argument('--analytic-elastic-postsolve',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--analytic-predictor-rounds',type=int,choices=(1,2,3),default=1)
    p.add_argument('--replay-hops',type=int,help='Explicit autonomous horizon for existing robot references')
    p.add_argument('--replay-every',type=int,default=4)
    p.add_argument('--balance-robot-phases',action='store_true')
    p.add_argument('--official-manifest',type=Path,help='Byte-audited whole-timeline course for all 42 standalone VBD demos')
    p.add_argument('--official-per-update',type=int,default=2)
    p.add_argument('--official-validation-interval',type=int,default=100)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--resume',type=Path,help='Resume an atomic completed-update recovery checkpoint into a new output directory')
    p.add_argument('--recovery-interval',type=int,default=10)
    p.add_argument('--self-contact-capacity',type=int,help='Raise offline reference query capacity without changing any candidate geometry or filters')
    p.add_argument('--preflight-official-only',action='store_true')
    p.add_argument('--preflight-demo',action='append',default=[])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    if (a.replay_hops is not None and not 1<=a.replay_hops<=24 or a.replay_every<1 or
        a.official_per_update<1 or a.official_validation_interval<1):
        p.error('Bounded positive replay budgets required')
    if (a.preflight_official_only or a.preflight_demo) and (not a.preflight or not a.official_manifest):
        p.error('Official-only/subset checks require --preflight and --official-manifest')
    if not 0<a.learning_rate<1:p.error('A positive finite learning rate below one required')
    if a.recovery_interval<1 or a.preflight and a.resume:p.error('Positive recovery interval and training-only resume required')
    if a.self_contact_capacity is not None and a.self_contact_capacity<1:p.error('Positive complete self-contact row capacity required')
    import hashlib
    snapshot=a.output/'implementation_snapshot';snapshot.mkdir()
    source_files=[Path(__file__),*(ROOT/'src/wp_vbd').glob('*.py')]
    source_hashes={}
    for source in source_files:
        relative=source.resolve().relative_to(ROOT)
        target=snapshot/relative;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);source_hashes[str(relative)]=hashlib.sha256(target.read_bytes()).hexdigest()
    (a.output/'implementation_snapshot_manifest.json').write_text(json.dumps(source_hashes,indent=2)+'\n')
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.gpu_ownership import claim_cuda
    from wp_vbd.gpu_budget import GpuBudget
    from wp_vbd.augmentation_data import load_reference,advance,FAMILIES
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.mixed_experiment import select,clip_gradient_norm_fp64
    from wp_vbd.multirate_objective import RolloutObjective
    from wp_vbd.multirate_window import inputs_for_span
    from checkpoint_storage import atomic_save
    from wp_vbd.full_demo_course import select_timeline_plan,device_window
    from wp_vbd.supervised_step import advance_reference_span
    ownership=claim_cuda(a.device,ROOT);budget=GpuBudget(a.device,a.gpu_memory_gib)
    budget.trim_unused_cache=True
    torch.set_num_threads(1);torch.manual_seed(20261119);torch.use_deterministic_algorithms(True)
    rng=random.Random(20261119)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(a.device)
    if a.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    collection=json.loads(a.collection.read_text())
    if collection['newton_sha']!=newton_revision():raise ValueError('Teacher source revision changed')
    train=[r for r in collection['scenes'] if r['split']=='training']
    dev=[r for r in collection['scenes'] if r['split']=='development']
    if not a.preflight and (len(train)<300 or len(dev)<30):raise ValueError('At least 300 independent training and 30 development trajectories required')
    official=[];official_audit=None
    if a.official_manifest:
        import hashlib
        official_audit=json.loads(a.official_manifest.read_text());official=official_audit['scenes']
        scope=json.loads((ROOT/'configs/pure_vbd_demo_scope.json').read_text())
        declared={r.get('demo') for r in official};allowed=set(scope['target_demos'])
        subset_probe=bool(a.preflight and a.preflight_demo)
        if (official_audit['newton_sha']!=newton_revision() or declared-allowed or
            (not subset_probe and (declared!=allowed or len(official)!=42)) or
            not all(r.get('full_demo_timeline') for r in official)):
            raise ValueError('Every one of the 42 pure VBD demos needs a continuous, indexed reference')
        from prepare_full_demo_course import digest
        for row in official:
            if subset_probe and row['demo'] not in a.preflight_demo:continue
            if digest(row['path'])!=row['sha256']:raise ValueError('Official trajectory changed: '+row['dataset_id'])
        official_audit_sha=hashlib.sha256(a.official_manifest.read_bytes()).hexdigest()
    official_development_occurrence={row['dataset_id']:i%6 for i,row in enumerate(official)}
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config={**(saved.get('model_config') or saved['config']['model']),
        'analytic_predictor_rounds':a.analytic_predictor_rounds,'analytic_elastic_response':True,'unroll':1,'rod_unroll':1,
        'learned_elastic_shape_direction':False,'elastic_shape_local_heads':0,
        'learned_contact_tangent_seed':False,'terminal_velocity_correction':True,'shared_terminal_velocity':True,
        'resident_static_structure':True,'frozen_residual_backward':True,'maximum_protected_columns':32}
    if a.analytic_elastic_postsolve is not None:config['analytic_elastic_postsolve']=a.analytic_elastic_postsolve
    if official:
        config.update(analytic_material_maximum_nodes=8192,analytic_material_block_solve=True)
    model=MixedSolver(config).to(a.device)
    shared_parent=bool((saved.get('model_config') or saved['config']['model']).get('shared_terminal_velocity',False))
    retired=('elastic_shape_head.','contact_tangent_head.')+(() if shared_parent else ('terminal_velocity.',))
    compatible={k:v for k,v in saved['model'].items() if not k.startswith(retired)}
    missing,extra=model.load_state_dict(compatible,strict=False)
    expected_missing=set() if shared_parent else {'terminal_velocity.shared_readout.weight','terminal_velocity.shared_readout.bias'}
    if set(missing)!=expected_missing or extra:
        raise ValueError(f'Checkpoint migration mismatch {missing}, {extra}')
    del saved,compatible
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(('protected_response.','terminal_velocity.')))
    model.step_saved_tensors_on_cpu=False;model.protected_response.checkpoint_inputs_on_cpu=False
    parameters=[p for p in model.parameters() if p.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate,weight_decay=1e-6)
    cache=OrderedDict();official_cache=OrderedDict();coverage=Counter();samples=[];validation=[];rejects=[]
    official_coverage=Counter();official_gradient_coverage=Counter();timeline_coverage={}
    report={'status':'RUNNING','architecture':model.architecture_id,'model_config':config,'completed_updates':0,
        'training_trajectories':len(train),'development_trajectories':len(dev),'trainable_parameters':sum(p.numel() for p in parameters),
        'native_teacher_calls':collection['native_vbd_steps'],'native_solver_calls_in_learning':0,
        'host_activation_offload':False,'dataset_residency':'Whole loaded trajectories remain on device; 400-scene cache',
        'rollout_horizons':'One hop normally, two autonomous hops every tenth update and throughout validation',
        'training_samples':samples,'validation':validation,'rejections':rejects,
        'robot_replay_hops':a.replay_hops,'robot_replay_every_updates':a.replay_every,
        'robot_phase_balancing':a.balance_robot_phases,
        'learning_rate':a.learning_rate,
        'official_training_window_policy':'initial_middle_terminal_hold_v2',
        'reference_self_contact_capacity_override':a.self_contact_capacity,
        'implementation_snapshot_manifest_sha256':hashlib.sha256((a.output/'implementation_snapshot_manifest.json').read_bytes()).hexdigest(),
        'trajectory_coverage':coverage,'source_checkpoint':str(a.checkpoint),'newton_sha':newton_revision()}
    if official:
        report.update(official_manifest=str(a.official_manifest.resolve()),official_manifest_sha256=official_audit_sha,
            official_demo_target_count=42,official_demo_microbatches=official_coverage,
            official_positive_gradient_microbatches=official_gradient_coverage,
            official_timeline_bins=timeline_coverage,official_per_update=a.official_per_update,
            official_reference_storage='CPU trajectories, bounded 2-scene cache; only chosen autonomous window uploaded once; scene geometry resident in cache',
            official_recorded_reference_steps=sum(r['timeline_provenance']['teacher_states']-1 for r in official),
            coverage_scope='Counters record successful finite gradient microbatches. Per-update changed shared weight probes are recorded separately.')
    def write():
        report['gpu']=budget.report();tmp=a.output/'training.tmp';tmp.write_text(json.dumps(report,indent=2,default=str)+'\n');tmp.replace(a.output/'training.json')
    def forbid(*args,**kwargs):raise AssertionError('No native dynamics during supervised learning')
    # Asset rebuilders inspect the original constructor signature. Preserve it.
    import functools
    def forbidden_constructor(original):
        @functools.wraps(original)
        def guard(*args,**kwargs):return forbid(*args,**kwargs)
        return guard
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=forbidden_constructor(cls.__init__);cls.step=forbid
    def load(row):
        key=row['dataset_id']
        storage=official_cache if row.get('full_demo_timeline') else cache
        if key in storage:storage.move_to_end(key);return storage[key]
        while len(storage)>=(2 if row.get('full_demo_timeline') else 400):storage.popitem(last=False)
        if row['loader']=='augmentation':scene,states=load_reference(row['path'],wp,newton,a.device,state_device=a.device)
        else:
            from wp_vbd.multirate_data import load_reference as original
            runtime_row=row
            if row['loader'] in ('self','elastic_rod') and a.self_contact_capacity is not None:
                runtime_row={**row,'self_contact_max_rows':max(a.self_contact_capacity,row.get('self_contact_max_rows',200000))}
            scene,states=original(runtime_row,wp,newton,a.device)
            if not row.get('full_demo_timeline'):states={k:t.to(a.device) for k,t in states.items()}
        scene.metadata.update(compact_numerical_diagnostics=True,elastic_shape_response=True)
        if scene.metadata.get('training_schema')=='authored_franka_substep_reference_v310':scene.metadata['separate_coarse_ray_step']=False
        if scene.metadata['contract']=='elastic_surface_joint_contact_v38':scene.metadata['joint_tree_coarse_modes']='scalar_tree_twelve_columns_v307'
        storage[key]=(scene,states,RolloutObjective(scene));return storage[key]
    def rollout(row,training=True,hops=None):
        material_before=getattr(model,'_analytic_material_total_solves',0)
        loaded_at=time.perf_counter()
        scene,states,objective=load(row);load_seconds=time.perf_counter()-loaded_at
        count=len(states['x']);reference_dt=row.get('reference_dt',scene.metadata['dt'])
        if row.get('full_demo_timeline'):
            occurrence=coverage[row['dataset_id']] if training else official_development_occurrence[row['dataset_id']]
            plan=select_timeline_plan(row['multirate'],occurrence,development=not training,
                complete_motion_sampling=training)
            start=plan['start'];span=sum(plan['strides'])
            window=device_window(states,start,span,a.device)
            current=select(window,slice(0,1));losses=[];network_rounds=0;offset=0;repairs=[]
            for stride,duration in zip(plan['strides'],plan['durations']):
                from wp_vbd.multirate_data import advance as original
                def step(state,actions,dt):return original(model,row,scene,state,actions,dt,wp)
                current,info,repair=advance_reference_span(current,window,offset,stride,reference_dt,step)
                repairs.append(repair)
                target=select(window,slice(offset+stride,offset+stride+1))
                value,terms,metrics=objective(current,target,info);losses.append(value)
                network_rounds+=repair['neural_rounds']
                offset+=stride
            return sum(losses)/len(losses),{'hops':len(losses),'start':start,'strides':plan['strides'],
                'durations':plan['durations'],'neural_rounds':network_rounds,'phase_bin':plan['phase_bin'],
                'teacher_end':start+span,'reference_seconds':row['timeline_provenance']['physical_seconds'],
                'demo':row['demo'],'robot_phase':None,
                'reference_load_seconds':load_seconds,
                'adaptive_repairs':repairs,
                'material_reduced_solves':getattr(model,'_analytic_material_total_solves',0)-material_before}
        hops=(1 if training and not a.preflight else 2) if hops is None else hops
        # Two autonomous student hops, mixed physical durations. No reset to
        # teacher state between hops; source inputs remain on device.
        qualified=row.get('multirate',{}).get('training' if training else 'development',{}).get('windows',[])
        if training:
            options=(1,1,2,3) if row['loader']=='augmentation' else (3,4,10)
            if qualified:
                longest=max(sum(plan['strides']) for plan in qualified)
                options=tuple(stride for stride in options if hops*stride<=longest)
                if not options:raise RuntimeError('Requested autonomous horizon exceeds qualified reference windows')
            stride=rng.choice(options)
        else:
            stride=(1,2,3)[row.get('seed',0)%3] if row['loader']=='augmentation' else 4
        start=rng.randrange(max(1,count-hops*stride-1)) if training else max(0,min(count-hops*stride-1,count//2))
        phase=None
        eligible=[plan for plan in qualified if sum(plan['strides'])>=hops*stride]
        if qualified:
            if not eligible:raise RuntimeError('Requested autonomous horizon exceeds qualified reference windows')
            start=(rng.choice(eligible) if training else eligible[len(eligible)//2])['start']
        if training and a.balance_robot_phases and scene.metadata.get('training_schema')=='authored_franka_substep_reference_v310':
            origin=scene.metadata['source_run']['initial_command_time']
            choices=[]
            for label,lower,upper in [('pinch',1.3,2.1),('lift',2.1,3.3),('hold',3.3,3.8),('release',3.8,4.5)]:
                first=max(0,int(round((lower-origin)/reference_dt)))
                last=min(count-hops*stride-1,int(round((upper-origin)/reference_dt))-hops*stride)
                if last>=first:
                    starts=[plan['start'] for plan in eligible if first<=plan['start']<=last] if qualified else None
                    if starts or not qualified:choices.append((label,first,last,starts))
            if choices:
                phase,first,last,starts=rng.choice(choices)
                start=rng.choice(starts) if starts else rng.randint(first,last)
        current=select(states,slice(start,start+1));losses=[];network_rounds=0
        for hop in range(hops):
            j=start+hop*stride
            actions=inputs_for_span(states,j,stride,durations=[reference_dt]*stride)
            if row['loader']=='augmentation':current,info=advance(model,scene,current,actions,stride*reference_dt,wp)
            else:
                from wp_vbd.multirate_data import advance as original
                current,info=original(model,row,scene,current,actions,stride*reference_dt,wp)
            if not bool(info['accepted'].all()):raise RuntimeError('Rejected student proposal')
            target=select(states,slice(j+stride,j+stride+1))
            value,terms,metrics=objective(current,target,info);losses.append(value)
            network_rounds+=sum(r.get('phase')=='neural_correction' for r in info['iterations'])
        return sum(losses)/hops,{'hops':hops,'start':start,'stride':stride,'dt':stride*reference_dt,
            'neural_rounds':network_rounds,'robot_phase':phase,
            'reference_load_seconds':load_seconds,
            'material_reduced_solves':getattr(model,'_analytic_material_total_solves',0)-material_before}
    chosen_dev=[next((r for r in dev if r['family']==family),None) for family in FAMILIES]
    if a.preflight:chosen_dev=[]
    def validate(step,full=False):
        model.eval();rows=[]
        with torch.no_grad():
            for row in (dev if full else chosen_dev):
                if row is None:continue
                try:
                    value,detail=rollout(row,False)
                    number=float(value)
                    if not math.isfinite(number):raise RuntimeError('Nonfinite development loss')
                    rows.append({'dataset':row['dataset_id'],'loss':number,'detail':detail})
                except RuntimeError as error:rows.append({'dataset':row['dataset_id'],'rejected':str(error)})
        validation.append({'update':step,'suite':'all_30_held_out' if full else 'fixed_10_development','windows':rows});write()
        if (official and not a.preflight and not full and
            (step==0 or step%a.official_validation_interval==0 or step==a.updates)):
            official_results=[]
            with torch.no_grad():
                for row in official:
                    try:
                        value,detail=rollout(row,False)
                        number=float(value)
                        if not math.isfinite(number):raise RuntimeError('Nonfinite official development loss')
                        official_results.append({'dataset':row['dataset_id'],'demo':row['demo'],'loss':number,'detail':detail})
                    except RuntimeError as error:
                        official_results.append({'dataset':row['dataset_id'],'demo':row['demo'],'rejected':str(error)})
            validation.append({'update':step,'suite':'fixed_42_official_temporal_development',
                'scope':'Same 42 fixed reserved windows at every comparison; temporal development of seen assets only',
                'windows':official_results});write()
    replay=[]
    if a.replay_manifest:
        replay=[r for r in json.loads(a.replay_manifest.read_text())['scenes'] if r['loader']=='surface']
        if official:
            official_demos={row['demo'] for row in official}
            duplicates=[row['dataset_id'] for row in replay if row.get('demo') in official_demos]
            replay=[row for row in replay if row.get('demo') not in official_demos]
            report['replay_excluded_duplicate_official_datasets']=duplicates
        report['active_replay_datasets']=[row['dataset_id'] for row in replay]
    schedule=list(train);rng.shuffle(schedule);cursor=0
    official_schedule=[row for row in official for phase in range(row['multirate']['phase_bins'])]
    official_cursor=0
    start_update=0
    def save_recovery(step):
        atomic_save({'schema':'analytic_corrector_completed_update_recovery_v1','model':model.state_dict(),
            'model_config':config,'step':step,'optimizer':optimizer.state_dict(),
            'report_json':json.dumps(report,default=str),'rng_state':rng.getstate(),
            'torch_rng_state':torch.get_rng_state(),
            'cuda_rng_state':torch.cuda.get_rng_state_all() if str(a.device).startswith('cuda') else [],
            'schedule_ids':[row['dataset_id'] for row in schedule],'cursor':cursor,'official_cursor':official_cursor,
            'collection_sha256':hashlib.sha256(a.collection.read_bytes()).hexdigest(),
            'replay_manifest_sha256':hashlib.sha256(a.replay_manifest.read_bytes()).hexdigest() if a.replay_manifest else None,
            'official_manifest_sha256':official_audit_sha if official else None,
            'learning_rate':a.learning_rate},a.output/'recovery.pt')
    if a.resume:
        recovery=torch.load(a.resume,map_location='cpu',weights_only=True)
        expected={'model_config':config,'collection_sha256':hashlib.sha256(a.collection.read_bytes()).hexdigest(),
            'replay_manifest_sha256':hashlib.sha256(a.replay_manifest.read_bytes()).hexdigest() if a.replay_manifest else None,
            'official_manifest_sha256':official_audit_sha if official else None,'learning_rate':a.learning_rate}
        if recovery.get('schema')!='analytic_corrector_completed_update_recovery_v1' or any(recovery.get(k)!=v for k,v in expected.items()):
            raise ValueError('Recovery model, data, replay and optimizer configuration must match')
        model.load_state_dict(recovery['model'],strict=True);optimizer.load_state_dict(recovery['optimizer'])
        previous=json.loads(recovery['report_json']);start_update=recovery['step']
        if previous.get('official_training_window_policy')!=report['official_training_window_policy']:
            report['training_window_policy_migration']={'previous':previous.get('official_training_window_policy','bin_beginning_v1'),
                'current':report['official_training_window_policy'],'at_completed_update':start_update,
                'scope':'Training chooses existing valid initial/middle/terminal windows; fixed development intervals and all source bytes unchanged'}
        if previous.get('reference_self_contact_capacity_override')!=a.self_contact_capacity:
            report['reference_query_capacity_migration']={'previous':previous.get('reference_self_contact_capacity_override'),
                'current':a.self_contact_capacity,'at_completed_update':start_update,
                'scope':'Capacity only: retain the same complete pairs, geometric envelope, filters, materials and source bytes; no truncated snapshot is accepted'}
        policy={'robot_replay_hops':a.replay_hops,'robot_replay_every_updates':a.replay_every,
                'robot_phase_balancing':a.balance_robot_phases,'official_per_update':a.official_per_update if official else None}
        if any(previous.get(key)!=value for key,value in policy.items()):
            raise ValueError('Recovery autonomous horizons, phase balance and optimizer sampling policy must match')
        if previous['completed_updates']!=start_update or [s['update'] for s in previous['training_samples']]!=list(range(1,start_update+1)):
            raise ValueError('Recovery requires an intact completed optimizer prefix')
        samples.extend(previous['training_samples']);validation.extend(previous['validation']);rejects.extend(previous['rejections'])
        coverage.update(previous['trajectory_coverage']);official_coverage.update(previous.get('official_demo_microbatches',{}))
        official_gradient_coverage.update(previous.get('official_positive_gradient_microbatches',{}));timeline_coverage.update(previous.get('official_timeline_bins',{}))
        by_id={row['dataset_id']:row for row in train};schedule=[by_id[key] for key in recovery['schedule_ids']]
        if sorted(recovery['schedule_ids'])!=sorted(by_id):raise ValueError('Recovery augmentation schedule changed')
        cursor=recovery['cursor'];official_cursor=recovery['official_cursor'];rng.setstate(recovery['rng_state'])
        torch.set_rng_state(recovery['torch_rng_state'])
        if str(a.device).startswith('cuda'):torch.cuda.set_rng_state_all(recovery['cuda_rng_state'])
        report.update(completed_updates=start_update,resume_from=str(a.resume.resolve()),
            resumed_prefix_updates=start_update,resumed_prefix_snapshot=previous.get('implementation_snapshot_manifest_sha256'))
        del recovery
    write()
    if not a.resume:validate(0)
    if not a.preflight:save_recovery(start_update)
    try:
        preflight_schedule=[*train,*official]
        if a.preflight_official_only:preflight_schedule=list(official)
        if a.preflight_demo:
            requested=set(a.preflight_demo)
            if requested-{r.get('demo') for r in official}:raise ValueError('Unknown requested preflight demo')
            preflight_schedule=[row for row in official if row['demo'] in requested]
        for update in range(start_update+1,(len(preflight_schedule) if a.preflight else a.updates)+1):
            began=time.perf_counter();optimizer.zero_grad(set_to_none=True);model.train();details=[]
            group=[]
            if a.preflight:
                group=[preflight_schedule[update-1]]
            else:
                for micro in range(2):
                    if cursor==len(schedule):rng.shuffle(schedule);cursor=0
                    group.append(schedule[cursor]);cursor+=1
                if replay and update%a.replay_every==0:group.append(replay[(update//a.replay_every-1)%len(replay)])
                for micro in range(a.official_per_update if official else 0):
                    group.append(official_schedule[official_cursor%len(official_schedule)]);official_cursor+=1
            for row in group:
                for retry in range(8):
                    try:
                        hops=(a.replay_hops if row['loader']!='augmentation' and a.replay_hops else
                              (2 if update%10==0 or a.preflight else 1))
                        value,detail=rollout(row,hops=hops);break
                    except RuntimeError as error:
                        rejects.append({'update':update,'dataset':row['dataset_id'],'error':str(error)})
                        if row.get('full_demo_timeline') or 'out of memory' in str(error).lower() or retry==7:raise
                if not bool(torch.isfinite(value)):raise ValueError('Nonfinite loss')
                probes={name:param for name,param in model.named_parameters() if name in (
                    'protected_response.hierarchy.encoder.0.query.weight',
                    'protected_response.coarse_response.readout.weight',
                    'terminal_velocity.shared_readout.weight')}
                before={name:None if param.grad is None else param.grad.detach().clone() for name,param in probes.items()}
                (value/len(group)).backward()
                detail['microbatch_gradient_probes']={name:float((param.grad if before[name] is None else param.grad-before[name]).norm())
                    for name,param in probes.items() if param.grad is not None}
                detail['positive_measured_gradient']=any(v>0 for v in detail['microbatch_gradient_probes'].values())
                coverage[row['dataset_id']]+=1
                if row.get('full_demo_timeline'):
                    official_coverage[row['demo']]+=1
                    if detail['positive_measured_gradient']:official_gradient_coverage[row['demo']]+=1
                    bins=timeline_coverage.setdefault(row['demo'],{})
                    key=str(detail['phase_bin']);bins[key]=bins.get(key,0)+1
                details.append({'dataset':row['dataset_id'],'loss':float(value.detach()),**detail});del value
            norm=clip_gradient_norm_fp64(parameters,1.)
            if not bool(torch.isfinite(norm)) or float(norm)==0:raise ValueError('Missing shared supervision gradient')
            probe_name,probe_parameter=max(((name,param) for name,param in probes.items() if param.grad is not None),
                key=lambda item:float(item[1].grad.abs().max()))
            probe_index=int(probe_parameter.grad.abs().flatten().argmax())
            probe_before=float(probe_parameter.detach().flatten()[probe_index])
            if not a.preflight:optimizer.step()
            changed=bool(float(probe_parameter.detach().flatten()[probe_index])!=probe_before)
            shared_probe={name:float(param.grad.norm()) for name,param in model.protected_response.named_parameters()
                if param.grad is not None and (name.endswith('coarse_response.readout.weight') or name.endswith('encoder.0.query.weight'))}
            samples.append({'update':update,'shared_gradient_probes':shared_probe,'loss':sum(d['loss'] for d in details)/len(details),
                'gradient_norm':float(norm),'microbatches':details,'seconds':time.perf_counter()-began})
            samples[-1]['actual_shared_weight_probe_changed']=changed
            samples[-1]['changed_weight_probe_parameter']=probe_name
            report['completed_updates']=0 if a.preflight else update;budget.observe();write()
            print(json.dumps(samples[-1]),flush=True)
            if not a.preflight and (update%a.validation_interval==0 or update==a.updates):
                validate(update)
                atomic_save({'model':model.state_dict(),'model_config':config,'step':update,
                    'architecture':model.architecture_id,'training_summary':{'independent_training_trajectories':len(train),
                     'seen_trajectories':len(coverage),'host_activation_offload':False,
                     'official_demo_microbatches':dict(official_coverage),'official_timeline_bins':timeline_coverage,
                     'official_positive_gradient_microbatches':dict(official_gradient_coverage),
                     'official_manifest_sha256':official_audit_sha if official else None}},a.output/f'step_{update:06d}.pt')
            if not a.preflight and (update%a.recovery_interval==0 or update==a.updates):save_recovery(update)
        if not a.preflight:validate(a.updates,full=True)
        report['status']='PREFLIGHT_PASS' if a.preflight else 'COMPLETED';write()
    except BaseException as error:
        report.update(status='FAILED',failure=repr(error));write();raise


if __name__=='__main__':main()
