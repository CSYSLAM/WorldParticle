"""Fine-tune the shared backbone at 3 substeps/frame and exactly one NN per step."""
import argparse
from collections import OrderedDict
import functools
import hashlib
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
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--replay-manifest',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--updates',type=int,default=60)
    p.add_argument('--hops',type=int,default=24)
    p.add_argument('--truncate',type=int,default=4)
    p.add_argument('--validation-interval',type=int,default=10)
    p.add_argument('--validation-hops',type=int,default=18)
    p.add_argument('--patience',type=int,default=3)
    p.add_argument('--learning-rate',type=float,default=1e-5)
    p.add_argument('--analytic-predictor-rounds',type=int,choices=(1,2),default=2)
    p.add_argument('--material-supervision-weight',type=float,default=0.,
                   help='Training-only analytic labels at current predicted states; requires a trained material head')
    p.add_argument('--freeze-terminal-velocity',action='store_true',help='Keep endpoint velocity readout fixed during material/prefix experiments')
    p.add_argument('--gpu-memory-gib',type=float,default=16.)
    p.add_argument('--preflight',action='store_true')
    a=p.parse_args()
    if min(a.hops,a.truncate,a.validation_hops,a.validation_interval,a.patience,a.updates)<1:p.error('Positive budgets required')
    if not math.isfinite(a.material_supervision_weight) or a.material_supervision_weight<0:p.error('Nonnegative finite supervision weight required')
    a.output.mkdir(parents=True,exist_ok=False)
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    import torch
    import warp as wp
    import newton
    from wp_vbd.gpu_budget import GpuBudget
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.mixed_experiment import select,clip_gradient_norm_fp64
    from wp_vbd.multirate_objective import RolloutObjective
    from wp_vbd.cloth_rollout_objective import ClothRolloutObjective
    from wp_vbd.multirate_data import load_reference,advance
    from wp_vbd.multirate_window import inputs_for_span
    from wp_vbd.augmentation_data import load_reference as load_augmentation,advance as advance_augmentation
    from checkpoint_storage import atomic_save
    torch.set_num_threads(1);torch.manual_seed(20261120);torch.use_deterministic_algorithms(True)
    rng=random.Random(20261120);budget=GpuBudget(a.device,a.gpu_memory_gib);budget.trim_unused_cache=True
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(a.device)
    if a.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    manifest=json.loads(a.manifest.read_text())
    if manifest['status']!='COMPLETE' or manifest['newton_sha']!=newton_revision():raise ValueError('Complete teacher manifest required')
    train=[r for r in manifest['scenes'] if r['split']=='training']
    dev=[r for r in manifest['scenes'] if r['split']=='development']
    if len(train)<2 or len(dev)<1 or {r['sha256'] for r in train}&{r['sha256'] for r in dev}:raise ValueError('Independent teacher assets required')
    collection=json.loads(a.collection.read_text())
    old_train=[r for r in collection['scenes'] if r['split']=='training']
    old_dev=[r for r in collection['scenes'] if r['split']=='development']
    families=sorted({r['family'] for r in old_train})
    by_family={name:[r for r in old_train if r['family']==name] for name in families}
    for rows in by_family.values():rng.shuffle(rows)
    old_train=[rows[i] for i in range(max(map(len,by_family.values())))
               for rows in by_family.values() if i<len(rows)]
    robot_replay=[]
    if a.replay_manifest:
        robot_replay=[r for r in json.loads(a.replay_manifest.read_text())['scenes'] if r['loader']=='surface' and r.get('is_authored_robot',False)]
        if not robot_replay:
            robot_replay=[r for r in json.loads(a.replay_manifest.read_text())['scenes'] if r['loader']=='surface' and 'franka' in r.get('dataset_id','')]
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=dict(saved.get('model_config') or saved['config']['model'])
    config.update(analytic_predictor_rounds=a.analytic_predictor_rounds,unroll=1,rod_unroll=1,
        mimic_metric_tangent=True,contact_aware_warm_start=True,periodic_bending=True)
    if config.get('shared_material_correction',False):
        config.update(shared_material_supervision_weight=a.material_supervision_weight,
            shared_material_analytic_other_contracts=True,omit_final_material_correction=False,
            shared_material_use_learned=True)
    elif a.material_supervision_weight:
        raise ValueError('Material supervision requires a trained shared material head checkpoint')
    model=MixedSolver(config).to(a.device);model.load_state_dict(saved['model'],strict=True)
    del saved
    for name,param in model.named_parameters():param.requires_grad_(name.startswith(('protected_response.','terminal_velocity.','shared_material_head.')))
    if a.freeze_terminal_velocity and model.terminal_velocity is not None:
        model.terminal_velocity.requires_grad_(False)
    model.step_saved_tensors_on_cpu=False;model.protected_response.checkpoint_inputs_on_cpu=False
    parameters=[param for param in model.parameters() if param.requires_grad]
    optimizer=torch.optim.AdamW(parameters,lr=a.learning_rate,weight_decay=1e-6)
    # No native solver may be built or stepped by any reference rebuilder.
    def fail(*args,**kwargs):raise AssertionError('Native dynamics forbidden in fine-tuning')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        original=cls.__init__
        cls.__init__=functools.wraps(original)(lambda *args,**kwargs:fail())
        cls.step=fail
    snapshot=a.output/'implementation_snapshot';snapshot.mkdir()
    hashes={}
    for source in [Path(__file__),*(ROOT/'src/wp_vbd').glob('*.py')]:
        rel=source.relative_to(ROOT);dst=snapshot/rel;dst.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,dst);hashes[str(rel)]=hashlib.sha256(dst.read_bytes()).hexdigest()
    (a.output/'implementation_snapshot_manifest.json').write_text(json.dumps(hashes,indent=2)+'\n')
    cache=OrderedDict()
    def load(row):
        key=row['dataset_id']
        if key in cache:cache.move_to_end(key);return cache[key]
        if len(cache)>=24:cache.popitem(last=False)
        if row['loader']=='augmentation':scene,states=load_augmentation(row['path'],wp,newton,a.device,state_device=a.device)
        else:
            scene,states=load_reference(row,wp,newton,a.device)
            states={k:v.to(a.device) for k,v in states.items()}
        scene.metadata.update(compact_numerical_diagnostics=True,elastic_shape_response=True)
        if row['loader']!='robot_cloth' and scene.metadata.get('contract')=='elastic_surface_joint_contact_v38':
            scene.metadata['joint_tree_coarse_modes']='scalar_tree_twelve_columns_v307'
            scene.metadata['separate_coarse_ray_step']=False
        objective=ClothRolloutObjective(scene) if row['loader']=='robot_cloth' else RolloutObjective(scene)
        cache[key]=(scene,states,objective)
        return cache[key]
    def rollout(row,start,hops,*,training,weight=1.):
        scene,states,objective=load(row);dt=row.get('reference_dt',scene.metadata['dt'])
        # Exact 4 fine teacher intervals = one deployed physical step.
        stride=4 if row['loader']=='robot_cloth' else max(1,round((1/180)/dt))
        if start+hops*stride>=len(states['x']):raise ValueError('Rollout outside teacher timeline')
        current=select(states,slice(start,start+1));loss=0.;chunk=[];nn=0;last_metrics={}
        for hop in range(hops):
            j=start+hop*stride;actions=inputs_for_span(states,j,stride,durations=[dt]*stride)
            if row['loader']=='augmentation':current,info=advance_augmentation(model,scene,current,actions,stride*dt,wp)
            else:current,info=advance(model,row,scene,current,actions,stride*dt,wp)
            if not bool(info['accepted'].all()):raise RuntimeError('Rejected fixed-budget student step')
            rounds=sum(item.get('phase')=='neural_correction' for item in info['iterations'])
            if rounds!=1:raise AssertionError('Exactly one backbone call per student step required')
            nn+=rounds;target=select(states,slice(j+stride,j+stride+1))
            value,terms,metrics=objective(current,target,info)
            if training and 'shared_material_supervision' in info:
                value=value+a.material_supervision_weight*info['shared_material_supervision']
                metrics['shared_material_supervision']=info['shared_material_supervision']
            if not bool(torch.isfinite(value)):raise ValueError('Nonfinite supervised objective')
            loss+=float(value.detach())/hops;chunk.append(value*weight/hops)
            last_metrics={k:float(v.detach()) for k,v in metrics.items()}
            if training and (len(chunk)==a.truncate or hop==hops-1):
                sum(chunk).backward();chunk=[]
                # Keep accepted predicted values/history; truncate only gradients.
                current=current.detach()
            elif not training:chunk=[]
        return {'dataset':row['dataset_id'],'loss':loss,'start':start,'hops':hops,
                'dt':stride*dt,'neural_rounds':nn,'terminal_metrics':last_metrics}
    report={'status':'RUNNING','source_checkpoint':str(a.checkpoint.resolve()),
            'source_checkpoint_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
            'newton_sha':newton_revision(),'model_config':config,'completed_updates':0,
            'trainable_parameters':sum(param.numel() for param in parameters),
            'training':[],'validation':[],'native_solver_calls_in_training':0,
            'budget':f'3 substeps/frame, {a.analytic_predictor_rounds} analytic predictors + 1 neural correction; no adaptive extra NN calls in training',
            'material_supervision_weight':a.material_supervision_weight,
            'terminal_velocity_frozen':a.freeze_terminal_velocity,
            'material_head_scope':'surface/joint v38; other replay contracts explicitly retain analytic material correction',
            'collection_manifest_sha256':hashlib.sha256(a.collection.read_bytes()).hexdigest(),
            'gradient_horizon':a.truncate,'autonomous_horizon':a.hops,
            'targeted_training_trajectories':len(train),'targeted_development_trajectories':len(dev),
            'replay_robot_trajectories':len(robot_replay),'teacher_manifest_sha256':hashlib.sha256(a.manifest.read_bytes()).hexdigest()}
    def write():
        from wp_vbd.shared_material_correction import statistics
        report['material_correction']=statistics(model)
        report['gpu']=budget.report();tmp=a.output/'training.tmp'
        tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output/'training.json')
    def validate(update):
        model.eval();rows=[]
        with torch.no_grad():
            for row in dev:
                for now in (.05,.65,1.4,2.3,2.55):
                    rows.append(rollout(row,round(now*720),a.validation_hops,training=False))
            # Fixed old-family windows are a regression signal, not model selection on training examples.
            replay=[]
            for family in sorted({r['family'] for r in old_dev}):
                row=next(r for r in old_dev if r['family']==family)
                _,states,_=load(row)
                replay.append(rollout(row,len(states['x'])//2,2,training=False))
        record={'update':update,'cloth_loss':sum(r['loss'] for r in rows)/len(rows),
                'replay_loss':sum(r['loss'] for r in replay)/len(replay),'cloth_windows':rows,'replay_windows':replay}
        report['validation'].append(record);budget.observe();write()
        print(json.dumps({'validation':update,'cloth_loss':record['cloth_loss'],'replay_loss':record['replay_loss']}),flush=True)
        return record
    def save(update,name):
        atomic_save({'model':model.state_dict(),'model_config':config,'step':update,
                     'architecture':model.architecture_id,'training_summary':report},a.output/name)
        atomic_save({'optimizer':optimizer.state_dict(),'completed_updates':update,
                     'rng_state':rng.getstate(),'torch_rng_state':torch.get_rng_state(),
                     'cuda_rng_state':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else [],
                     'matching_checkpoint':name,'model_config':config},a.output/(name+'.training_state.pt'))
    try:
        baseline=validate(0);best=baseline['cloth_loss'];stale=0
        for update in range(1,(1 if a.preflight else a.updates)+1):
            begin=time.perf_counter();model.train();optimizer.zero_grad(set_to_none=True)
            row=train[(update-1)%len(train)]
            # Cyclic phase coverage, with half the windows in late lift/hold.
            phases=[(.0,.25),(.35,.85),(.85,1.4),(1.4,1.9),(2.,2.3),(2.3,2.6)]
            lower,upper=phases[((update-1)+(update-1)//len(train))%len(phases)]
            start=4*round(rng.uniform(lower,upper)*180)
            details=[rollout(row,start,a.hops,training=True,weight=.75)]
            old=old_train[(update-1)%len(old_train)]
            _,states,_=load(old);start=rng.randrange(max(1,len(states['x'])-8))
            details.append(rollout(old,start,2,training=True,weight=.25))
            if robot_replay and update%4==0:
                old=robot_replay[(update//4-1)%len(robot_replay)]
                _,states,_=load(old);dt=old.get('reference_dt',load(old)[0].metadata['dt'])
                stride=max(1,round((1/180)/dt))
                qualified=old.get('multirate',{}).get('training',{}).get('windows',[])
                starts=[r['start'] for r in qualified if sum(r['strides'])>=2*stride]
                if not starts:starts=old.get('qualified_train_starts',[])
                if not starts:raise ValueError('Robot replay requires its qualified training partition')
                start=rng.choice(starts)
                details.append(rollout(old,start,2,training=True,weight=.25))
            probes={name:param for name,param in model.named_parameters() if name in (
                'protected_response.hierarchy.encoder.0.query.weight',
                'protected_response.coarse_response.readout.weight','terminal_velocity.shared_readout.weight',
                'shared_material_head.readout.4.weight','shared_material_head.local_readout.weight')}
            gradients={name:float(param.grad.norm()) for name,param in probes.items() if param.grad is not None}
            norm=clip_gradient_norm_fp64(parameters,1.)
            if not math.isfinite(float(norm)) or not gradients or not any(v>0 for v in gradients.values()):raise ValueError('No finite shared gradient')
            if not a.preflight:optimizer.step()
            report['completed_updates']=0 if a.preflight else update
            report['training'].append({'update':update,'microbatches':details,'gradient_norm':float(norm),
                'shared_gradient_probes':gradients,'seconds':time.perf_counter()-begin})
            budget.observe();write();print(json.dumps(report['training'][-1]),flush=True)
            if a.preflight:break
            if update%a.validation_interval==0 or update==a.updates:
                result=validate(update);save(update,'latest.pt')
                # Old-family loss must remain reasonably close to the frozen parent's.
                admissible=result['replay_loss']<=max(1.3*baseline['replay_loss'],baseline['replay_loss']+.05)
                if result['cloth_loss']<best*.995 and admissible:
                    best=result['cloth_loss'];stale=0;report['selected_update']=update;save(update,'selected.pt')
                else:stale+=1
                if stale>=a.patience:
                    report['stop_reason']='Independent development loss reached a plateau or replay regressed';break
        report['status']='PREFLIGHT_PASS' if a.preflight else 'COMPLETE';write()
    except BaseException as exc:
        report['status']='FAILED';report['failure']=repr(exc);write();raise


if __name__=='__main__':main()
