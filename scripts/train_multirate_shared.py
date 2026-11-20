"""Train and monitor a bounded multirate shared-response pilot."""

import argparse
from collections import OrderedDict
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

from checkpoint_storage import atomic_save
from train_normal_shared_648 import sha, write
from wp_vbd.newton_source import activate_newton_source, newton_revision


def make_cases(manifest,datasets=None):
    rows=[row for row in manifest['scenes']
          if row.get('multirate',{}).get('training',{}).get('windows')
          and row.get('multirate',{}).get('development',{}).get('windows')]
    if datasets:
        selected=set(datasets)
        rows=[row for row in rows if row['dataset_id'] in selected]
        missing=selected-{row['dataset_id'] for row in rows}
        if missing:
            raise ValueError('Requested datasets have no eligible long windows: '+', '.join(sorted(missing)))
    if not rows:
        raise ValueError('No training datasets have long-step windows in both splits')
    return rows


def development_plans(row,count):
    windows=row['multirate']['development']['windows']
    # Spread fixed diagnostics over phases and requested time schedules.
    if len(windows)<=count:return windows
    patterns=list(dict.fromkeys(tuple(p['requested_durations']) for p in windows))
    chosen=[]
    for index in range(count):
        group=[p for p in windows if tuple(p['requested_durations'])==patterns[index%len(patterns)]]
        chosen.append(group[round(index*(len(group)-1)/max(count-1,1))])
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'reports/multirate_course_v1/data_manifest.json')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--optimizer-checkpoint',type=Path,help='Migrate Adam moments paired with the exact starting weights')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--updates', type=int, default=100)
    parser.add_argument('--validation-interval', type=int, default=20)
    parser.add_argument('--device', default='cuda:0', choices=('cpu', 'cuda:0'))
    parser.add_argument('--gpu-memory-gib', type=float, default=12.)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--direction-learning-rate',type=float,default=5e-5)
    parser.add_argument('--analytic-prior-fraction',type=float,default=0.,
                        help='Residual-projection coefficient mixed into the learned direction; use 0 for learned-only calibration')
    parser.add_argument('--learned-coefficient-scale',type=float,default=1.,
                        help='Scale the neural coefficient logits before mixing the small analytic prior')
    parser.add_argument('--reset-elastic-head',action='store_true',
                        help='Initialize a new elastic head when its feature contract changes')
    parser.add_argument('--random-init-elastic-head',action='store_true',
                        help='Use a small random final layer for a reset head so physical loss can train without a prior')
    parser.add_argument('--local-elastic-heads',type=int,default=2,choices=(0,1,2),
                        help='Add learned material-local response columns alongside the pretrained global column')
    parser.add_argument('--learned-contact-memory',action=argparse.BooleanOptionalAction,default=True,
                        help='Train a Coulomb-bounded tangent force seed with accepted surface-contact memory')
    parser.add_argument('--trainable-scope',choices=('shared','elastic-direction','learned-physics-heads'),default='learned-physics-heads',
                        help='Train only learned elastic/contact heads, only the elastic head, or the shared backbone too')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--datasets',nargs='*',help='Limit to exact dataset IDs in the prepared manifest')
    parser.add_argument('--rounds',type=int,choices=(1,2,3),default=3,help='Physical response rounds per student step')
    parser.add_argument('--checkpoint-interval',type=int,default=200,help='Retain a separate weights-only snapshot at this update interval')
    parser.add_argument('--validation-windows',type=int,default=3,help='Fixed development windows per dataset')
    parser.add_argument('--scene-cache-size',type=int,default=2,help='Maximum resident scene/reference pairs')
    parser.add_argument('--host-saved-tensors',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--preflight-only',action='store_true',help='Run fixed validation and one backward graph per dataset without updating weights')
    args = parser.parse_args()
    if min(args.updates,args.validation_interval,args.checkpoint_interval,args.validation_windows,args.scene_cache_size)<1:
        parser.error('Positive update, validation and rollout budgets required')
    if not math.isfinite(args.learning_rate) or args.learning_rate<=0:
        parser.error('Positive finite learning rate required')
    if not math.isfinite(args.direction_learning_rate) or args.direction_learning_rate<=0:
        parser.error('Positive finite direction learning rate required')
    if not math.isfinite(args.analytic_prior_fraction) or not 0<=args.analytic_prior_fraction<.5:
        parser.error('Analytic prior fraction must be finite and in [0, .5)')
    if not math.isfinite(args.learned_coefficient_scale) or args.learned_coefficient_scale<=0:
        parser.error('Learned coefficient scale must be positive and finite')
    args.output.mkdir(parents=True, exist_ok=args.resume)
    report_path = args.output / 'training.json'
    if args.resume and not report_path.is_file():
        parser.error('Resume requires an existing training report')
    manifest = json.loads(args.manifest.read_text())
    activate_newton_source()
    if newton_revision() != manifest['newton_sha']:
        raise ValueError('Use the Newton revision pinned by the input manifest')

    import torch
    import warp as wp
    import newton
    from wp_vbd.gpu_budget import GpuBudget
    from wp_vbd.gpu_ownership import claim_cuda
    from wp_vbd.mixed_experiment import clip_gradient_norm_fp64, select
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.multirate_objective import RolloutObjective, WEIGHTS
    from wp_vbd.multirate_data import advance as advance_model,load_reference
    from wp_vbd.multirate_window import inputs_for_span
    from wp_vbd.training_window import transfer_window

    gpu_ownership = claim_cuda(args.device, ROOT)
    torch.set_num_threads(1)
    torch.manual_seed(20261118)
    torch.use_deterministic_algorithms(True)
    rng = random.Random(20261118)
    budget = GpuBudget(args.device, args.gpu_memory_gib)
    wp.config.kernel_cache_dir = str(ROOT / '.local/p0/warp-cache')
    wp.init()
    wp.set_device(args.device)
    if args.device.startswith('cuda'):
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))

    stored = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    checkpoint_step=int(stored.get('step',0))
    parent_step=checkpoint_step if args.optimizer_checkpoint else 0
    starting_model=stored['model']
    config = {**(stored.get('model_config') or stored['config']['model']),
              'unroll': args.rounds, 'rod_unroll': args.rounds,
              'terminal_velocity_correction': True,'learned_elastic_shape_direction':True,
              'elastic_shape_distillation_weight':.5,
              'elastic_shape_analytic_prior_fraction':args.analytic_prior_fraction,
              'elastic_shape_learned_coefficient_scale':args.learned_coefficient_scale,
              'elastic_shape_objective_version':'projected_energy_v1',
              'elastic_shape_feature_version':2,
              'elastic_shape_local_heads':args.local_elastic_heads,
              'learned_contact_tangent_seed':args.learned_contact_memory,
              'contact_tangent_seed_scale':.1,
              'elastic_shape_head_initialization':'normal' if args.random_init_elastic_head else 'zero'}
    if args.random_init_elastic_head and not args.reset_elastic_head:
        parser.error('--random-init-elastic-head requires --reset-elastic-head')
    if args.reset_elastic_head and args.optimizer_checkpoint:
        parser.error('A reset elastic head cannot inherit optimizer moments')
    load_model={key:value for key,value in starting_model.items()
                if not (args.reset_elastic_head and key.startswith('elastic_shape_head.'))}
    model = MixedSolver(config).to(args.device)
    model.step_saved_tensors_on_cpu=args.host_saved_tensors
    model.protected_response.checkpoint_inputs_on_cpu=args.host_saved_tensors
    missing, unexpected = model.load_state_dict(load_model, strict=False)
    allowed_missing=set()
    for prefix,module in (('terminal_velocity',model.terminal_velocity),('elastic_shape_head',model.elastic_shape_head),
                          ('contact_tangent_head',model.contact_tangent_head)):
        keys={prefix+'.'+name for name in module.state_dict()}
        absent=keys & set(missing)
        if absent:
            if prefix=='elastic_shape_head':
                expected={key for key in keys if '.local_network.' in key}
                if absent != expected:
                    if not keys.issubset(missing):
                        raise ValueError('Checkpoint contains an incomplete elastic direction module')
                    allowed_missing|=keys
                else:allowed_missing|=expected
            elif prefix=='contact_tangent_head' and not args.learned_contact_memory:
                if not keys.issubset(missing):raise ValueError('Checkpoint contains an incomplete contact tangent module')
                allowed_missing|=keys
            elif keys.issubset(missing):allowed_missing|=keys
            else:raise ValueError('Checkpoint contains an incomplete module: '+prefix)
    if unexpected or set(missing) != allowed_missing:
        raise ValueError(f'Unexpected checkpoint migration: missing={missing}, unexpected={unexpected}')
    del stored
    for name, param in model.named_parameters():
        prefixes=(('elastic_shape_head.',) if args.trainable_scope=='elastic-direction' else
                  ('elastic_shape_head.','contact_tangent_head.') if args.trainable_scope=='learned-physics-heads' else
                  ('protected_response.','terminal_velocity.','elastic_shape_head.','contact_tangent_head.'))
        param.requires_grad_(name.startswith(prefixes))
    parameters = [p for p in model.parameters() if p.requires_grad]
    direction_parameters=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith(('terminal_velocity.','elastic_shape_head.','contact_tangent_head.'))]
    protected_parameters=[p for name,p in model.named_parameters() if p.requires_grad and name.startswith('protected_response.')]
    optimizer = torch.optim.AdamW([{'params':protected_parameters,'lr':args.learning_rate},
                                   {'params':direction_parameters,'lr':args.direction_learning_rate}],
                                  weight_decay=0., foreach=False)
    optimizer_parent=None
    if args.optimizer_checkpoint:
        if args.resume:parser.error('--optimizer-checkpoint is only for a new migrated run')
        parent=torch.load(args.optimizer_checkpoint,weights_only=True,map_location='cpu')
        if parent.get('step')!=checkpoint_step or parent.get('model',{}).keys()!=starting_model.keys():
            raise ValueError('Optimizer checkpoint and starting weights have different ancestry')
        if any(not torch.equal(parent['model'][name],value) for name,value in starting_model.items()):
            raise ValueError('Optimizer moments are not paired with the starting weights')
        saved=parent.get('optimizer')
        if not isinstance(saved,dict) or not saved.get('param_groups'):
            raise ValueError('Complete original AdamW state required')
        old_ids=[identifier for group in saved['param_groups'] for identifier in group['params']]
        if len(old_ids)!=len(parameters):raise ValueError('Optimizer parameter ordering or coverage changed')
        for parameter,identifier in zip(parameters,old_ids):
            state=saved['state'].get(identifier)
            if state is not None:
                optimizer.state[parameter]={key:(value.to(parameter.device) if torch.is_tensor(value) else value)
                                            for key,value in state.items()}
        optimizer_parent={'path':str(args.optimizer_checkpoint.resolve()),
                          'sha256':sha(args.optimizer_checkpoint),'step':checkpoint_step,
                          'moment_tensors_migrated_by_model_parameter_order':True}
        del parent
    if manifest.get('schema')!='shared_vbd_multirate_data_v1':
        parser.error('Run scripts/prepare_multirate_data.py to create the original-demo window manifest')
    cases = make_cases(manifest,args.datasets)
    signature = {
        'checkpoint_sha256': sha(args.checkpoint), 'manifest_sha256': sha(args.manifest),
        'newton_sha': manifest['newton_sha'], 'learning_rate': args.learning_rate,
        'direction_learning_rate':args.direction_learning_rate,
        'trainable_scope':args.trainable_scope,
        'reset_elastic_head':args.reset_elastic_head,
        'initial_step':parent_step,'optimizer_parent':optimizer_parent,
        'cases': [row['dataset_id'] for row in cases], 'loss_weights': WEIGHTS,
        'model_config':config,
        'runtime':{'host_saved_tensors':args.host_saved_tensors,'validation_windows':args.validation_windows,
                   'scene_cache_size':args.scene_cache_size,'gpu_memory_gib':args.gpu_memory_gib},
        'source_sha256': {path.name: sha(path) for path in (
            Path(__file__), *(ROOT / 'src/wp_vbd').glob('*.py'))},
    }
    report = {
        'status': 'RUNNING', 'configuration': config, 'identity': signature,
        'completed_updates': parent_step, 'samples': [], 'validation': [], 'rejections': [],
        'scope': 'Pinned pure-VBD multirate pilot; fixed windows are development diagnostics.',
    }
    if args.resume:
        report = json.loads(report_path.read_text())
        last = torch.load(args.output / 'last.pt', weights_only=True, map_location='cpu')
        if signature != report['identity'] or signature != last['training_identity']:
            raise ValueError('Training identity changed across resume')
        if last['step'] != report['completed_updates']:
            raise ValueError('Training report and checkpoint step differ')
        model.load_state_dict(last['model'])
        optimizer.load_state_dict(last['optimizer'])
        rng.setstate(last['rng_state'])
        del last

    cache = OrderedDict()

    def load(row, validation):
        label = row['dataset_id']
        use_holdout = validation and bool(row.get('validation_path'))
        identity = label, use_holdout
        if identity in cache:
            cache.move_to_end(identity)
            return cache[identity]
        while len(cache)>=args.scene_cache_size:
            cache.popitem(last=False)
        gc.collect()
        path = Path(row['validation_path' if use_holdout else 'path'])
        if not path.is_absolute(): path=ROOT/path
        if sha(path) != row['validation_sha256' if use_holdout else 'sha256']:
            raise ValueError('Reference file changed: ' + str(path))
        scene,states=load_reference(row,wp,newton,args.device,validation=validation)
        if scene.metadata.get('training_schema')=='authored_franka_substep_reference_v310':
            from wp_vbd.multirate_motion import audit_endpoint_motion
            split='development' if validation else 'training'
            audit=audit_endpoint_motion(path,row['multirate'][split]['windows'],
                                        scene.geometry_oracle.motion_budget)
            if audit.get('supported') and audit['unreachable_macro_hops']:
                raise ValueError(f'{label}: {audit["unreachable_macro_hops"]} {split} macro endpoints exceed '
                    'the real collision-query envelope. Regenerate the manifest with an explicit reachable '
                    'detection_motion_budget before training or evaluation.')
        scene.metadata['learned_elastic_shape_response']=True
        value = (scene, states, RolloutObjective(scene))
        cache[identity] = value
        budget.observe(force=True)
        return value

    def start_for(row, states, validation):
        split='development' if validation else 'training'
        windows=row['multirate'][split]['windows']
        return rng.choice(windows)

    def rollout(row, scene, states, objective, plan, *, learned=True):
        start=plan['start'];pattern=plan['strides'];durations=plan['durations']
        window = transfer_window(states, start, sum(pattern), args.device)
        current = select(window, slice(0, 1))
        diagnostic_rows = []
        losses = []
        direction_losses=[]
        cursor = 0
        for stride,tick in zip(pattern,durations):
            teacher_durations=plan['teacher_durations'][cursor:cursor+stride]
            actions = inputs_for_span(window, cursor, stride,durations=teacher_durations)
            current,info=advance_model(model,row,scene,current,actions,tick,wp,learned=learned)
            if not bool(info['accepted'].all()):
                flags={key:bool(value.all()) for key in ('finite','predictor_motion_valid','predictor_physics_domain_valid')
                       if isinstance(value:=info.get(key),torch.Tensor)}
                raise RejectedWindow(f'{row["dataset_id"]} at {start + cursor}, dt={tick:g}, flags={flags}')
            cursor += stride
            target = select(window, slice(cursor, cursor + 1))
            value, terms, metrics = objective(current, target, info)
            losses.append(value * tick / sum(durations))
            if info.get('elastic_shape_supervision') is not None:
                direction_losses.append(info['elastic_shape_supervision']*tick/sum(durations))
            diagnostic_rows.append({
                'dt_s': tick, 'weighted_loss': float(value.detach()),
                'terms': {k: float(v.detach()) for k, v in terms.items()},
                'metrics': {k: float(v.detach()) for k, v in metrics.items()},
                'elastic_shape_supervision':float(info['elastic_shape_supervision'].detach()) if info.get('elastic_shape_supervision') is not None else None,
                'repair_calls': info.get('contact_repair_calls', 0),
            })
        rollout_loss=sum(losses)
        direction_loss=sum(direction_losses) if direction_losses else rollout_loss.new_zeros(())
        return rollout_loss+model.config['elastic_shape_distillation_weight']*direction_loss, diagnostic_rows

    def persist():
        report['gpu'] = budget.report()
        write(report_path, report)

    def save():
        atomic_save({
            'architecture': model.architecture_id, 'model': model.state_dict(), 'model_config': config,
            'optimizer': optimizer.state_dict(), 'step': report['completed_updates'],
            'rng_state': rng.getstate(), 'training_identity': signature,
        }, args.output / 'last.pt')
        persist()

    def save_weights(name):
        atomic_save({'architecture':model.architecture_id,'model':model.state_dict(),
                     'model_config':config,'step':report['completed_updates'],
                     'training_identity':signature},args.output/name)

    @torch.no_grad()
    def validate(step):
        model.eval()
        result = []
        for row in cases:
            scene,states,objective=load(row,True)
            plans=development_plans(row,args.validation_windows)
            for plan in plans:
                for learned in ((True, False) if step == signature['initial_step'] else (True,)):
                    item = {'dataset': row['dataset_id'], 'start': plan['start'],
                            'durations': plan['durations'],'strides': plan['strides'],
                            'learned': learned}
                    try:
                        value, details = rollout(row, scene, states, objective, plan, learned=learned)
                        item.update(loss=float(value), hops=details)
                    except RejectedWindow as error:
                        item['rejected'] = str(error)
                    result.append(item)
                    print(json.dumps({'validation': item}), flush=True)
        report['validation'].append({'after_update': step, 'windows': result})
        persist()

    persist()
    try:
        if not args.resume:
            save_weights('initial.pt')
        if not report['validation']:
            validate(parent_step)
        if args.preflight_only:
            report['backward_probes']=[]
            for row in cases:
                model.train();optimizer.zero_grad(set_to_none=True)
                scene,states,objective=load(row,False)
                plans=sorted(row['multirate']['training']['windows'],key=lambda w:sum(w['durations']),reverse=True)
                rejected=[]
                for plan in plans[:8]:
                    try:
                        value,details=rollout(row,scene,states,objective,plan)
                        break
                    except RejectedWindow as error:
                        rejected.append(str(error))
                        optimizer.zero_grad(set_to_none=True)
                else:
                    report['backward_probes'].append({'dataset':row['dataset_id'],'rejected':rejected})
                    persist()
                    raise RuntimeError('No accepted preflight window in eight attempts: '+row['dataset_id'])
                value.backward()
                norm=clip_gradient_norm_fp64(parameters,1.)
                if float(norm)==0:raise RuntimeError('No gradient in preflight: '+row['dataset_id'])
                report['backward_probes'].append({'dataset':row['dataset_id'],'loss':float(value.detach()),
                    'gradient_norm':float(norm),'hops':details,'gpu_mib':budget.observe(force=True),
                    'rejected_attempts':rejected})
                del value,norm;optimizer.zero_grad(set_to_none=True);persist()
            report['status']='PREFLIGHT_COMPLETED_NO_OPTIMIZER_UPDATES';persist();return
        for update in range(report['completed_updates'] + 1, args.updates + 1):
            begin = time.perf_counter()
            row = cases[(update - 1) % len(cases)]
            scene, states, objective = load(row, False)
            plan=start_for(row,states,False)
            for attempt in range(8):
                model.train()
                optimizer.zero_grad(set_to_none=True)
                try:
                    value, details = rollout(row, scene, states, objective, plan)
                    break
                except RejectedWindow as error:
                    item = {'update': update, 'start': plan['start'], 'strides': plan['strides'], 'reason': str(error)}
                    report['rejections'].append(item)
                    print(json.dumps({'rejection': item}), flush=True)
                    plan = start_for(row, states, False)
            else:
                raise RuntimeError(f'Eight rejected windows for {row["dataset_id"]} at the same time budget')
            if not bool(torch.isfinite(value)):
                raise RuntimeError('Nonfinite multirate training loss')
            value.backward()
            norm = clip_gradient_norm_fp64(parameters, 1.)
            if float(norm) == 0:
                raise RuntimeError('No gradient in the supervised window')
            head_norms={}
            for prefix,head in (('velocity',model.terminal_velocity),('elastic_shape',model.elastic_shape_head),
                                ('contact_tangent',model.contact_tangent_head)):
                head_norms[prefix]={name:float(param.grad.norm()) for name,param in head.named_parameters()
                                    if param.grad is not None}
            optimizer.step()
            if args.device.startswith('cuda'):
                torch.cuda.synchronize()
            item = {'update': update, 'dataset': row['dataset_id'], 'teacher_start': plan['start'],
                    'strides': plan['strides'],'durations':plan['durations'],
                    'loss': float(value.detach()), 'gradient_norm': float(norm),
                    'module_gradients': head_norms, 'hops': details,
                    'seconds': time.perf_counter() - begin, 'gpu_mib': budget.observe(force=True)}
            report['samples'].append(item)
            report['completed_updates'] = update
            del value, norm
            print(json.dumps({'training': item}), flush=True)
            if update == 1 or update % args.validation_interval == 0 or update == args.updates:
                save()
                validate(update)
            if update%args.checkpoint_interval==0 or update==args.updates:
                save_weights(f'step_{update:06d}.pt')
        report['status'] = 'COMPLETED'
        persist()
    except BaseException as error:
        report.update(status='STOPPED', failure=repr(error))
        persist()
        raise


class RejectedWindow(RuntimeError):
    pass


if __name__ == '__main__':
    main()
