"""Bounded, isolated two-hop fitting study; never overwrites course checkpoints."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cuda:0')
    parser.add_argument('--updates', type=int, default=32)
    parser.add_argument('--index', type=int, default=520)
    parser.add_argument('--development-index', type=int, default=720)
    parser.add_argument('--train-indices', type=int, nargs='+')
    parser.add_argument('--development-indices', type=int, nargs='+')
    parser.add_argument('--learning-rate', type=float, default=1e-6)
    parser.add_argument('--residual-backward', choices=('recorded', 'differentiate'), default='recorded',
                        help='Diagnostic residual derivative policy; local metric/selection stops are unchanged')
    parser.add_argument('--gpu-memory-gib', type=float, default=12.)
    args = parser.parse_args()
    if not 1 <= args.updates <= 200:parser.error('A bounded 1..200 update study is required')
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:parser.error('Finite positive learning rate required')
    if args.output.exists():raise FileExistsError(args.output)
    training = args.train_indices if args.train_indices is not None else [args.index]
    development = args.development_indices if args.development_indices is not None else [args.development_index]
    if len(set(training)) != len(training) or len(set(development)) != len(development):
        parser.error('Window lists must not contain duplicates')
    if any(abs(a-b) <= 2 for a in training for b in development):
        parser.error('Training/development two-hop windows must not overlap')
    from wp_vbd.newton_source import activate_newton_source, source_record
    activate_newton_source()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership = claim_cuda(args.device, ROOT)
    import torch
    import warp as wp
    import newton
    from wp_vbd.gpu_budget import GpuBudget
    from wp_vbd.franka_reference_data import load_reference
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.mixed_experiment import select, loss, external_inputs, clip_gradient_norm_fp64
    from wp_vbd.general_joint_data import set_controls
    from wp_vbd.mixed_contact_data import query
    torch.set_num_threads(1);torch.manual_seed(312);torch.use_deterministic_algorithms(True)
    budget = GpuBudget(args.device, args.gpu_memory_gib)
    wp.config.kernel_cache_dir = str(ROOT / '.local/p0/warp-cache')
    wp.init();wp.set_device(args.device)
    if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    def forbidden(*unused, **kwargs):raise AssertionError('No complete teacher solver in fitting')
    for cls in (newton.solvers.SolverVBD, newton.solvers.SolverFeatherstone):
        cls.__init__ = forbidden;cls.step = forbidden
    scene, states = load_reference(args.dataset, wp, newton, args.device)
    for index in (*training, *development):
        if not 0 <= index < len(states['x']) - 2:parser.error('Complete two-hop windows required')
    digest = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    stored = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    stored.pop('optimizer', None)
    config = copy.deepcopy(stored.get('model_config') or stored['config']['model'])
    if args.residual_backward == 'differentiate':config['frozen_residual_backward'] = False
    model = MixedSolver(config).to(args.device);model.load_state_dict(stored['model'])
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith('protected_response.'))
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, eps=1e-12, weight_decay=0., foreach=False)
    args.output.mkdir(parents=True)
    began = time.perf_counter()
    report = {'status': 'RUNNING', 'parent_checkpoint': str(args.checkpoint.resolve()),
              'parent_sha256': digest, 'parent_course_step': stored['step'], 'model_config': config,
              'newton_source': source_record(), 'train_indices': training,
              'development_indices': development, 'planned_updates': args.updates,
              'residual_backward_policy': args.residual_backward,
              'residual_backward_scope': 'Differentiate residual only; detached local metric and discrete selection remain approximate.',
              'optimizer': {'name': 'AdamW', 'learning_rate': args.learning_rate, 'eps': 1e-12,
                            'weight_decay': 0., 'gradient_clip': 1.},
              'samples': [], 'monitors': [], 'new_optimizer_updates': 0,
              'full_vbd_calls': 0, 'featherstone_calls': 0,
              'scope': 'Qualified teacher-conditioned training windows, sampled cyclically. Same-trajectory development '
                       'window is not independent generalization. Diagnostic fit only; not an all-demo '
                       'selected checkpoint, autonomous grasp result, or convergence certification.'}
    def persist():
        report['seconds'] = time.perf_counter() - began
        report['gpu_budget'] = budget.report()
        path = args.output / 'study.json';temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n');temporary.replace(path)
    def objective(index):
        current = select(states, slice(index, index + 1))
        state_loss = current.x.new_zeros(());repair_loss = current.x.new_zeros(())
        for hop in range(2):
            i = index + hop;set_controls(scene, states, i);query(scene, current, wp)
            current, diagnostic = model(scene, current, scene.metadata['dt'], **external_inputs(states, slice(i, i + 1)))
            if not bool(diagnostic['accepted'].all()):raise RuntimeError('Microfit window rejected')
            scale = ((1 / 120 / scene.metadata['dt']) ** 4) / 2
            state_loss = state_loss + loss(scene, current, select(states, slice(i + 1, i + 2))) * scale
            repair_loss = repair_loss + diagnostic.get('contact_repair_supervision', 0.) * scale
        total = state_loss + repair_loss
        if not bool(torch.isfinite(total)):raise RuntimeError('Nonfinite microfit objective')
        return total, {'state_loss': float(state_loss.detach()), 'repair_loss': float(repair_loss.detach())}
    best = float('inf')
    def monitor(update):
        nonlocal best
        model.eval()
        with torch.no_grad():
            train_values = {str(index): objective(index) for index in training}
            dev_values = {str(index): objective(index) for index in development}
        train = sum(float(v[0]) for v in train_values.values()) / len(training)
        dev = sum(float(v[0]) for v in dev_values.values()) / len(development)
        parts = lambda values: {key: sum(v[1][key] for v in values.values()) / len(values)
                                for key in ('state_loss', 'repair_loss')}
        row = {'after_update': update, 'train_loss': train, 'development_loss': dev,
               'train_parts': parts(train_values), 'development_parts': parts(dev_values),
               'training_windows': {k: float(v[0]) for k, v in train_values.items()},
               'development_windows': {k: float(v[0]) for k, v in dev_values.items()}}
        if train < best:
            best = train;report['best_training_update'] = update
            if update:
                payload = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                           'model_config': config, 'step': update, 'parent_course_step': stored['step'],
                           'parent_sha256': digest, 'schema': 'franka_microfit_study_v312',
                           'qualified_for_all_demo_deployment': False}
                temporary = args.output / 'best.tmp';torch.save(payload, temporary)
                temporary.replace(args.output / 'best.pt')
        report['monitors'].append(row);budget.observe(force=True);persist()
        print(json.dumps({'monitor': row}), flush=True)
    try:
        monitor(0)
        for update in range(1, args.updates + 1):
            model.train();optimizer.zero_grad(set_to_none=True)
            index = training[(update - 1) % len(training)]
            value, details = objective(index);value.backward()
            norm = clip_gradient_norm_fp64(parameters, 1.)
            available = [p for p in parameters if p.grad is not None]
            chosen = available[int(torch.stack([p.grad.abs().amax() for p in available]).argmax())]
            position = int(chosen.grad.abs().flatten().argmax())
            before = float(chosen.flatten()[position]);optimizer.step()
            changed = before != float(chosen.flatten()[position])
            report['new_optimizer_updates'] = update
            report['samples'].append({'update': update, 'index': index, 'loss_before_update': float(value.detach()),
                                      **details, 'gradient_norm': float(norm), 'tracked_scalar_changed': changed})
            del value
            if update in (1, 4, 8, 16, 32, 64, 128, args.updates):
                monitor(update)
                if update >= 8 and report['monitors'][-1]['train_loss'] > report['monitors'][0]['train_loss'] * 1.05:
                    report['status'] = 'STOPPED_TRAINING_LOSS_REGRESSION';break
        else:report['status'] = 'COMPLETE'
        assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == digest
        report['parent_checkpoint_unchanged'] = True
        report['training_loss_reduced'] = best < report['monitors'][0]['train_loss']
    except BaseException as error:
        report['status'] = 'FAILED';report['error'] = repr(error);raise
    finally:
        persist()
    print(json.dumps({'status': report['status'], 'updates': report['new_optimizer_updates'],
                      'training_loss_reduced': report['training_loss_reduced']}), flush=True)


if __name__ == '__main__':main()
