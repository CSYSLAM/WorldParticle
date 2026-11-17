"""Verify completed surface/joint searches against the full candidate schedule."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def toy_checks(torch):
    from wp_vbd.data import State
    from wp_vbd.rigid_contact_solver import choose
    from wp_vbd import self_contact_repair
    results = []
    for magnitude, repair in ((3., False), (100., False), (3., True)):
        variants = []
        for full in (True, False):
            mobile = torch.tensor([True, True, False])
            scene = SimpleNamespace(movable=mobile, fast_rigid_contact=False,
                learned_contact_repair=repair, dof=mobile[:, None].expand(-1, 6),
                metadata={'contract': 'elastic_surface_joint_contact_v38',
                          'evaluate_all_line_search_rays': full})
            x = torch.zeros(1, 3, 3, dtype=torch.float64)
            q = torch.zeros(1, 3, 4, dtype=torch.float64);q[..., 3] = 1
            state = State(x, q, x.clone(), x.clone())
            update = torch.zeros(1, 3, 6, dtype=torch.float64)
            update[0, 0, 0] = 1.;update[0, 1, 0] = magnitude
            update.requires_grad_()
            calls = {'retract': 0, 'repair': 0}
            def retract(s, candidate):
                calls['retract'] += 1
                return candidate, torch.ones(1, dtype=torch.bool)
            def repair_candidate(s, candidate, context):
                calls['repair'] += 1
                return candidate
            target = torch.tensor([[1., 1., 0.]], dtype=torch.float64)
            physics = SimpleNamespace(retract=retract,
                merit=lambda s, y, *unused: (y.x[..., 0] - target).square(),
                acceptance_groups=lambda *unused, **kwargs: [torch.tensor([0]), torch.tensor([1])])
            scene.geometry_oracle = SimpleNamespace(motion=lambda s, y:
                (torch.ones(1, dtype=torch.bool), (y.x - s.x).norm(dim=-1)))
            context = SimpleNamespace(contacts=SimpleNamespace())
            if repair:context.contacts.self_snapshot = object()
            saved = self_contact_repair.candidate
            self_contact_repair.candidate = repair_candidate
            try:
                chosen, selected, used = choose(scene, state, state, update, x, q, context,
                    {'raw_depth': torch.zeros(1, 0)}, [1., .5, .25, .125, .0625], physics=physics)
                chosen.x.square().sum().backward()
            finally:
                self_contact_repair.candidate = saved
            variants.append((chosen.x.detach(), selected, used, update.grad, calls))
        for left, right in zip(variants[0][:4], variants[1][:4]):
            assert torch.equal(left, right)
        expected = 5 if repair or magnitude == 100 else 2
        assert variants[0][4]['retract'] == 5 and variants[1][4]['retract'] == expected
        if repair:assert variants[0][4]['repair'] == variants[1][4]['repair'] == 5
        results.append({'second_update': magnitude, 'repair_enabled': repair,
                        'full_calls': variants[0][4], 'optimized_calls': variants[1][4],
                        'state_selection_feedback_gradient_equal': True})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--dataset', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--toy-only', action='store_true')
    parser.add_argument('--indices', type=int, nargs='+', default=[0, 520, 720, 1350])
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    from wp_vbd.gpu_ownership import claim_cuda
    ownership = claim_cuda(args.device, ROOT)
    import torch
    torch.set_num_threads(1);torch.manual_seed(312);torch.use_deterministic_algorithms(True)
    result = {'status': 'PASS', 'toy_cases': toy_checks(torch), 'device': args.device,
              'new_optimizer_updates': 0, 'full_vbd_calls': 0, 'featherstone_calls': 0}
    if not args.toy_only:
        if args.dataset is None or args.checkpoint is None:parser.error('Dataset and checkpoint required')
        from wp_vbd.newton_source import activate_newton_source, source_record
        activate_newton_source()
        import warp as wp
        import newton
        from wp_vbd.franka_reference_data import load_reference
        from wp_vbd.mixed_experiment import select, loss, external_inputs
        from wp_vbd.mixed_contact_data import query
        from wp_vbd.general_joint_data import set_controls
        from wp_vbd.mixed_solver import MixedSolver
        from wp_vbd import surface_joint_system as physics
        wp.config.kernel_cache_dir = str(ROOT / '.local/p0/warp-cache')
        wp.init();wp.set_device(args.device)
        if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
        def forbidden(*unused, **kwargs):raise AssertionError('No complete solver in verification')
        for cls in (newton.solvers.SolverVBD, newton.solvers.SolverFeatherstone):
            cls.__init__ = forbidden;cls.step = forbidden
        scene, states = load_reference(args.dataset, wp, newton, args.device)
        stored = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
        stored.pop('optimizer', None)
        config = stored.get('model_config') or stored['config']['model']
        model = MixedSolver(config).to(args.device);model.load_state_dict(stored['model'])
        def evaluate(index, full, backward):
            scene.metadata['evaluate_all_line_search_rays'] = full
            model.train(backward);model.zero_grad(set_to_none=True)
            current = select(states, slice(index, index + 1));objective = current.x.new_zeros(())
            saved = physics.retract;calls = 0
            def counted(*args, **kwargs):
                nonlocal calls
                calls += 1
                return saved(*args, **kwargs)
            physics.retract = counted
            try:
                with torch.set_grad_enabled(backward):
                    for hop in range(2):
                        i = index + hop;set_controls(scene, states, i);query(scene, current, wp)
                        current, d = model(scene, current, scene.metadata['dt'], **external_inputs(states, slice(i, i + 1)))
                        assert bool(d['accepted'].all())
                        objective = objective + (loss(scene, current, select(states, slice(i + 1, i + 2)))
                            + d.get('contact_repair_supervision', 0.)) * ((1 / 120 / scene.metadata['dt']) ** 4) / 2
                    if backward:objective.backward()
            finally:
                physics.retract = saved
            values = {key: getattr(current, key).detach().cpu().clone() for key in ('x', 'q', 'v', 'omega')}
            values.update({'history__' + k: v.detach().cpu().clone() for k, v in current.history.items()})
            gradients = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
            return values, gradients, float(objective.detach()), calls
        cases = [];maximum_state = maximum_gradient = 0.
        for index in args.indices:
            backward = index == 720
            left = evaluate(index, True, backward)
            repeat = evaluate(index, True, backward) if index == 0 else None
            right = evaluate(index, False, backward)
            assert left[0].keys() == right[0].keys()
            differences = {}
            repeat_differences = {}
            for key, value in left[0].items():
                other = right[0][key]
                if value.dtype == torch.bool:assert torch.equal(value, other)
                else:
                    difference = float((value - other).abs().max()) if value.numel() else 0.
                    differences[key] = difference
                    maximum_state = max(maximum_state, difference)
                    # Velocities reconstruct position differences divided by
                    # h=1/600. Use an absolute tolerance in their own units.
                    atol = 1e-9 if key in ('v', 'omega') else 1e-12
                    torch.testing.assert_close(value, other, atol=atol, rtol=1e-12)
                    if repeat is not None:
                        repeat_differences[key] = float((value-repeat[0][key]).abs().max()) if value.numel() else 0.
                        torch.testing.assert_close(value, repeat[0][key], atol=atol, rtol=1e-12)
            assert left[1].keys() == right[1].keys()
            for key, value in left[1].items():
                maximum_gradient = max(maximum_gradient, float((value - right[1][key]).abs().max()))
                torch.testing.assert_close(value, right[1][key], atol=1e-10, rtol=1e-6)
            print(json.dumps({'comparison_index': index, 'full_loss': left[2], 'optimized_loss': right[2],
                              'maximum_state_difference': maximum_state,
                              'maximum_gradient_difference': maximum_gradient,
                              'full_retractions': left[3], 'optimized_retractions': right[3]}), flush=True)
            torch.testing.assert_close(torch.tensor(left[2],dtype=torch.float64),
                                       torch.tensor(right[2],dtype=torch.float64),atol=1e-10,rtol=1e-9)
            assert right[3] <= left[3]
            cases.append({'index': index, 'two_hop_loss': left[2], 'optimized_loss': right[2],
                          'state_absolute_differences': differences,
                          'full_schedule_repeat_differences': repeat_differences,
                          'full_schedule_repeat_loss': repeat[2] if repeat is not None else None,
                          'gradient_tensors': len(left[1]),
                          'full_retractions': left[3], 'optimized_retractions': right[3]})
            print(json.dumps(cases[-1]), flush=True)
        assert all(torch.equal(p.detach().cpu(), stored['model'][name]) for name, p in model.named_parameters())
        result.update(cases=cases, maximum_state_difference=maximum_state,
                      maximum_parameter_gradient_difference=maximum_gradient, newton_source=source_record(),
                      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                      implementation_sha256=hashlib.sha256((ROOT / 'src/wp_vbd/rigid_contact_solver.py').read_bytes()).hexdigest())
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':main()
