"""Compare learned and fixed physical responses on identical causal windows."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda:0'), default='cpu')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    from wp_vbd.newton_source import activate_newton_source, source_record
    activate_newton_source()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership = claim_cuda(args.device, ROOT)
    import torch
    import warp as wp
    import newton
    from wp_vbd.franka_reference_data import load_reference
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.mixed_experiment import select, loss, external_inputs
    from wp_vbd.general_joint_data import set_controls
    from wp_vbd.mixed_contact_data import query
    torch.set_num_threads(1);torch.manual_seed(312);torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir = str(ROOT / '.local/p0/warp-cache')
    wp.init();wp.set_device(args.device)
    if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    def forbidden(*unused, **kwargs):raise AssertionError('No complete solver in response baseline')
    for cls in (newton.solvers.SolverVBD, newton.solvers.SolverFeatherstone):
        cls.__init__ = forbidden;cls.step = forbidden
    scene, states = load_reference(args.dataset, wp, newton, args.device)
    stored = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    stored.pop('optimizer', None)
    config = stored.get('model_config') or stored['config']['model']
    model = MixedSolver(config).to(args.device).eval();model.load_state_dict(stored['model'])
    calls = 0
    def count(*unused):
        nonlocal calls
        calls += 1
    model.protected_response.hierarchy.register_forward_hook(count)
    rows = []
    with torch.no_grad():
        for index in (0, 520, 720, 1350):
            for policy in ('local_coarse', 'physical_vector', 'learned'):
                model.protected_response.mode = 'physical_vector' if policy == 'physical_vector' else 'learned'
                current = select(states, slice(index, index + 1));total = current.x.new_zeros(())
                start_calls = calls;began = time.perf_counter();accepted = True;hops = []
                for hop in range(2):
                    i = index + hop;set_controls(scene, states, i);query(scene, current, wp)
                    current, diagnostic = model(scene, current, scene.metadata['dt'],
                        learned=policy != 'local_coarse', **external_inputs(states, slice(i, i + 1)))
                    ok = bool(diagnostic['accepted'].all());accepted &= ok
                    total = total + (loss(scene, current, select(states, slice(i + 1, i + 2)))
                        + diagnostic.get('contact_repair_supervision', 0.)) * ((1 / 120 / scene.metadata['dt']) ** 4) / 2
                    hops.append({'accepted': ok})
                    if not ok:break
                target = select(states, slice(index + len(hops), index + len(hops) + 1))
                n = scene.metadata['particle_count'];b = scene.metadata['body_count']
                row = {'index': index, 'response_policy': policy, 'accepted': accepted, 'hops': len(hops),
                       'two_hop_loss': float(total) if accepted and len(hops) == 2 else None,
                       'material_position_rms_m': float((current.x[:, :n] - target.x[:, :n]).square().mean().sqrt()),
                       'robot_com_position_rms_m': float((current.x[:, n:n+b] - target.x[:, n:n+b]).square().mean().sqrt()),
                       'backbone_calls': calls - start_calls, 'seconds': time.perf_counter() - began}
                assert row['backbone_calls'] == (4 * len(hops) if policy == 'learned' else 0)
                rows.append(row);print(json.dumps(row), flush=True)
    model.protected_response.mode = 'learned'
    assert all(torch.equal(p.detach().cpu(), stored['model'][name]) for name, p in model.named_parameters())
    result = {'status': 'COMPLETE', 'device': args.device, 'cases': rows,
              'checkpoint_sha256': hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              'newton_source': source_record(), 'unroll': config['unroll'], 'dt': scene.metadata['dt'],
              'new_optimizer_updates': 0, 'full_vbd_calls': 0, 'featherstone_calls': 0,
              'scope': 'Same physical operators, controls, timestep, history and maximum correction budget. '
                       'Local/coarse removes learned response; physical_vector uses fixed unit port shapes and '
                       'a physical inverse gate. No teacher correction is applied to predicted states. '
                       'Teacher-conditioned windows are not autonomous task validation or fair end-to-end speed benchmarks.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':main()
