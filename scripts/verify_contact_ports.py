"""Independent kinematics/virtual-work checks for prototype contact ports."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='cpu'); parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    import torch
    from wp_vbd import math3d as so3
    from wp_vbd.contact_ports import PointPorts
    from wp_vbd.data import State
    from wp_vbd.rigid_contact_system import ContactMetric
    torch.set_num_threads(1); torch.manual_seed(20261101)
    device = args.device
    def floating(value): return torch.tensor(value, device=device, dtype=torch.float64)
    owners = torch.tensor([[0, 1, 2, 3], [0, 1, 2, -1], [0, 1, -1, -1]], device=device)
    weights = floating([[[1, 0, 0, 0], [0, .2, .3, .5]], [[1, 0, 0, 0], [0, .4, .6, 0]], [[1, 0, 0, 0], [0, 1, 0, 0]]])
    rigid = torch.tensor([[True, False]]*3, device=device)
    local = floating([[[.02, -.01, .04], [0, 0, 0]]]*3)
    body_mask = torch.tensor([True, False, False, False], device=device)
    ports = PointPorts(owners, weights, rigid, local, body_mask)
    x = floating([[[.03, -.02, .1], [-.3, 0, -.1], [.3, 0, -.1], [.07, 0, .5]]])
    q = so3.exp(floating([[[.1, -.2, .3], [0, 0, 0], [0, 0, 0], [0, 0, 0]]]))
    state = State(x, q, torch.zeros_like(x), torch.zeros_like(x))
    com = floating([[.01, -.02, .03], [0, 0, 0], [0, 0, 0], [0, 0, 0]])
    point, jac = ports.geometry(state, com)
    assert torch.allclose(point[:, 0, 1], .2*x[:, 1]+.3*x[:, 2]+.5*x[:, 3], atol=1e-15, rtol=0)
    assert torch.equal(point[:, 2, 1], x[:, 1])
    assert not bool(jac[:, :, 1, :, :, 3:].any())
    turn = so3.exp(floating([.2, -.1, .4]))
    shift = floating([10., -4., 7.])
    transformed = State(so3.rotate(turn.expand(1, 4, 4), x)+shift,
                        so3.multiply(turn, q), state.v, state.omega)
    expected = so3.rotate(turn.expand(1, 3, 2, 4), point)+shift
    assert torch.allclose(ports.geometry(transformed, com)[0], expected, atol=2e-15, rtol=1e-14)
    fp32 = PointPorts(owners, weights.float(), rigid, local.float(), body_mask)
    assert torch.allclose(fp32.weights.sum(-1), torch.ones_like(fp32.weights[..., 0]), atol=1e-15, rtol=0)
    direction = torch.randn(1, 4, 6, device=device, dtype=torch.float64)
    direction[:, 1:, 3:] = 0
    predicted = torch.einsum('brpsci,brsi->brpc', jac, direction[:, ports.ids])
    def moved(alpha):
        return State(x+alpha*direction[..., :3], so3.multiply(so3.exp(alpha*direction[..., 3:]), q), state.v, state.omega)
    eps = 1e-6
    finite = (ports.geometry(moved(eps), com)[0]-ports.geometry(moved(-eps), com)[0])/(2*eps)
    derivative_error = float((predicted-finite).abs().max())
    assert derivative_error < 1e-9
    covectors = torch.randn_like(point)
    generalized = ports.pullback(jac, covectors)
    work_error = float(((generalized*direction).sum()-(covectors*predicted).sum()).abs())
    assert work_error < 1e-12
    # Reciprocal normal forces act along the two endpoint positions. Global
    # force and torque include the actual COM offset and weighted particles.
    normal = point[:, :, 1]-point[:, :, 0]
    normal = normal/normal.norm(dim=-1, keepdim=True)
    force = ports.pullback(jac, torch.stack((-normal, normal), 2))
    net_force = force[..., :3].sum(1)
    net_torque = (torch.linalg.cross(x, force[..., :3])+force[..., 3:]).sum(1)
    assert float(net_force.abs().max()) < 1e-12 and float(net_torque.abs().max()) < 1e-12
    matrix = torch.randn(1, 3, 3, 3, device=device, dtype=torch.float64)
    port_metric = matrix.transpose(-1, -2)@matrix
    term = ports.metric_term(jac, port_metric)
    dof = torch.ones(4, 6, device=device, dtype=torch.bool); dof[1:, 3:] = False
    metric = ContactMetric(x.new_zeros(1, 4, 6, 6), [term], dof)
    vector = direction[..., None]
    metric_energy = (vector*metric(vector)).sum()
    relative = predicted[:, :, 1]-predicted[:, :, 0]
    direct_energy = torch.einsum('brc,brcd,brd->', relative, port_metric, relative)
    assert float(metric_energy) >= 0 and torch.allclose(metric_energy, direct_energy, atol=1e-12, rtol=1e-12)
    zero_term = ports.metric_term(jac, port_metric*0)
    assert not bool(zero_term.weight.any())
    inv = torch.eye(6, device=device, dtype=torch.float64)[None, None].expand(1, 4, -1, -1).clone()
    compliance = ports.normal_compliance(jac, normal, inv)
    assert bool((compliance > 0).all())
    lever = point[:, 0, 0]-x[:, 0]
    analytic = 1.+.2**2+.3**2+.5**2+torch.linalg.cross(lever, normal[:, 0]).square().sum(-1)
    assert torch.allclose(compliance[:, 0], analytic, atol=1e-14, rtol=1e-14)
    # Same physical particle at both endpoints cancels in force, full metric
    # AND compliance. A sum of per-slot inverse masses would be wrong here.
    duplicate = PointPorts(torch.tensor([[1, 1]], device=device), floating([[[1, 0], [0, 1]]]), torch.zeros(1, 2, device=device, dtype=torch.bool), x.new_zeros(1, 2, 3), body_mask)
    _, dj = duplicate.geometry(state, com)
    assert not bool(duplicate.relative_jacobian(dj).any())
    assert not bool(duplicate.normal_compliance(dj, normal[:, :1], inv).any())
    assert not bool(duplicate.pullback(dj, torch.stack((-normal[:, :1], normal[:, :1]), 2)).any())
    # Factorized particle pullback retains padded and repeated real owners,
    # including exactly the same gradients with lower temporary storage.
    particle=PointPorts(torch.tensor([[0,1,2,-1],[2,2,1,-1]],device=device),
        floating([[[1,0,0,0],[0,.4,.6,0]],[[.25,.75,0,0],[0,0,1,0]]]),
        torch.zeros(2,2,device=device,dtype=torch.bool),x.new_zeros(2,2,3),
        torch.zeros(4,device=device,dtype=torch.bool))
    loads=torch.randn(1,2,2,3,device=device,dtype=torch.float64,requires_grad=True)
    full=particle.owner_wrenches(None,loads)
    expected=full.new_zeros(1,4,6).index_add(1,particle.ids.flatten(),full.flatten(1,2))
    actual=particle.pullback(None,loads)
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    first=torch.autograd.grad(actual.square().sum(),loads,retain_graph=True)[0]
    second=torch.autograd.grad(expected.square().sum(),loads)[0]
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    rejected = []
    for name, changed in (('negative_weight', weights.index_fill(2, torch.tensor([0], device=device), -.1)), ('non_affine_weight', weights*2), ('weighted_padding', weights.roll(1, -1))):
        try: PointPorts(owners, changed, rigid, local, body_mask)
        except ValueError: rejected.append(name)
        else: raise AssertionError(name)
    try: ports.metric_term(jac, -torch.eye(3, device=device, dtype=torch.float64)[None, None].expand(1, 3, -1, -1))
    except ValueError: rejected.append('negative_metric')
    else: raise AssertionError('Negative metric accepted')
    # Port snapshots own their arrays; a later caller edit cannot silently
    # move a detector's frozen barycentric point.
    weights.zero_(); local.add_(100); owners.zero_()
    assert torch.equal(point, ports.geometry(state, com)[0])
    result = {'status': 'PASS', 'device': device, 'face_edge_vertex_ports_verified': True,
              'nonzero_body_com_and_rotation_verified': True, 'maximum_point_jacobian_fd_error': derivative_error,
              'virtual_work_error': work_error, 'net_internal_force_max': float(net_force.abs().max()),
              'net_internal_torque_max': float(net_torque.abs().max()), 'sparse_psd_energy': float(metric_energy),
              'repeated_physical_owner_cancellation_and_compliance_verified': True, 'owned_snapshot_verified': True,
              'factorized_particle_pullback_values_and_gradients_preserved':True,
              'rigid_frame_equivariance_verified': True, 'native_fp32_barycentric_affinity_restored': True,
              'weighted_squared_mass_compliance_matches_analytic_value': True,
              'invalid_cases_rejected': rejected, 'new_optimizer_updates': 0, 'new_training_admissions': 0,
              'full_vbd_prediction_calls': 0,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT/'src/wp_vbd/contact_ports.py')},
              'scope': 'Prototype port algebra only. No collision candidates, face/edge law, mixed joint route, selected training or full gripper task is implemented by these checks.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
