"""Check physical response supervision against independent quadratic algebra."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    import torch
    from wp_vbd.elastic_shape_objective import (projected_modal_systems,
        directional_energy_recovery, physical_direction_loss, ray_energy_recovery)
    from wp_vbd.elastic_shape_modes import material_basis
    torch.set_num_threads(1); torch.manual_seed(20261118)
    dtype = torch.float64
    matrix = torch.randn(5, 5, dtype=dtype)
    gram = matrix.T @ matrix + torch.eye(5, dtype=dtype)
    rhs = torch.randn(5, dtype=dtype)
    optimum = torch.linalg.solve(gram, rhs)
    system = {'gram': gram[None], 'rhs': rhs[None],
              'optimal_work': (rhs @ optimum)[None]}
    coefficients = torch.randn(1, 5, dtype=dtype, requires_grad=True)
    recovery, active = directional_energy_recovery(coefficients, system)
    assert bool(active.all()) and 0 <= float(recovery.detach()) <= 1
    for scale in (-7.3, 1e-8, 1e8):
        other, _ = directional_energy_recovery(coefficients * scale, system)
        torch.testing.assert_close(other, recovery, rtol=1e-12, atol=1e-12)
    perfect, _ = directional_energy_recovery(optimum[None], system)
    torch.testing.assert_close(perfect, torch.ones_like(perfect), rtol=1e-12, atol=1e-12)
    transform = torch.randn(5, 5, dtype=dtype) + 3 * torch.eye(5, dtype=dtype)
    changed = {'gram': (transform.T @ gram @ transform)[None],
               'rhs': (transform.T @ rhs)[None], 'optimal_work': system['optimal_work']}
    transformed = torch.linalg.solve(transform, coefficients[..., None])[..., 0]
    transformed_recovery, _ = directional_energy_recovery(transformed, changed)
    torch.testing.assert_close(transformed_recovery, recovery, rtol=1e-11, atol=1e-11)
    loss = physical_direction_loss(coefficients, system)
    gradient, = torch.autograd.grad(loss, coefficients)
    epsilon = 1e-6
    finite_difference = torch.zeros_like(gradient)
    for index in range(5):
        offset = torch.zeros_like(coefficients); offset[:, index] = epsilon
        finite_difference[:, index] = (physical_direction_loss(coefficients + offset, system)
            - physical_direction_loss(coefficients - offset, system)) / (2 * epsilon)
    torch.testing.assert_close(gradient, finite_difference, rtol=1e-6, atol=1e-9)
    empty = {**system, 'rhs': torch.zeros_like(system['rhs']),
             'optimal_work': torch.zeros_like(system['optimal_work'])}
    assert not bool(directional_energy_recovery(coefficients, empty)[1].any())
    assert float(physical_direction_loss(coefficients, empty)) == 0
    assert float(physical_direction_loss(torch.zeros_like(coefficients), system)) == 1

    # Two material objects with a coupled SPD operator and a nonempty protected
    # subspace. Compare the reduced objective to a direct full-space energy step.
    n = 20
    points = torch.randn(1, n, 3, dtype=dtype)
    mass = torch.rand(n, dtype=dtype) + .5
    dof = torch.zeros(n, 6, dtype=torch.bool); dof[:, :3] = True
    scene = SimpleNamespace(n=n, dof=dof, interaction_entities=[torch.arange(n)])
    lowrank = torch.randn(6*n, 9, dtype=dtype)
    h = torch.diag(torch.rand(6*n, dtype=dtype) + 1) + lowrank @ lowrank.T
    mask = dof.flatten().to(dtype)
    h = h * mask[:, None] * mask[None, :]
    class Metric:
        terms = []
        def __call__(self, value):
            return (h @ value.flatten(1, 2)).reshape_as(value)
    metric = Metric()
    z = torch.randn(6*n, 3, dtype=dtype) * mask[:, None]
    reduced = z.T @ h @ z
    def coarse(value):
        vector = value.ndim == 3
        source = value[..., None] if vector else value
        projected = z @ torch.linalg.solve(reduced, z.T @ source.flatten(1, 2))
        projected = projected.reshape_as(source)
        return projected[..., 0] if vector else projected
    details = []
    combined = torch.zeros(1, n, 6, dtype=dtype)
    all_coefficients = []
    for object_id, ids in enumerate((torch.arange(10), torch.arange(10, 20))):
        basis = material_basis(points[:, ids], points[0, ids], mass[ids])
        details.append({'object_id': object_id, 'ids': ids, 'basis': basis})
        c = torch.randn(1, 24, dtype=dtype); all_coefficients.append(c)
        combined[:, ids, :3] = torch.einsum('bncp,bp->bnc', basis, c)
    residual = torch.randn_like(combined) * dof
    systems = projected_modal_systems(scene, metric, coarse, residual, details, joint=True)
    assert len(systems) == 1 and systems[0]['gram'].shape == (1, 48, 48)
    projected = combined - coarse(metric(combined[..., None])[..., 0])
    image = metric(projected[..., None])[..., 0]
    actual, active = ray_energy_recovery(projected, image, residual, systems[0]['optimal_work'])
    expected, _ = directional_energy_recovery(torch.cat(all_coefficients, -1), systems[0])
    torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    assert bool(active.all()) and 0 <= float(actual) <= 1 + 1e-10
    alpha = -(projected * residual).sum() / (projected * image).sum()
    update = alpha * projected
    energy = (residual * update + .5 * update * metric(update[..., None])[..., 0]).sum()
    torch.testing.assert_close(-2 * energy, actual[0] * systems[0]['optimal_work'][0],
                               rtol=1e-10, atol=1e-10)
    cross = systems[0]['gram'][0, :24, 24:]
    assert float(cross.abs().max()) > 1e-6
    report = {'status': 'PASS', 'sign_and_scale_invariant': True,
              'basis_coordinate_invariant': True, 'oracle_recovers_one': True,
              'zero_load_has_no_label': True, 'coupled_objects_cross_terms_retained': True,
              'zero_predicted_direction_counts_as_failure': True,
              'matches_full_space_quadratic_energy': True,
              'gradient_finite_difference_max_error': float((gradient-finite_difference).abs().max()),
              'optimizer_updates': 0}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
