"""Check real analytic face/edge rows and vertex compatibility on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import activate_newton_source
activate_newton_source()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True); args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd import math3d as so3, surface_contact_rows as physics
    from wp_vbd.soft_surface_ports import SoftSurfacePorts, native_state
    from wp_vbd.data import State
    from wp_vbd.rigid_contact_system import ContactMetric
    torch.set_num_threads(1); torch.manual_seed(20261101)
    wp.config.kernel_cache_dir = str(ROOT/'.local/p0/warp-cache'); wp.init(); wp.set_device('cpu')
    def inverse_mass(model, state):
        n, b = model.particle_count, model.body_count
        inverse = state.x.new_zeros(1, n+b+1, 6, 6)
        # Use the new model's FP64 physical mass convention, rather than
        # the source's separately rounded FP32 inverse-mass cache.
        particle_mass = wp.to_torch(model.particle_mass).double()
        active = (wp.to_torch(model.particle_flags).long() & int(newton.ParticleFlags.ACTIVE)) != 0
        particle_inverse = torch.where(active & (particle_mass > 0), particle_mass.clamp_min(1e-30).reciprocal(), 0.)
        body_inverse = state.x.new_zeros(b)
        if b:
            body_mass = wp.to_torch(model.body_mass).double()
            dynamic = wp.to_torch(model.body_flags).long() == int(newton.BodyFlags.DYNAMIC)
            body_inverse = torch.where(dynamic & (body_mass > 0), body_mass.clamp_min(1e-30).reciprocal(), 0.)
        mass = torch.cat((particle_inverse, body_inverse, state.x.new_zeros(1)))
        inverse[..., :3, :3] = mass[None, :, None, None]*torch.eye(3, dtype=torch.float64)
        if b:
            rotation = so3.matrix(state.q[:, n:n+b])
            inverse[:, n:n+b, 3:, 3:] = rotation@wp.to_torch(model.body_inv_inertia).double()@rotation.transpose(-1, -2)
            inverse[:, n:n+b, 3:, 3:] *= (body_inverse > 0)[None, :, None, None]
        return inverse
    def check(model, native, contacts, dt, epsilon):
        snapshot = SoftSurfacePorts.snapshot(model, contacts, wp)
        start, com, body_mask = native_state(model, native, wp)
        material = physics.export_material(model, snapshot, wp, newton)
        context = physics.begin(snapshot, start, com, inverse_mass(model, start), dt, material, epsilon)
        displacement = torch.randn(1, len(com), 6, dtype=torch.float64)*1e-5
        displacement[:, :model.particle_count, :3] += displacement.new_tensor([.0001, -.0001, -.0002])
        displacement[:, :model.particle_count, 3:] = 0; displacement[:, -1] = 0
        current = State(start.x+displacement[..., :3], so3.multiply(so3.exp(displacement[..., 3:]), start.q), start.v, start.omega)
        frozen = physics.rows(snapshot, current, com, context)
        assert bool(frozen['valid'].all()) and bool(frozen['active'].any())
        perturbation = torch.zeros_like(displacement, requires_grad=True)
        variable = State(current.x+perturbation[..., :3], so3.multiply(so3.exp(perturbation[..., 3:]), current.q), current.v, current.omega)
        energy = physics.merit(snapshot, variable, com, context, frozen).sum()
        gradient, = torch.autograd.grad(energy, perturbation)
        residual_error = float((gradient-frozen['residual']).abs().max())
        assert residual_error < 1e-9*max(1., float(frozen['residual'].abs().max()))
        dof = torch.ones(len(com), 6, dtype=torch.bool); dof[:model.particle_count, 3:] = False; dof[-1] = False
        metric = ContactMetric(current.x.new_zeros(1, len(com), 6, 6), [frozen['metric_term']], dof)
        vector = torch.randn(1, len(com), 6, 2, dtype=torch.float64)
        metric_energy = (vector*metric(vector)).sum()
        assert float(metric_energy) >= -1e-12
        zero = physics.SurfaceMaterial(material.stiffness*0, material.damping*0, material.friction*0)
        zero_context = physics.begin(snapshot, start, com, inverse_mass(model, start), dt, zero, epsilon)
        empty = physics.rows(snapshot, current, com, zero_context)
        assert not bool(empty['residual'].any()) and not bool(empty['metric_term'].weight.any())
        assert not bool(physics.merit(snapshot, current, com, zero_context, empty).any())
        changes = {}
        variants = {'stiffness': physics.SurfaceMaterial(material.stiffness*2, material.damping, material.friction),
                    'damping': physics.SurfaceMaterial(material.stiffness, material.damping*0, material.friction),
                    'friction': physics.SurfaceMaterial(material.stiffness, material.damping, material.friction*0)}
        for name, altered in variants.items():
            changed_context = physics.begin(snapshot, start, com, inverse_mass(model, start), dt, altered, epsilon)
            changed = physics.rows(snapshot, current, com, changed_context)
            changes[name] = float((changed['residual']-frozen['residual']).abs().max())
            assert changes[name] > 0, name
        return {'active_rows': int(frozen['active'].sum()), 'gradient_merit_max_error': residual_error,
                'sparse_metric_energy': float(metric_energy), 'zero_material_force_metric_energy_exactly_zero': True,
                'parameter_max_residual_changes': changes}, (snapshot, start, current, com, material, context, frozen, dof)
    results = []
    for identifier in ('vbd.example_vbd_gripper_soft_triangle', 'vbd.example_vbd_gripper_soft_grid'):
        example, configurations, _ = make_example(identifier, reference=False, preserve_defaults=True)
        configuration = configurations[0][1]
        pose = example.state_0.body_q.numpy(); fingers = np.abs(pose[:, 1]) > .5*example.params['open_half_gap']
        pose[fingers, 1] = np.sign(pose[fingers, 1])*example.params['closed_half_gap']; example.state_0.body_q.assign(pose)
        example.collision_pipeline.collide(example.state_0, example.contacts)
        row, _ = check(example.model, example.state_0, example.contacts, example.sim_dt, configuration['friction_epsilon'])
        results.append({'demo': identifier, **row})
    def forbidden(*unused, **kwargs): raise AssertionError('Full VBD forbidden in row verification')
    newton.solvers.SolverVBD.__init__ = forbidden; newton.solvers.SolverVBD.step = forbidden
    builder = newton.ModelBuilder(gravity=(0., 0., 0.))
    builder.add_particle(wp.vec3(0., 0., .005), wp.vec3(0.), .01, radius=.01)
    builder.add_ground_plane(); builder.color(); model = builder.finalize()
    model.soft_contact_ke = 1000.; model.soft_contact_kd = .1; model.soft_contact_mu = .8
    native = model.state(); pipeline = newton.CollisionPipeline(model, soft_contact_gap=.02, enable_rigid_soft_full_surface_contact=True)
    contacts = pipeline.contacts(); pipeline.collide(native, contacts)
    row, packed = check(model, native, contacts, .01, configuration['friction_epsilon'])
    snapshot, start, current, com, material, context, frozen, dof = packed
    from wp_vbd.mixed_contact_scene import export_model
    from wp_vbd.mixed_collision import MixedContacts
    from wp_vbd import rigid_contact_system as vertex
    scene = export_model(model, configuration, 'research.vertex_port_degeneracy', newton, .01, 'cpu')
    old_contacts = MixedContacts.snapshot(model, contacts, wp, start)
    old_context = vertex.begin(scene, start, old_contacts, .01)
    old_rows = vertex.rows(scene, current, start, old_context)
    difference = float((old_rows['residual']-frozen['residual']*scene.dof).abs().max())
    assert difference < 1e-12
    h0 = ContactMetric(current.x.new_zeros(1, len(com), 6, 6), old_rows['terms'], dof)
    h1 = ContactMetric(h0.local, [frozen['metric_term']], dof)
    vector = torch.randn(1, len(com), 6, 2, dtype=torch.float64)
    metric_difference = float((h0(vector)-h1(vector)).abs().max())
    assert metric_difference < 1e-9
    old_merit = vertex.merit(scene, current, start, current.x, current.q, old_context, old_rows, include_inertia=False).sum()
    new_merit = physics.merit(snapshot, current, com, context, frozen).sum()
    assert torch.allclose(old_merit, new_merit, atol=1e-14, rtol=1e-14)
    wp.to_torch(contacts.soft_contact_count).zero_()
    no_contacts = SoftSurfacePorts.snapshot(model, contacts, wp)
    no_material = physics.export_material(model, no_contacts, wp, newton)
    no_context = physics.begin(no_contacts, start, com, inverse_mass(model, start), .01, no_material, configuration['friction_epsilon'])
    empty_rows = physics.rows(no_contacts, current, com, no_context)
    assert not bool(empty_rows['residual'].any()) and not empty_rows['metric_term'].ids.numel()
    assert not physics.merit(no_contacts, current, com, no_context, empty_rows).numel()
    try: physics.begin(snapshot, start, com, -torch.eye(6, dtype=torch.float64)[None, None].expand(1, len(com), -1, -1), .01, material, configuration['friction_epsilon'])
    except ValueError: pass
    else: raise AssertionError('Negative inverse mass accepted')
    result = {'status': 'PASS', 'device': 'cpu', 'face_edge_results': results, 'vertex_result': row,
              'vertex_residual_difference': difference, 'vertex_metric_apply_difference': metric_difference,
              'vertex_merit_matches_existing_contract': True, 'new_optimizer_updates': 0,
              'empty_contacts_have_zero_force_and_empty_metric': True, 'negative_inverse_mass_rejected': True,
              'new_training_admissions': 0, 'full_vbd_prediction_calls': 0,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__), ROOT/'src/wp_vbd/contact_ports.py', ROOT/'src/wp_vbd/soft_surface_ports.py', ROOT/'src/wp_vbd/surface_contact_rows.py')},
              'scope': 'Analytic shape local constitutive assembly and derivative/metric checks. This adds no full mixed solver route, mesh force-eligibility filter, motion guard, gripper training or autonomous task acceptance.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
