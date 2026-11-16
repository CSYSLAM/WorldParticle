"""Admit a complete authored mixed episode with explicit moving pin commands."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), '/home/oem/code/engine/newton']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', default='vbd.example_vbd_soft_rigid_contact')
    parser.add_argument('--tape', type=Path, required=True); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists(): raise FileExistsError('Preserve mixed data admissions')
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.reference_tape import read_tape
    from wp_vbd.mixed_contact_scene import export_model
    from wp_vbd.mixed_contact_data import STATIC_FIELDS, restore_source
    torch.set_num_threads(1); wp.config.kernel_cache_dir = str(ROOT / '.local/p0/warp-cache'); wp.init(); wp.set_device('cpu')
    manifest_path = args.tape / 'manifest.json'; manifest = json.loads(manifest_path.read_text())
    assert manifest['id'] == args.demo and manifest['status'] == 'PHYSICAL_SUBSTEP_TRACE_READY' and manifest['vbd_instances'] == 1 and manifest['source_sha'] == '25298aea1691e060c86f95e8d2b773c898c006d3'
    example, solvers, _ = make_example(args.demo, reference=False, preserve_defaults=True)
    if getattr(example.contacts, '_enable_rigid_soft_full_surface_contact', False): raise ValueError('Source full-surface contacts need a barycentric operator')
    config = solvers[0][1]; assert config == manifest['solvers'][0]['configuration']
    static_path = args.tape / manifest['static']['path']; assert hashlib.sha256(static_path.read_bytes()).hexdigest() == manifest['static']['sha256']
    with np.load(static_path, allow_pickle=False) as archive:
        source = {'source__' + name: archive['solver0__model__' + name].copy() for name in STATIC_FIELDS if getattr(example.model, name, None) is not None}
        source.update({key.removeprefix('solver0__'): archive[key].copy() for key in archive.files if key.startswith('solver0__geometry')})
    # Keep names unambiguous in the stored source asset namespace.
    source = {('source_' + key if key.startswith('geometry') else key): value for key, value in source.items()}
    roundoff = restore_source(example.model, source)
    scene = export_model(example.model, config, args.demo, newton, example.sim_dt, 'cpu')
    n = example.model.particle_count; b = example.model.body_count; pins = scene.fixed_particle_ids.numpy(); mobile = scene.movable[:n].numpy()
    def state(particle, particle_velocity, pose, body_velocity):
        result = {key: np.zeros((scene.n, 4 if key == 'q' else 3), np.float64) for key in ('x', 'q', 'v', 'omega')}; result['q'][:, 3] = 1.
        result['x'][:n] = particle; result['v'][:n] = particle_velocity
        rotation = pose[:, 3:].astype(np.float64); rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
        com = scene.body_com[n:n+b].numpy()
        result['x'][n:n+b] = pose[:, :3] + com + 2 * np.cross(rotation[:, :3], np.cross(rotation[:, :3], com) + rotation[:, 3:] * com)
        result['q'][n:n+b] = rotation; result['v'][n:n+b] = body_velocity[:, :3]; result['omega'][n:n+b] = body_velocity[:, 3:]
        return result
    trajectory = []; input_pins = []; pin_history = []; initialized = []; forces = []; contact_counts = []; previous = None; calls = 0; moving_inputs = 0
    for index, (values, timing) in enumerate(read_tape(args.tape)):
        assert timing['sequence'] == index and timing['solver_slot'] == 0 and timing['dt'] == example.sim_dt and not timing['observables_requested']
        arguments = [values['input__' + name] for name in ('particle_q', 'particle_qd', 'body_q', 'body_qd')]
        output = [values['output__' + name] for name in ('particle_q', 'particle_qd', 'body_q', 'body_qd')]
        assert all(np.isfinite(v).all() for v in (*arguments, *output))
        assert np.array_equal(arguments[0][pins], output[0][pins]), 'Source fixed particles changed inside the solver'
        if previous is None:
            trajectory.append(state(*arguments)); pin_history.append(arguments[0][pins].astype(np.float64)); initialized.append(False)
        else:
            assert np.array_equal(previous[0][mobile], arguments[0][mobile]) and np.array_equal(previous[1][mobile], arguments[1][mobile]), 'Unrepresented dynamic particle control'
            assert np.array_equal(previous[2], arguments[2]) and np.array_equal(previous[3], arguments[3]), 'Unrepresented rigid pose/velocity control'
            if not np.array_equal(previous[0][pins], arguments[0][pins]): moving_inputs += 1
        for name, value in values.items():
            if name.startswith('model_input__'):
                original = getattr(example.model, name.removeprefix('model_input__'), None)
                if original is None:
                    if value.size:raise ValueError('Unrepresented mixed model input: '+name)
                    continue
                original=original.numpy() if hasattr(original,'numpy') else np.asarray(original)
                assert np.array_equal(value,original), 'Mutable material needs an input channel: ' + name
            if name in ('control__tri_activations','control__tet_activations') and np.any(value!=0):
                raise ValueError('Active material control needs an explicit mixed input channel: '+name)
        assert not np.any(values.get('control__joint_f',np.empty(0))), 'FREE efforts need an explicit force adapter'
        force = np.zeros((scene.n, 3), np.float64); torque = force.copy()
        force[:n] = values['input__particle_f']; force[n:n+b] = values['input__body_f'][:, :3]; torque[n:n+b] = values['input__body_f'][:, 3:]
        forces.append((force, torque)); input_pins.append(arguments[0][pins].astype(np.float64)); contact_counts.append(int(values['contacts_before__soft_contact_count'][0]))
        trajectory.append(state(*output)); pin_history.append(output[0][pins].astype(np.float64)); initialized.append(True)
        previous = output; calls += 1
    assert calls == manifest['physical_vbd_calls'] and calls == manifest['frames_completed'] * example.sim_substeps
    if args.demo in ('vbd.example_vbd_soft_rigid_contact', 'vbd.example_vbd_soft_rigid_mix_contact'):
        assert manifest['frames_completed'] >= example.params['settle_frames'] + example.params['lift_frames'] and moving_inputs >= example.params['lift_frames']
    for name, value in zip(('particle_q', 'particle_qd', 'body_q', 'body_qd'), previous): getattr(example.state_0, name).assign(value)
    example.sim_time = calls * example.sim_dt; example.test_final()
    arrays = {name: np.stack([state[name] for state in trajectory]) for name in trajectory[0]}
    arrays.update(history__pin_x_prev=np.asarray(pin_history), history__pin_initialized=np.asarray(initialized, dtype=bool), input_pin_x=np.asarray(input_pins), input_force=np.asarray([f[0] for f in forces]), input_torque=np.asarray([f[1] for f in forces]), source_soft_contact_count=np.asarray(contact_counts, np.int64))
    metadata = {**scene.metadata, 'teacher_iterations': config['iterations'], 'teacher_frames': manifest['frames_completed'], 'teacher_substeps': calls, 'reference_final_check': 'PASS', 'native_preserve_defaults': True, 'newton_sha': manifest['source_sha'], 'source_trace_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(), 'source_soft_material': {name: getattr(example.model, name) for name in ('soft_contact_ke', 'soft_contact_kd', 'soft_contact_mu')}, 'source_moving_pin_calls': moving_inputs, 'teacher_target': 'Authored finite-budget VBD outputs. Pin commands are external inputs; accepted pin history belongs to the new physical model, not imported solver duals.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: value.numpy() for key, value in scene.values.items()}, **source, **arrays, metadata=np.asarray(json.dumps(metadata)))
    result = {'status': 'MIXED_TEACHER_DATA_ADMITTED', 'demo': args.demo, 'path': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'teacher_frames': manifest['frames_completed'], 'teacher_substeps': calls, 'source_final_check': 'PASS', 'particle_count': n, 'body_count': b, 'pinned_particle_count': len(pins), 'moving_pin_command_calls': moving_inputs, 'maximum_source_contact_candidates': max(contact_counts), 'static_geometry_and_material_boundary_checked': True, 'derived_static_roundoff_restored': roundoff, 'chunk_hashes_checked': True, 'dynamic_state_continuity_exact': True, 'external_pin_commands_and_accepted_history_explicit': True, 'new_optimizer_updates': 0, 'new_training_admissions': 0, 'full_vbd_prediction_calls': 0, 'scope': 'Data qualification only; no mixed training count, autonomous model behavior or realtime claim.'}
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(json.dumps(result, indent=2) + '\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
