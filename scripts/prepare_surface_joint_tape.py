"""Qualify complete authored weighted-contact gripper episodes for training."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), '/home/oem/code/engine/newton']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', required=True); parser.add_argument('--tape', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.report.exists(): raise FileExistsError('Preserve data qualifications')
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.reference_tape import read_tape
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_data import STATIC_FIELDS, JOINT_FIELDS, restore_joint_source
    from wp_vbd.data import State
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = str(ROOT/'.local/p0/warp-cache'); wp.init(); wp.set_device('cpu')
    manifest_path = args.tape/'manifest.json'; manifest = json.loads(manifest_path.read_text())
    assert manifest['id'] == args.demo and manifest['status'] == 'PHYSICAL_SUBSTEP_TRACE_READY'
    assert manifest['vbd_instances'] == 1 and manifest['source_sha'] == '25298aea1691e060c86f95e8d2b773c898c006d3'
    example, solvers, _ = make_example(args.demo, reference=False, preserve_defaults=True)
    assert len(solvers) == 1 and example.contacts._enable_rigid_soft_full_surface_contact
    configuration = solvers[0][1]; assert configuration == manifest['solvers'][0]['configuration']
    assert manifest['frames_completed'] == example.params['num_frames'] == 360
    static_path = args.tape/manifest['static']['path']
    assert hashlib.sha256(static_path.read_bytes()).hexdigest() == manifest['static']['sha256']
    with np.load(static_path, allow_pickle=False) as archive:
        # The pinned Newton property is an exact alias of joint_q_start;
        # vars(model) records the underlying array, not its property name.
        aliases = {'joint_target_q_start':'joint_q_start'}
        source = {'source__'+name: archive['solver0__model__'+aliases.get(name, name)].copy()
                  for name in dict.fromkeys((*STATIC_FIELDS, *JOINT_FIELDS)) if getattr(example.model, name, None) is not None}
        source.update({key.removeprefix('solver0__'): archive[key].copy() for key in archive.files if key.startswith('solver0__geometry')})
    source = {('source_'+key if key.startswith('geometry') else key): value for key, value in source.items()}
    roundoff = restore_joint_source(example.model, source)
    scene = export_model(example.model, configuration, args.demo, newton, example.sim_dt, 'cpu')
    n = example.model.particle_count; b = example.model.body_count
    pins = scene.fixed_particle_ids.numpy(); mobile = scene.movable[:n].numpy()
    def convert(arguments):
        particle, velocity, pose, body_velocity = arguments
        x = np.zeros((scene.n, 3), np.float64); v = x.copy(); omega = x.copy()
        q = np.zeros((scene.n, 4), np.float64); q[:, 3] = 1.
        x[:n] = particle; v[:n] = velocity
        rotation = pose[:, 3:].astype(np.float64); rotation /= np.linalg.norm(rotation, axis=-1, keepdims=True)
        com = scene.body_com[n:n+b].numpy()
        x[n:n+b] = pose[:, :3]+com+2*np.cross(rotation[:, :3], np.cross(rotation[:, :3], com)+rotation[:, 3:]*com)
        q[n:n+b] = rotation; v[n:n+b] = body_velocity[:, :3]; omega[n:n+b] = body_velocity[:, 3:]
        return dict(x=x, q=q, v=v, omega=omega)
    spec = scene.joint_spec
    valid = spec.dof_index.numpy() >= 0; index = spec.dof_index.numpy(); qindex = spec.target_index.numpy()
    used = np.zeros(example.model.joint_dof_count, bool); used[index[valid]] = True
    def controls(data):
        raw = [data['control__'+name] for name in ('joint_target_q', 'joint_target_qd', 'joint_f')]
        assert all(np.isfinite(value).all() for value in raw) and not raw[2][~used].any()
        result = []
        for value, selected in zip(raw, (qindex, index, index)):
            out = np.zeros(valid.shape, np.float64); out[valid] = value[selected[valid]]; result.append(out)
        return result
    single = spec.angular_mode == 1
    angle = spec.coordinate_offset[None, :, 3].clone()
    def winding(state):
        nonlocal angle
        if bool(single.any()):
            current = State(*(torch.as_tensor(state[key])[None] for key in ('x', 'q', 'v', 'omega')))
            value = scene.joint_coordinates(current, angle).axis[..., 3]
            angle = torch.where(single, value, 0.)
        else: angle = torch.zeros_like(angle)
        return angle[0].numpy().copy()
    trajectory = []; pin_history = []; initialized = []; angle_history = []; pin_input = []
    control_input = []; forces = []; feature_counts = []; previous = None; calls = 0; target_changes = 0; prior_target = None
    native_names = ('particle_q', 'particle_qd', 'body_q', 'body_qd')
    for data, timing in read_tape(args.tape):
        assert timing['sequence'] == calls and timing['solver_slot'] == 0 and timing['dt'] == example.sim_dt
        assert not timing['observables_requested']
        arguments = [data['input__'+key] for key in native_names]
        output = [data['output__'+key] for key in native_names]
        assert all(np.isfinite(value).all() for value in (*arguments, *output))
        assert np.array_equal(arguments[0][pins], output[0][pins])
        if previous is None:
            start = convert(arguments); trajectory.append(start); angle_history.append(winding(start))
            pin_history.append(arguments[0][pins].astype(np.float64)); initialized.append(False)
        else:
            assert np.array_equal(previous[0][mobile], arguments[0][mobile]) and np.array_equal(previous[1][mobile], arguments[1][mobile])
            assert np.array_equal(previous[2], arguments[2]) and np.array_equal(previous[3], arguments[3])
        for name, value in data.items():
            if not name.startswith('model_input__'): continue
            original = getattr(example.model, name.removeprefix('model_input__'), None)
            if original is None: continue
            current = original.numpy() if hasattr(original, 'numpy') else np.asarray(original)
            assert np.array_equal(value, current), 'Unrepresented mutable physical input: '+name
        command = controls(data); control_input.append(command)
        if prior_target is not None and not np.array_equal(prior_target, command[0]): target_changes += 1
        prior_target = command[0]
        force = np.zeros((scene.n, 3), np.float64); torque = force.copy()
        force[:n] = data['input__particle_f']; force[n:n+b] = data['input__body_f'][:, :3]
        torque[n:n+b] = data['input__body_f'][:, 3:]; forces.append((force, torque))
        count = int(data['contacts_before__soft_contact_count'][0])
        sizes = (data['contacts_before__soft_contact_indices'][:count] >= 0).sum(-1)
        feature_counts.append([int((sizes == size).sum()) for size in (1, 2, 3)])
        current = convert(output); trajectory.append(current); angle_history.append(winding(current))
        pin_history.append(output[0][pins].astype(np.float64)); initialized.append(True)
        pin_input.append(arguments[0][pins].astype(np.float64)); previous = output; calls += 1
    assert calls == manifest['physical_vbd_calls'] == 360*example.sim_substeps == 7200 and target_changes > 0
    for name, value in zip(native_names, previous): getattr(example.state_0, name).assign(value)
    example.sim_time = calls*example.sim_dt; example.test_final()
    arrays = {name: np.stack([state[name] for state in trajectory]) for name in trajectory[0]}
    arrays.update(history__pin_x_prev=np.asarray(pin_history), history__pin_initialized=np.asarray(initialized, dtype=bool),
                  history__joint_angle_prev=np.asarray(angle_history), input_pin_x=np.asarray(pin_input),
                  input_force=np.asarray([f[0] for f in forces]), input_torque=np.asarray([f[1] for f in forces]),
                  source_soft_feature_count=np.asarray(feature_counts, np.int64))
    for number, name in enumerate(('input_joint_target', 'input_joint_target_rate', 'input_joint_effort')):
        arrays[name] = np.asarray([value[number] for value in control_input])
    metadata = {**scene.metadata, 'teacher_iterations': configuration['iterations'], 'teacher_frames': 360,
                'teacher_substeps': calls, 'reference_final_check': 'PASS', 'native_preserve_defaults': True,
                'newton_sha': manifest['source_sha'], 'source_trace_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'source_field_aliases': aliases,
                'source_soft_material': {name: getattr(example.model, name) for name in ('soft_contact_ke', 'soft_contact_kd', 'soft_contact_mu')},
                'teacher_target': 'Authored finite-budget VBD states, not a converged-teacher claim. Explicit joint controls and accepted pin/winding history belong to the new physical schema.'}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: value.numpy() for key, value in scene.values.items()},
                        **{'joint__'+key: value.numpy() for key, value in vars(spec).items()},
                        **source, **arrays, metadata=np.asarray(json.dumps(metadata)))
    result = {'status': 'SURFACE_JOINT_TEACHER_DATA_QUALIFIED', 'demo': args.demo, 'path': str(args.output),
              'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(), 'teacher_frames': 360,
              'teacher_substeps': calls, 'source_final_check': 'PASS', 'target_changes': target_changes,
              'particle_count': n, 'body_count': b, 'finite_joint_count': len(spec.owners),
              'source_soft_feature_occurrences': np.asarray(feature_counts).sum(0).tolist(),
              'static_source_roundoff_restored': roundoff, 'chunk_hashes_verified': True,
              'source_field_aliases': aliases,
              'native_state_continuity_exact': True, 'control_force_and_accepted_history_channels_explicit': True,
              'new_optimizer_updates': 0, 'new_training_admissions': 0, 'full_vbd_prediction_calls': 0,
              'scope': 'Reference data qualification only. It does not certify optimizer sampling, autonomous model tasks, continuous geometry guarantees or realtime.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n'); print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
