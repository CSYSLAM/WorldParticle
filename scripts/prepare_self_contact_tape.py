"""Qualify self-contact episodes with causal particle and owned free-body states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), '/home/oem/code/engine/newton']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--tape', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--motion-budget', type=float, default=.04)
    ownership=parser.add_mutually_exclusive_group()
    ownership.add_argument('--owned-free-bodies',action='store_true',help='Also admit dynamic owned FREE bodies with exact native continuity and explicit COM force/torque channels')
    ownership.add_argument('--prescribed-massless-bodies',action='store_true',help='Explicit authored massless FREE pose commands, accepted COM history and first-call rebaseline')
    args = parser.parse_args()
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Preserve reference qualification evidence')
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example, solver_configuration_record
    from wp_vbd.reference_tape import read_tape
    from wp_vbd.self_contact_scene import export_model, authored_full_surface
    from wp_vbd.surface_joint_data import STATIC_FIELDS, JOINT_FIELDS, restore_joint_source

    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = str(ROOT/'.local/p0/warp-cache')
    wp.init()
    wp.set_device('cpu')
    manifest_path = args.tape/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    assert manifest['id'] == args.demo and manifest['status'] == 'PHYSICAL_SUBSTEP_TRACE_READY'
    assert manifest['vbd_instances'] == 1 and manifest['source_sha'] == '25298aea1691e060c86f95e8d2b773c898c006d3'
    capacity=manifest.get('contact_capacity_overrides',{})
    example, configurations, _ = make_example(args.demo, reference=False, preserve_defaults=True,contact_capacity=capacity)
    asset = example.model
    if len(configurations) != 1:raise ValueError('Exactly one owned self-contact subsystem is required')
    n,b=asset.particle_count,asset.body_count
    if not (args.owned_free_bodies or args.prescribed_massless_bodies) and (b or asset.joint_count):
        raise ValueError('Rigid ownership requires an explicit owned-free or prescribed-massless contract')
    if args.owned_free_bodies or args.prescribed_massless_bodies:
        if not b or not np.all(asset.joint_type.numpy()==int(newton.JointType.FREE)):
            raise ValueError('This owned-body data contract requires actual FREE rigid bodies')
        if np.any(asset.joint_target_ke.numpy()) or np.any(asset.joint_target_kd.numpy()):
            raise ValueError('Generalized FREE drives need a separate force adapter')
    configuration = configurations[0][1]
    assert solver_configuration_record(configuration) == manifest['solvers'][0]['configuration']
    static_path = args.tape/manifest['static']['path']
    assert hashlib.sha256(static_path.read_bytes()).hexdigest() == manifest['static']['sha256']
    aliases = {'joint_target_q_start': 'joint_q_start'}
    with np.load(static_path, allow_pickle=False) as archive:
        source = {'source__'+name: archive['solver0__model__'+aliases.get(name, name)].copy()
                  for name in dict.fromkeys((*STATIC_FIELDS, *JOINT_FIELDS)) if getattr(asset, name, None) is not None}
        source.update({key.replace('solver0__geometry', 'source_geometry'): archive[key].copy()
                       for key in archive.files if key.startswith('solver0__geometry')})
    roundoff = restore_joint_source(asset, source)
    scene = export_model(asset, configuration, args.demo, newton, example.sim_dt,
                          motion_budget=args.motion_budget, full_surface=authored_full_surface(example),
                          prescribed_massless_bodies=args.prescribed_massless_bodies)
    if b and not args.prescribed_massless_bodies and (len(scene.joint_spec.owners) or not bool(scene.movable[n:n+b].all())):
        raise ValueError('Prescribed, externally owned or finite-joint bodies require their explicit history contract')
    checks=manifest.get('authored_post_step_checks',{})
    if callable(getattr(example,'test_post_step',None)):
        if not checks.get('requested') or checks.get('calls')!=manifest['frames_completed'] or manifest.get('authored_final_check')!='PASS':
            raise ValueError('Original accumulated post-step metrics and final check must be captured, not reinitialized at qualification')
    defaults=asset.control(clone_variables=False)
    free_targets={name:getattr(defaults,name).numpy().copy() for name in ('joint_target_q','joint_target_qd')} if b else {}
    def forbidden(*unused, **kwargs):
        raise AssertionError('Full VBD forbidden during data qualification')
    newton.solvers.SolverVBD.__init__ = forbidden
    newton.solvers.SolverVBD.step = forbidden
    pins = scene.fixed_particle_ids.numpy()
    mobile = scene.movable[:n].numpy()
    trajectory, forces, torques, pin_history, pin_input = [], [], [], [], []
    previous = None
    calls = 0;replayed_checks=0
    body_history=[];body_initialized=[];body_commands={key:[] for key in ('x','q','v','omega')}
    maximum_prescribed_velocity_error=0.

    def convert(position, velocity, body_pose=None, body_velocity=None):
        x = np.zeros((scene.n, 3), np.float64)
        x[:n] = position
        v = np.zeros_like(x)
        v[:n] = velocity
        q = np.zeros((scene.n, 4), np.float64)
        q[:, 3] = 1.
        omega=np.zeros_like(x)
        if b:
            from wp_vbd import math3d as so3
            pose=torch.as_tensor(body_pose).double();rotation=so3.normalize(pose[:,3:])
            if not bool((pose[:,3:].norm(dim=-1)>1e-12).all()):raise ValueError('Degenerate source rigid quaternion')
            x[n:n+b]=(pose[:,:3]+so3.rotate(rotation,scene.body_com[n:n+b])).numpy()
            q[n:n+b]=rotation.numpy();v[n:n+b]=body_velocity[:,:3];omega[n:n+b]=body_velocity[:,3:]
        return dict(x=x, q=q, v=v, omega=omega)

    for data, timing in read_tape(args.tape):
        assert timing['sequence'] == calls and timing['solver_slot'] == 0 and timing['dt'] == example.sim_dt
        assert not timing['observables_requested']
        incoming = (data['input__particle_q'], data['input__particle_qd'])
        output = (data['output__particle_q'], data['output__particle_qd'])
        if b:
            incoming+= (data['input__body_q'],data['input__body_qd'])
            output+= (data['output__body_q'],data['output__body_qd'])
        assert all(np.isfinite(value).all() for value in (*incoming, *output))
        if previous is None:
            trajectory.append(convert(*incoming))
            pin_history.append(incoming[0][pins].astype(float))
        else:
            assert all(np.array_equal(a[mobile], c[mobile]) for a, c in zip(previous[:2], incoming[:2]))
            if b and not args.prescribed_massless_bodies:assert all(np.array_equal(a,c) for a,c in zip(previous[2:],incoming[2:])), 'Unrepresented rigid pose/velocity edit between physical calls'
        if args.prescribed_massless_bodies:
            from wp_vbd import math3d as so3
            assert np.array_equal(incoming[2],output[2]),'A prescribed pose changed during the source solve'
            if previous is not None:
                assert np.array_equal(previous[2],data['aux_before__body_q_prev'])
                assert np.array_equal(previous[3],incoming[3]),'Unrepresented prescribed velocity edit'
            rebaseline=data['aux_before___rigid_pose_rebaseline_mask']
            assert np.array_equal(rebaseline,np.full(b,calls==0)), 'Unexpected prescribed pose-history reset'
            commanded=convert(*incoming)
            old=convert(incoming[0],incoming[1],data['aux_before__body_q_prev'],incoming[3])
            if calls==0:
                body_history.append({key:old[key][n:n+b].copy() for key in ('x','q')});body_initialized.append(False)
            for key in body_commands:body_commands[key].append(commanded[key][n:n+b].copy())
            baseline=commanded if calls==0 else old
            measured=(commanded['x'][n:n+b]-baseline['x'][n:n+b])/example.sim_dt
            turn=so3.log(so3.multiply(torch.from_numpy(commanded['q'][n:n+b]),so3.conjugate(torch.from_numpy(baseline['q'][n:n+b])))).numpy()/example.sim_dt
            error=max(float(np.abs(measured-output[3][:,:3]).max()),float(np.abs(turn-output[3][:,3:]).max()))
            maximum_prescribed_velocity_error=max(maximum_prescribed_velocity_error,error)
            assert error<2e-6,'Native prescribed velocity disagrees with causal accepted COM history'
            assert np.array_equal(data['aux_after__body_q_prev'],output[2])
            assert not np.any(data['aux_after___rigid_pose_rebaseline_mask'])
            body_history.append({key:commanded[key][n:n+b].copy() for key in ('x','q')});body_initialized.append(True)
        assert np.array_equal(incoming[0][pins], output[0][pins])
        for name, value in data.items():
            if b and name.removeprefix('control__') in free_targets and name.startswith('control__'):
                assert np.array_equal(value,free_targets[name.removeprefix('control__')]),'Unrepresented generalized target change'
                continue
            if name.startswith('control__') and value.size and np.any(value != 0):
                raise ValueError('Unrepresented nonzero particle activation: '+name)
            if not name.startswith('model_input__'):
                continue
            current = getattr(asset, name.removeprefix('model_input__'), None)
            if current is None:
                if value.size:
                    raise ValueError('Unrepresented model input: '+name)
                continue
            current = current.numpy() if hasattr(current, 'numpy') else np.asarray(current)
            assert np.array_equal(current, value), 'Mutable physical input needs a new explicit channel: '+name
        force = np.zeros((scene.n, 3), np.float64)
        torque=np.zeros_like(force)
        force[:n] = data['input__particle_f']
        if b:
            wrench=data['input__body_f'];force[n:n+b]=wrench[:,:3];torque[n:n+b]=wrench[:,3:]
        assert np.isfinite(force).all() and np.isfinite(torque).all()
        forces.append(force)
        torques.append(torque)
        pin_input.append(incoming[0][pins].astype(float))
        trajectory.append(convert(*output))
        pin_history.append(output[0][pins].astype(float))
        previous = output
        calls += 1
        if checks.get('requested') and calls%example.sim_substeps==0:
            example.state_0.particle_q.assign(output[0]);example.state_0.particle_qd.assign(output[1])
            if b:
                example.state_0.body_q.assign(output[2]);example.state_0.body_qd.assign(output[3])
            example.test_post_step();replayed_checks+=1
    assert calls == manifest['physical_vbd_calls'] == manifest['frames_completed']*example.sim_substeps
    assert calls > 2
    example.state_0.particle_q.assign(previous[0])
    example.state_0.particle_qd.assign(previous[1])
    if b:
        example.state_0.body_q.assign(previous[2]);example.state_0.body_qd.assign(previous[3])
    if checks.get('requested'):
        assert replayed_checks==manifest['frames_completed']
        if hasattr(example,'max_penetration'):
            assert example.max_penetration==manifest['maximum_authored_penetration_m'],'Original accumulated metric changed when replayed from recorded per-frame native poses'
    example.sim_time = calls*example.sim_dt
    example.test_final()
    arrays = {name: np.stack([state[name] for state in trajectory]) for name in trajectory[0]}
    arrays.update(history__pin_x_prev=np.asarray(pin_history), history__pin_initialized=np.ones(calls+1, bool),
                  history__joint_angle_prev=np.zeros((calls+1, 0)), input_pin_x=np.asarray(pin_input),
                  input_joint_target=np.zeros((calls, 0, 6)), input_joint_target_rate=np.zeros((calls, 0, 6)),
                  input_joint_effort=np.zeros((calls, 0, 6)), input_force=np.asarray(forces),
                  input_torque=np.asarray(torques))
    if args.prescribed_massless_bodies:
        arrays.update(history__prescribed_x_prev=np.stack([row['x'] for row in body_history]),
                      history__prescribed_q_prev=np.stack([row['q'] for row in body_history]),
                      history__prescribed_initialized=np.asarray(body_initialized,dtype=bool))
        arrays.update({'input_prescribed_'+key:np.stack(value) for key,value in body_commands.items()})
    metadata = {**scene.metadata, 'teacher_iterations': configuration['iterations'],
                'teacher_frames': manifest['frames_completed'], 'teacher_substeps': calls,
                'reference_final_check': 'PASS', 'native_preserve_defaults': True,
                'newton_sha': manifest['source_sha'], 'source_field_aliases': aliases,
                'source_trace_manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                'source_soft_material': {name: getattr(asset, name) for name in ('soft_contact_ke', 'soft_contact_kd', 'soft_contact_mu')},
                'teacher_target': 'Authored finite-budget states; no converged-teacher claim. New reciprocal EE topology and independent path geometry are explicit research model choices, not hidden source equivalence.'}
    if capacity:metadata['reference_contact_capacity_overrides']=capacity
    if b:metadata.update(source_post_step_checks=checks,
                         replayed_original_post_step_checks=replayed_checks,
                         source_maximum_authored_penetration_m=manifest.get('maximum_authored_penetration_m'))
    if args.prescribed_massless_bodies:metadata['prescribed_body_data_schema']='commanded_com_pose_accepted_history_rebaseline_v122'
    elif b:metadata['owned_free_body_data_schema']='continuous_com_pose_velocity_wrench_v94'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{key: value.numpy() for key, value in scene.values.items()},
                        **{'joint__'+key: value.numpy() for key, value in vars(scene.joint_spec).items()},
                        **source, **arrays, metadata=np.asarray(json.dumps(metadata)))
    report = {'status': 'SELF_CONTACT_TEACHER_DATA_QUALIFIED', 'demo': args.demo,
              'path': str(args.output), 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
              'teacher_frames': manifest['frames_completed'], 'teacher_substeps': calls,
              'teacher_iterations': configuration['iterations'], 'source_final_check': 'PASS',
              'particle_count': n, 'body_count': b, 'static_source_roundoff_restored': roundoff,
              'authored_shape_full_surface': scene.metadata['self_shape_full_surface'],
              'source_manifest_sha256': metadata['source_trace_manifest_sha256'],
              'chunk_hashes_verified': True, 'native_mobile_continuity_bitexact': True,
              'accepted_history_and_force_pin_inputs_explicit': True,
              'initial_pin_history_initialized_before_first_command': True,
              'new_optimizer_updates': 0, 'new_training_admissions': 0, 'full_vbd_prediction_calls': 0,
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'Original full trace, causal state/input schema and final check only. No candidate or neural convergence qualification and no selected optimizer admission. Loaded self-contact coverage must be audited separately.'}
    if b:report.update(native_body_continuity_bitexact=True,body_position_representation='com',
                       owned_body_force_torque_channels_explicit=True,source_post_step_checks=checks,
                       replayed_original_post_step_checks=replayed_checks,
                       source_maximum_authored_penetration_m=manifest.get('maximum_authored_penetration_m'))
    if args.prescribed_massless_bodies:
        report.update(native_body_continuity_bitexact=False,
                      prescribed_pose_commands_and_accepted_history_explicit=True,prescribed_first_call_rebaseline=True,
                      maximum_prescribed_velocity_reconstruction_error=maximum_prescribed_velocity_error,
                      native_rigid_input_contract='authored_massless_pose_commands_and_continuous_incoming_velocity',
                      scope='Original full trace, explicit prescribed COM commands/accepted history, source rebaseline, causal particle/force inputs and replayed final check only. No candidate, neural resource, complete model quality or training admission.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
