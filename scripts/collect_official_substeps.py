"""Collect authored VBD calls with controls, forces and auxiliary history.

This is a training-data preparation stage, not training admission or inference
acceptance. CUDA capture is disabled only to observe every original substep.
The original controllers, solver budgets, contacts and assets remain enabled.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from wp_vbd.newton_source import source_root
NEWTON = source_root()


def worker(args):
    import numpy as np
    import warp as wp
    from importlib.metadata import PackageNotFoundError,version
    from wp_vbd.official import make_example, solver_configuration_record
    from wp_vbd.reference_tape import MUTABLE_MODEL_FIELDS, TRACE_FEATURE_SET, EXTERNAL_TRACE_FEATURE_SET, EXTERNAL_INERTIAL_FIELDS, TapeWriter, arrays, auxiliary, contact_arrays, matcher_arrays, self_contact_arrays
    wp.config.kernel_cache_dir = str(ROOT / '.local/p0/warp-cache')
    wp.init()
    wp.set_device(args.device)
    directory = args.output / args.worker
    writer = TapeWriter(directory, chunk_steps=args.chunk_steps, max_bytes=args.max_gb * 1024**3)
    report = {'id': args.worker, 'status': 'RUNNING', 'frames_requested': args.frames, 'frames_completed': 0, 'training_admission': False, 'capture_preserved': False, 'authored_physics_preserved': True, 'trace_feature_set': TRACE_FEATURE_SET, 'collector_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'tape_module_sha256': hashlib.sha256((ROOT / 'src/wp_vbd/reference_tape.py').read_bytes()).hexdigest()}
    if args.external_ownership:
        report.update(trace_feature_set=EXTERNAL_TRACE_FEATURE_SET,external_inertial_fields=list(EXTERNAL_INERTIAL_FIELDS),
            external_ownership_snapshots=True,coupling_notifications_recorded=True,
            call_count_scope='VBD calls include numerical coupling replays; they are not summed as elapsed physical time.')
    def optional_version(name):
        try:return version(name)
        except PackageNotFoundError:return None
    report['environment'] = {'python': sys.executable, 'warp': wp.__version__, 'numpy': np.__version__,
                             'trimesh':optional_version('trimesh'),'torch':optional_version('torch'),
                             'mujoco': optional_version('mujoco'), 'mujoco_warp': optional_version('mujoco-warp'),
                             'overrides': os.environ.get('WP_OFFICIAL_REFERENCE_OVERRIDE'), 'device': args.device}
    begin = time.perf_counter()
    try:
        capacity={name:value for name,value in (('particle_vertex_contact_buffer_size',args.particle_vertex_capacity),
                                               ('particle_edge_contact_buffer_size',args.particle_edge_capacity)) if value is not None}
        example, solvers, arguments = make_example(args.worker, reference=True, allow_capture=False, preserve_defaults=True,contact_capacity=capacity)
        if capacity:report['contact_capacity_overrides']=capacity
        if args.test_post_step:
            if not callable(getattr(example,'test_post_step',None)) or not callable(getattr(example,'test_final',None)):
                raise ValueError('Requested original post-step and final checks are unavailable')
            report['authored_post_step_checks']={'requested':True,'calls':0}
            report['authored_final_check']='NOT_RUN'
        if args.test_final:
            if not callable(getattr(example,'test_final',None)):
                raise ValueError('Requested original final check is unavailable')
            report['authored_final_check_requested']=True
            report['authored_final_check']='NOT_RUN'
        report['collision_refresh_policy'] = 'authored'
        if args.fresh_rod_contacts:
            from wp_vbd.rod_input_policy import refresh_every_substep
            report['collision_policy_override'] = refresh_every_substep(example, args.worker, solvers)
            report['collision_refresh_policy'] = 'every_physical_substep'
        if not solvers:
            raise ValueError('No VBD instance observed')
        report.update(arguments=arguments, vbd_instances=len(solvers), source_sha=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=NEWTON, text=True).strip())
        static = {}
        controller_clock = getattr(example,'sim_time_array',None) if args.worker=='cable.example_cable_bundle_hysteresis' else None
        if controller_clock is not None:
            static['controller_initial__sim_time_array']=controller_clock.numpy().copy()
            report['controller_clock_contract']='Clock sampled after pose command and clock increment, before solver; command time is the previous sampled clock, initially controller_initial__sim_time_array.'
        definitions = []
        pipelines = []
        for owner in [example] + [solver for solver, _ in solvers]:
            for value in vars(owner).values():
                if value.__class__.__name__ == 'CollisionPipeline' and all(value is not previous for previous in pipelines):
                    pipelines.append(value)
        for slot, (solver, configuration) in enumerate(solvers):
            notifications=[]
            if args.external_ownership:
                original_notify=solver.coupling_notify_input_state_update
                def notify(state,flags,*,iteration_restart=False,dt=0.,_original=original_notify,_events=notifications):
                    _events.append({'flags':int(flags),'iteration_restart':bool(iteration_restart),'entry_dt':float(dt)})
                    return _original(state,flags,iteration_restart=iteration_restart,dt=dt)
                solver.coupling_notify_input_state_update=notify
            prefix = f'solver{slot}__'
            static.update(arrays(solver.model, prefix=prefix + 'model__'))
            static.update(arrays(solver, names=('joint_constraint_start', 'joint_constraint_dim', 'joint_material_k', 'joint_penalty_kd', 'joint_is_hard', 'joint_rest_angle', 'joint_rod_rest_kb_local', 'joint_rod_rest_twist', 'joint_dahl_eps_max', 'joint_dahl_tau'), prefix=prefix + 'definition__'))
            # GPU mesh IDs alone do not encode the asset. Retain source points
            # and faces where available; callable procedural assets retain
            # their pinned source identity and arguments as well.
            for shape, geometry in enumerate(solver.model.shape_source):
                for name in ('vertices', 'indices', 'points'):
                    value = getattr(geometry, name, None)
                    if value is not None:
                        value = value.numpy() if isinstance(value, wp.array) else np.asarray(value)
                        if not value.dtype.hasobject:
                            static[prefix + f'geometry{shape}__' + name] = value
            definitions.append({'slot': slot, 'particle_count': solver.model.particle_count, 'body_count': solver.model.body_count, 'joint_count': solver.model.joint_count, 'configuration': solver_configuration_record(configuration)})
            original = solver.step
            signature = inspect.signature(original)
            def observe(*call_args, _solver=solver, _original=original, _signature=signature, _slot=slot, _events=notifications, **call_kwargs):
                bound = _signature.bind(*call_args, **call_kwargs)
                bound.apply_defaults()
                p = bound.arguments
                dt = float(p['dt'])
                if dt <= 0:
                    raise ValueError('Nonpositive original physical step')
                state_in, state_out = p['state_in'], p['state_out']
                contacts = p['contacts'] if p['contacts'] is not None else getattr(_solver, '_pipeline_contacts', None)
                fields = {}
                def collect(owner, name, function=arrays):
                    fields.update({name + '__' + key: value for key, value in function(owner).items()})
                collect(state_in, 'input')
                if controller_clock is not None:
                    fields['controller_before__sim_time_array']=controller_clock.numpy().copy()
                control = p['control'] if p['control'] is not None else _solver.model.control(clone_variables=False)
                collect(control, 'control')
                fields.update({'model_input__' + key: value for key, value in arrays(_solver.model, names=MUTABLE_MODEL_FIELDS).items()})
                if args.external_ownership:
                    fields.update({'model_input__'+key:value for key,value in arrays(_solver.model,names=EXTERNAL_INERTIAL_FIELDS).items()})
                collect(_solver, 'aux_before', auxiliary)
                collect(contacts, 'contacts_before', contact_arrays)
                collect(_solver, 'self_contact_before', self_contact_arrays)
                collect(pipelines, 'matching_before', matcher_arrays)
                for key, value in fields.items():
                    if key.startswith('input__') and value.dtype.kind == 'f' and not np.isfinite(value).all():
                        raise ValueError(f'Nonfinite reference input: {key}')
                _original(*call_args, **call_kwargs)
                if args.external_ownership:
                    fields.update({'model_output__'+key:value for key,value in arrays(_solver.model,names=EXTERNAL_INERTIAL_FIELDS).items()})
                collect(state_out, 'output')
                collect(_solver, 'aux_after', auxiliary)
                collect(p.get('observables'), 'output_observables')
                collect(contacts, 'contacts_after', contact_arrays)
                collect(_solver, 'self_contact_after', self_contact_arrays)
                collect(pipelines, 'matching_after', matcher_arrays)
                for key, value in fields.items():
                    if key.startswith('output__') and value.dtype.kind == 'f' and not np.isfinite(value).all():
                        raise ValueError(f'Nonfinite reference output: {key}')
                metadata={'solver_slot': _slot, 'frame': report['frames_completed'], 'sequence': writer.steps, 'dt': dt, 'authored_frame_time': float(getattr(example, 'sim_time', 0.)), 'control_was_none': p['control'] is None, 'contacts_from_owned_pipeline': p['contacts'] is None and contacts is not None, 'observables_requested': sorted(str(kind) for kind in getattr(p.get('observables'), '_kinds', ()))}
                if args.external_ownership:metadata['coupling_notifications_since_last_call']=list(_events);_events.clear()
                writer.append(fields,metadata)
            solver.step = observe
        path = directory / 'static.npz'
        np.savez_compressed(path, **static)
        report.update(solvers=definitions, static={'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'fields': len(static)})
        for frame in range(args.frames):
            example.step()
            report['frames_completed'] = frame + 1
            if args.test_post_step:
                example.test_post_step()
                report['authored_post_step_checks']['calls']+=1
            if (frame + 1) % 10 == 0:
                print(args.worker, 'frames', frame + 1, 'VBD calls', writer.steps, flush=True)
        if not writer.steps:
            raise ValueError('No physical VBD calls were intercepted; capture or coupling adapter is incomplete')
        if args.test_post_step or args.test_final:
            report['authored_final_check']='RUNNING'
            example.test_final()
            report['authored_final_check']='PASS'
            maximum=getattr(example,'max_penetration',None)
            if maximum is not None:report['maximum_authored_penetration_m']=float(maximum)
        report['status'] = 'PHYSICAL_SUBSTEP_TRACE_READY'
    except Exception as error:
        if report.get('authored_final_check')=='RUNNING':report['authored_final_check']='FAIL'
        report.update(status='COLLECTION_FAILED', error=str(error), traceback=traceback.format_exc())
    report.update(seconds=time.perf_counter() - begin, scope='Offline per-call data with explicit owner/timing. External coupling schedule, self-contact candidates and restart completeness still require dedicated admission checks; never a training or acceptance claim.')
    report = writer.manifest(report)
    output = args.reports / (args.worker + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(args.worker, report['status'], writer.steps, flush=True)
    return 0 if report['status'] == 'PHYSICAL_SUBSTEP_TRACE_READY' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker')
    parser.add_argument('--device', default='cuda:0', choices=('cpu', 'cuda:0'))
    parser.add_argument('--only', nargs='+')
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--chunk-steps', type=int, default=16)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--max-gb', type=int, default=20)
    parser.add_argument('--output', type=Path, default=ROOT / '.local/official_substeps_v1')
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports/official_substeps/full')
    parser.add_argument('--overrides', type=Path, default=ROOT / '.local/official_ref_overrides')
    parser.add_argument('--retry-failed', action='store_true', help='Archive failed/outdated tapes before retrying; current successful tapes remain unchanged')
    parser.add_argument('--fresh-rod-contacts', action='store_true', help='Explicit supported cable numerical reference variant: refresh collision every physical step; keep original assets/material/control')
    parser.add_argument('--particle-vertex-capacity',type=int,help='Explicit source contact capacity expansion after a recorded overflow')
    parser.add_argument('--particle-edge-capacity',type=int,help='Explicit source contact capacity expansion after a recorded overflow')
    parser.add_argument('--test-post-step',action='store_true',help='Run original per-frame validation hook and final check, including accumulated metrics')
    parser.add_argument('--test-final',action='store_true',help='Run the original final check, also for examples without a per-frame validation hook')
    parser.add_argument('--external-ownership',action='store_true',help='Record view-local inertial properties before/after every call and coupling restart notifications (feature4)')
    args = parser.parse_args()
    from wp_vbd.reference_tape import TRACE_FEATURE_SET,EXTERNAL_TRACE_FEATURE_SET
    if args.frames < 1 or args.chunk_steps < 1:
        parser.error('frames and chunk-steps must be positive')
    if any(value is not None and value<=0 for value in (args.particle_vertex_capacity,args.particle_edge_capacity)):
        parser.error('Contact capacities must be positive')
    if args.worker:
        return worker(args)
    active_worker = None
    def stop_workers(signum, unused_frame):
        if active_worker is not None and active_worker.poll() is None:
            os.killpg(active_worker.pid, signal.SIGTERM)
            active_worker.wait(timeout=10)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop_workers)
    signal.signal(signal.SIGINT, stop_workers)
    identifiers = [row['id'] for row in json.loads((ROOT / 'configs/vbd_demo_inventory.json').read_text())['examples']]
    if args.only:
        unknown = set(args.only) - set(identifiers)
        if unknown:
            parser.error(f'Unknown VBD demos: {sorted(unknown)}')
        identifiers = [identifier for identifier in identifiers if identifier in args.only]
    args.reports.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(args.overrides) + os.pathsep + environment.get('PYTHONPATH', '')
    environment['WP_OFFICIAL_REFERENCE_OVERRIDE'] = str(args.overrides)
    results = []
    def persist_summary():
        summary = {'candidate_count': len(identifiers), 'processed': len(results), 'ready': sum(row['status'] == 'PHYSICAL_SUBSTEP_TRACE_READY' for row in results), 'training_count_from_this_collector': 0, 'results': results}
        (args.reports / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    for identifier in identifiers:
        path = args.reports / (identifier + '.json')
        if path.exists():
            row = json.loads(path.read_text())
            current_format = row.get('trace_feature_set') == (EXTERNAL_TRACE_FEATURE_SET if args.external_ownership else TRACE_FEATURE_SET)
            # Earlier collectors always used cuda:0 and did not store the
            # device. A ready CPU tape must not satisfy a later CUDA request.
            requested_device = row.get('environment', {}).get('device', 'cuda:0') == args.device
            requested_policy = row.get('collision_refresh_policy', 'authored') == ('every_physical_substep' if args.fresh_rod_contacts else 'authored')
            requested_capacity={name:value for name,value in (('particle_vertex_contact_buffer_size',args.particle_vertex_capacity),
                                                               ('particle_edge_contact_buffer_size',args.particle_edge_capacity)) if value is not None}
            capacity_matches=row.get('contact_capacity_overrides',{})==requested_capacity
            checks_match=row.get('authored_post_step_checks',{}).get('requested',False)==args.test_post_step
            if args.test_post_step:checks_match=checks_match and row.get('authored_post_step_checks',{}).get('calls',0)>=args.frames and row.get('authored_final_check')=='PASS'
            if args.test_final:checks_match=checks_match and row.get('authored_final_check')=='PASS'
            if row['status'] == 'PHYSICAL_SUBSTEP_TRACE_READY' and row['frames_completed'] >= args.frames and current_format and requested_device and requested_policy and capacity_matches and checks_match:
                results.append(row)
                persist_summary()
                continue
            if args.retry_failed and (row['status'] != 'PHYSICAL_SUBSTEP_TRACE_READY' or not current_format):
                archive = args.reports / 'prior_attempts' / f'{identifier}_{time.time_ns()}'
                archive.mkdir(parents=True)
                shutil.move(str(path), str(archive / path.name))
                log_path = args.reports / (identifier + '.log')
                if log_path.exists():
                    shutil.move(str(log_path), str(archive / log_path.name))
                partial = args.output / identifier
                if partial.exists():
                    retained = args.output / 'prior_attempts' / archive.name
                    retained.parent.mkdir(exist_ok=True)
                    shutil.move(str(partial), str(retained))
        if (args.output / identifier).exists():
            raise ValueError(f'Existing partial/short tape retained; use a fresh output path or archive it explicitly: {identifier}')
        command = ['uv', 'run', '--no-sync', '--offline', '--project', str(NEWTON), 'python', str(Path(__file__).resolve()), '--worker', identifier, '--device', args.device, '--frames', str(args.frames), '--chunk-steps', str(args.chunk_steps), '--max-gb', str(args.max_gb), '--output', str(args.output), '--reports', str(args.reports)]
        if args.fresh_rod_contacts: command.append('--fresh-rod-contacts')
        if args.test_post_step:command.append('--test-post-step')
        if args.test_final:command.append('--test-final')
        if args.external_ownership:command.append('--external-ownership')
        for flag,value in (('--particle-vertex-capacity',args.particle_vertex_capacity),('--particle-edge-capacity',args.particle_edge_capacity)):
            if value is not None:command.extend((flag,str(value)))
        with (args.reports / (identifier + '.log')).open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            active_worker = process
            try:
                process.wait(timeout=args.timeout)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait(timeout=10)
            active_worker = None
        if not path.exists():
            path.write_text(json.dumps({'id': identifier, 'status': 'WORKER_FAILED_OR_TIMEOUT', 'returncode': process.returncode, 'training_admission': False}, indent=2) + '\n')
        row = json.loads(path.read_text())
        results.append(row)
        persist_summary()
        print(identifier, row['status'], flush=True)
    return int(any(row['status'] != 'PHYSICAL_SUBSTEP_TRACE_READY' for row in results))


if __name__ == '__main__':
    raise SystemExit(main())
