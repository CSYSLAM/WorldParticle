"""Convert complete pure-rod call traces into causal shared-training episodes."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), '/home/oem/code/engine/newton']

import numpy as np
import newton

from wp_vbd.reference_tape import TRACE_FEATURE_SET, read_tape
from wp_vbd.rod_adapter import export_fields

DEMOS = (
    'vbd.example_cable_bend_stiffness',
    'vbd.example_cable_bend_twist_analytic',
    'vbd.example_cable_dahl_hysteresis',
    'vbd.example_cable_michell_threshold',
    'vbd.example_cable_torsion_material_mapping',
    'vbd.example_cable_twist_transfer',
)


def history(fields, prefix, ids):
    get = lambda name: fields[prefix + '__' + name]
    previous = get('body_q_prev')
    return {'dual': np.concatenate((get('joint_lambda_lin')[ids], get('joint_lambda_ang')[ids]), -1), 'kappa': get('joint_kappa_prev')[ids], 'increment': get('joint_dkappa_prev')[ids], 'stress': get('joint_sigma_prev')[ids], 'previous_x': previous[:, :3], 'previous_q': previous[:, 3:], 'reset': np.asarray([bool(get('_rigid_pose_rebaseline_mask').any())])}


def state(fields, prefix):
    poses, velocities = fields[prefix + '__body_q'], fields[prefix + '__body_qd']
    return dict(x=poses[:, :3], q=poses[:, 3:], v=velocities[:, :3], omega=velocities[:, 3:])


def convert(directory, destination):
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['status'] != 'PHYSICAL_SUBSTEP_TRACE_READY' or manifest.get('trace_feature_set') != TRACE_FEATURE_SET:
        raise ValueError('Complete current-schema physical trace required')
    if manifest['vbd_instances'] != 1:raise ValueError('Coupled VBD slots need the coupling operator')
    definition = manifest['solvers'][0]
    with np.load(directory / 'static.npz', allow_pickle=False) as archive:
        if hashlib.sha256((directory / 'static.npz').read_bytes()).hexdigest() != manifest['static']['sha256']:raise ValueError('Static definition changed')
        source = {key.removeprefix('solver0__model__'): archive[key].copy() for key in archive.files if key.startswith('solver0__model__')}
    config = {**definition['configuration'], 'free_joint_type': int(newton.JointType.FREE), 'rod_joint_type': int(newton.JointType.ROD)}
    records = iter(read_tape(directory)); first, first_meta = next(records)
    static, metadata = export_fields(source, config, manifest['id'], first_meta['dt'])
    ids, fixed = static['source_joint_ids'], static['kinematic_ids']
    trajectory = [state(first, 'input')]; memories = [history(first, 'aux_before', ids)]
    commands_x, commands_q, forces, torques = [], [], [], []
    import itertools
    for fields, step in itertools.chain(((first, first_meta),), records):
        if step['solver_slot'] != 0 or step['dt'] != first_meta['dt']:raise ValueError('Unexpected solver/time change')
        for name, array in fields.items():
            if name.startswith(('contacts_before__', 'contacts_after__')) and name.endswith('_contact_count') and np.any(array):raise ValueError('Active contacts need the contact-history path')
        if step.get('contacts_from_owned_pipeline'):raise ValueError('Owned collision pipelines require contact admission')
        if np.any(fields.get('control__joint_f', np.zeros(1))):raise ValueError('Joint actuation needs wrench conversion')
        before = state(fields, 'input')
        for key in ('x', 'q'):
            if np.max(np.abs(before[key][static['movable']] - trajectory[-1][key][static['movable']])) > 2e-6:raise ValueError('A controller changes dynamic pose outside the boundary contract')
        commands_x.append(before['x'][fixed].copy()); commands_q.append(before['q'][fixed].copy())
        wrench = fields['input__body_f']; forces.append(wrench[:, :3].copy()); torques.append(wrench[:, 3:].copy())
        trajectory.append(state(fields, 'output')); memories.append(history(fields, 'aux_after', ids))
    arrays = {key: np.stack([row[key] for row in trajectory]) for key in trajectory[0]}
    arrays.update({'history__' + key: np.stack([row[key] for row in memories]) for key in memories[0]})
    arrays.update(input_x=np.stack(commands_x), input_q=np.stack(commands_q), input_force=np.stack(forces), input_torque=np.stack(torques))
    if not all(np.isfinite(value).all() for value in arrays.values()):raise ValueError('Nonfinite physical episode')
    metadata.update(index=0, teacher_substeps=len(forces), teacher_frames=manifest['frames_completed'], teacher_iterations=config['iterations'], source_sha=manifest['source_sha'], source_trace=str(directory), source_trace_manifest_sha256=hashlib.sha256((directory / 'manifest.json').read_bytes()).hexdigest(), external_inputs='Current external wrenches and fixed boundary poses only; complete physical/numeric history is state', initial_state_representation='body origin + SO(3), COM linear/world angular velocities')
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **static, **arrays, metadata=np.asarray(json.dumps(metadata)))
    return {'demo': manifest['id'], 'path': str(destination.relative_to(ROOT)), 'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(), 'substeps': len(forces), 'dt': metadata['dt'], 'status': 'PASS'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--only', nargs='+', default=list(DEMOS))
    args = parser.parse_args()
    results = []
    for identifier in args.only:
        try:row = convert(ROOT / '.local/official_substeps_v1' / identifier, ROOT / '.local/rod_history_v9' / (identifier + '_train.npz'))
        except Exception as error:row = {'demo': identifier, 'status': 'ADAPTER_REQUIRED', 'error': str(error)}
        results.append(row); print(row, flush=True)
    output = ROOT / 'reports/rod_history_v9'; output.mkdir(exist_ok=True)
    (output / 'conversion.json').write_text(json.dumps({'results': results}, indent=2) + '\n')


if __name__ == '__main__':main()
