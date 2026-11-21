"""Compare complete optimized robot rollouts against accepted model recordings.

Position/orientation tolerances certify a small numerical change at saved
frames. They do not establish collision-free continuous trajectories. Timing
against historical recordings is descriptive; use benchmark_robot_step.py for
interleaved comparisons under the same GPU workload.
"""
import argparse
import json
from pathlib import Path

import numpy as np


CASES = {'franka': 'franka_soft', 'w1': 'w1_bag', 'cloth': 'franka_cloth',
         'piper': 'piper', 'w1_dual': 'w1_dual', 'franka_rigid_cloth': 'franka_rigid_cloth'}


def compare(reference, candidate):
    old = json.loads((reference / 'run.json').read_text())
    new = json.loads((candidate / 'run.json').read_text())
    checks = {
        'complete': old['status'] == new['status'] == 'COMPLETE',
        'same_checkpoint': old['checkpoint_sha256'] == new['checkpoint_sha256'],
        'same_output_frames': old['frames_completed'] == new['frames_completed'],
        'same_requested_substeps': old.get('substeps', 3) == new.get('substeps', 3),
        'same_accepted_physical_steps': old['physical_substeps'] == new['physical_substeps'],
        'same_neural_calls': old['nn_backbone_calls'] == new['nn_backbone_calls'],
        'no_native_dynamics': not any(new['forbidden_solver_calls'].values()),
        'no_teacher_reads': new['teacher_trajectory_reads'] == 0,
        'no_retries': new.get('adaptive_retries', 0) == 0,
    }
    errors = {}
    with np.load(reference / 'trajectory.npz') as a, np.load(candidate / 'trajectory.npz') as b:
        checks['same_saved_times'] = a['physical_time'].shape == b['physical_time'].shape and np.allclose(
            a['physical_time'], b['physical_time'], rtol=0, atol=1e-9)
        checks['finite'] = all(np.isfinite(b[key]).all() for key in
                               ('particle_position', 'body_pose', 'particle_velocity', 'body_velocity'))
        if checks['same_saved_times']:
            for key in ('particle_position', 'particle_velocity', 'body_velocity'):
                delta = b[key].astype(np.float64) - a[key].astype(np.float64)
                errors[key + '_max_abs'] = float(np.max(np.abs(delta), initial=0.))
                errors[key + '_rms'] = float(np.sqrt(np.mean(delta ** 2))) if delta.size else 0.
            errors['body_position_max_abs'] = float(np.max(np.abs(b['body_pose'][..., :3].astype(np.float64)
                                                               - a['body_pose'][..., :3]), initial=0.))
            qa = a['body_pose'][..., 3:].astype(np.float64)
            qb = b['body_pose'][..., 3:].astype(np.float64)
            qa /= np.linalg.norm(qa, axis=-1, keepdims=True)
            qb /= np.linalg.norm(qb, axis=-1, keepdims=True)
            qb *= np.where(np.sum(qa * qb, axis=-1, keepdims=True) < 0., -1., 1.)
            errors['body_angle_max_rad'] = float(np.max(4 * np.arctan2(
                np.linalg.norm(qa - qb, axis=-1), np.linalg.norm(qa + qb, axis=-1)), initial=0.))
            checks['position_within_0_1_mm'] = max(errors['particle_position_max_abs'], errors['body_position_max_abs']) <= 1e-4
            checks['orientation_within_0_1_degree'] = bool(errors['body_angle_max_rad'] <= np.deg2rad(.1))
            checks['velocity_components_within_0_001'] = max(errors['particle_velocity_max_abs'], errors['body_velocity_max_abs']) <= .001
        else:
            checks['matching_trajectory_required'] = False
    for name in ('quality.json',):
        if (candidate / name).exists():
            checks[name + '_accepted'] = bool(json.loads((candidate / name).read_text())['accepted'])
    return {'status': 'PASS' if all(checks.values()) else 'FAIL', 'checks': checks, 'errors': errors,
            'reference': str(reference.resolve()), 'candidate': str(candidate.resolve()),
            'historical_reference_fps': old['frames_completed'] / old['wall_seconds'],
            'candidate_fps_including_cold_execution': new['frames_completed'] / new['wall_seconds'],
            'candidate_wall_seconds': new['wall_seconds'], 'neural_calls': new['nn_backbone_calls'],
            'physical_steps': new['physical_substeps'], 'execution_policy': new.get('execution_policy', 'reference')}


def main():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=root / 'outputs/current_weight_demos_20261120/recordings')
    p.add_argument('--candidate', type=Path, required=True, help='Suite recordings directory')
    p.add_argument('--cases', nargs='+', choices=tuple(CASES), default=list(CASES))
    p.add_argument('--report', type=Path, required=True)
    args = p.parse_args()
    rows = {name: compare(args.reference / CASES[name], args.candidate / name) for name in args.cases}
    result = {'status': 'PASS' if all(r['status'] == 'PASS' for r in rows.values()) else 'FAIL',
              'cases': rows, 'timing_scope': 'Cold inclusive full inference and recording wall time; excludes video rendering. '
              'Historical FPS uses a different GPU workload and is not a controlled speedup measurement.',
              'accuracy_scope': 'Complete saved trajectories, same checkpoint and step budget; '
              '0.1 mm position, 0.1 degree orientation, 0.001 in stored velocity components (m/s or rad/s). '
              'Does not replace task-specific audits or continuous contact verification.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if result['status'] != 'PASS':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
