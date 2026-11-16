"""Compare executed Franka experiments and screen unsuitable training references.

Task and mesh checks are separate. This report never admits a trajectory to
training: successful task behavior still needs visual and reference review.
"""
import argparse
import hashlib
import json
from pathlib import Path


def review(directory, log_path):
    import numpy as np
    run = json.loads((directory/'run.json').read_text())
    if run['status'] not in ('COMPLETE', 'FAILED', 'INTERRUPTED'):
        raise ValueError('Wait for the experiment to finish: '+str(directory))
    with np.load(directory/'trajectory.npz') as archive:
        position = archive['particle_position'].astype(np.float64)
        poses = archive['body_pose']; tets = archive['tets']
        times = archive['physical_time']; phases = archive['phase']
        targets = archive['joint_target']
    vertices = position[:,tets]
    determinant = np.linalg.det(vertices[:,:,1:]-vertices[:,:,:1])
    if not len(tets) or bool((determinant[0] == 0).any()):
        raise ValueError('Nondegenerate reference tetrahedra required')
    ratio = determinant/determinant[:1]
    finite = bool(np.isfinite(position).all() and np.isfinite(poses).all() and
                  np.isfinite(ratio).all() and np.isfinite(targets).all())
    inverted = (ratio <= 0).any(1)
    log = log_path.read_text() if log_path.exists() else ''
    overflow = [line for line in log.splitlines() if 'overflow' in line.lower()]
    centers = position.mean(1)
    hold = np.flatnonzero(phases == 'hold')
    hold_start = int(hold[0]) if len(hold) else None
    hold_seconds = float(times[-1]-times[hold_start]) if hold_start is not None else 0.
    window = (run['initial_command_time'] <= 1.3+1e-9 and
              run['initial_command_time']+times[-1] >= 3.8-1e-9 and hold_seconds >= .49)
    lift_value = float(centers[-1,2]-centers[0,2])
    lift = lift_value if np.isfinite(lift_value) else None
    # Same wrist/tool convention as franka_vbd_scene. A thrown object is
    # not a successful grasp even when its height increases dramatically.
    q = poses[:,6,3:].astype(np.float64)
    q = q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-30)
    offset = np.array([0.,0.,.22])
    tool = poses[:,6,:3]+offset+2*np.cross(q[:,:3],np.cross(q[:,:3],offset)+q[:,3:]*offset)
    relative = centers-tool
    tracking = np.linalg.norm(relative-relative[:1],axis=-1)
    grasp_tolerance = max(.03,.5*float(np.linalg.norm(np.ptp(position[0],axis=0))))
    final_tracking = float(tracking[-1]) if np.isfinite(tracking[-1]) else None
    hold_tracking = float(tracking[hold_start:].max()) if hold_start is not None and np.isfinite(tracking[hold_start:]).all() else None
    tool_error = run['tool_position_error_m']
    reasons = []
    if run['status'] != 'COMPLETE': reasons.append('Incomplete executed window')
    if not finite: reasons.append('Nonfinite recorded state/volume')
    if bool(inverted.any()): reasons.append('Nonpositive tetrahedral volume in recorded states')
    if overflow: reasons.append('Contact buffer overflow reported by the native implementation')
    if run['mode'] == 'offline_vbd_reference' and not log_path.exists():
        reasons.append('Native reference log missing; contact overflow could not be screened')
    if not window: reasons.append('Complete closing/lift/hold window not recorded')
    if lift is None or lift < .02: reasons.append('Less than 2 cm final lift in this authored 12 cm lift task')
    if final_tracking is None or final_tracking > grasp_tolerance or hold_tracking is not None and hold_tracking > grasp_tolerance:
        reasons.append('Object does not remain near the gripper; lift alone is not successful grasp')
    if not np.isfinite(tool_error) or tool_error > .05:
        reasons.append('Gross tool tracking error exceeds the authored task tolerance of 5 cm')
    native_audit = run.get('native_substep_audit')
    if native_audit and native_audit['enabled']:
        if native_audit['steps'] != run['physical_substeps']:
            reasons.append('Native substep audit is incomplete')
        if native_audit['nonpositive_tet_steps']:
            reasons.append('Nonpositive tetrahedral volume in a native physical substep')
        if native_audit['max_mimic_coordinate_error'] > 1e-5:
            reasons.append('Native ideal mimic constraint error exceeds the reference-data tolerance')
    static_digest = run.get('source_static_sha256')
    if static_digest:
        assert hashlib.sha256((directory/'source_static.npz').read_bytes()).hexdigest() == static_digest
    variant_path = directory/'experiment.json'
    variant = json.loads(variant_path.read_text()) if variant_path.exists() else None
    if variant is not None:
        payload = (directory/'instrumentation.py').read_bytes()
        assert hashlib.sha256(payload).hexdigest() == variant['instrumentation_sha256']
    manifest = json.loads((directory/'implementation_snapshot_manifest.json').read_text())
    for relative, expected in manifest.items():
        assert hashlib.sha256((directory/'implementation_snapshot'/relative).read_bytes()).hexdigest() == expected
    if run['mode'] == 'shared_network_inference':
        assert not any(run['forbidden_solver_calls'].values()) and run['teacher_trajectory_reads'] == 0
    assert run['new_optimizer_updates'] == 0
    summary = {'recording':str(directory), 'status':run['status'], 'mode':run['mode'],
               'run_sha256':hashlib.sha256((directory/'run.json').read_bytes()).hexdigest(),
               'trajectory_sha256':hashlib.sha256((directory/'trajectory.npz').read_bytes()).hexdigest(),
               'physical_dt':run['physical_dt'], 'physical_seconds':float(times[-1]),
               'device':run['device'], 'checkpoint_sha256':run['checkpoint_sha256'],
               'physical_parameters':run['physical_parameters'],
               'solver_parameters':run['solver_parameters'], 'prototype':variant,
               'ideal_mimic_count':run['ideal_mimic_count'], 'native_substep_audit':native_audit,
               'source_static_verified':bool(static_digest),
               'recorded_states':len(position),
               'adapted_mimic_projection_calls':run.get('adapted_mimic_projection_calls',0),
               'final_center_lift_m':lift, 'tool_position_error_m':tool_error if np.isfinite(tool_error) else None,
               'minimum_tet_volume_ratio':float(ratio[np.isfinite(ratio)].min()) if np.isfinite(ratio).any() else None,
               'maximum_tet_volume_ratio':float(ratio[np.isfinite(ratio)].max()) if np.isfinite(ratio).any() else None,
               'final_relative_gripper_tracking_error_m':final_tracking,
               'maximum_hold_relative_gripper_tracking_error_m':hold_tracking,
               'relative_gripper_tracking_tolerance_m':grasp_tolerance,
               'final_nonpositive_tet_count':int((ratio[-1] <= 0).sum()),
               'first_nonpositive_tet_frame':int(np.flatnonzero(inverted)[0]) if bool(inverted.any()) else None,
               'contact_buffer_overflow_messages':overflow[:10], 'overflow_message_count':len(overflow),
               'native_log_available':log_path.exists(),
               'hold_seconds':hold_seconds,
               'hold_center_vertical_change_m':float(centers[-1,2]-centers[hold_start,2]) if hold_start is not None else None,
               'automatic_task_and_mesh_screen_passed':not reasons, 'screen_failures':reasons,
               'training_admitted':False, 'new_optimizer_updates':run['new_optimizer_updates'],
               'nn_calls':run['nn_backbone_calls'], 'offline_vbd_steps':run['reference_vbd_steps'],
               'wall_seconds':run['wall_seconds'],
               'initial_state_sha256':hashlib.sha256(position[0].tobytes()+poses[0].tobytes()).hexdigest(),
               'source_snapshot_verified':True}
    return summary, targets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--case', action='append', required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists(): raise FileExistsError(args.report)
    import numpy as np
    cases = {}; baseline = None
    for name in args.case:
        row, targets = review(args.root/name, args.root/(name+'.log'))
        if baseline is None: baseline = targets
        count = min(len(baseline), len(targets))
        row['maximum_control_target_difference_from_first_case'] = float(np.abs(targets[:count]-baseline[:count]).max())
        cases[name] = row
        print(name, 'lift_m=', row['final_center_lift_m'], 'screen=', row['screen_failures'])
    result = {'status':'EXECUTED_EXPERIMENT_REVIEW', 'cases':cases,
              'criteria':'Numerical agreement with VBD is not required. Finite, orientation-preserving states, complete task motion and no native contact-buffer overflow are screened independently; small contact compression is allowed.',
              'training_admitted':False,
              'scope':'Recorded-frame mesh/task screen. A pass still requires visual inspection and reference validation; this does not prove continuous collision freedom, force convergence, realtime speed or all-demo acceptance.'}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
