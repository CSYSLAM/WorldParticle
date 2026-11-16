"""Generate true parameter episodes through authored rod controllers and VBD."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', required=True)
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--frames', type=int, default=120)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--iterations',type=int)
    parser.add_argument('--report',type=Path,help='Record the original terminal check separately from raw-data generation')
    args = parser.parse_args()
    import numpy as np
    import warp as wp
    import newton
    from wp_vbd.official import make_example
    from wp_vbd.rod_adapter import export_model
    from wp_vbd.reference_tape import arrays, auxiliary
    from prepare_rod_history import history, state
    wp.config.kernel_cache_dir = str(ROOT / '.local/p0/warp-cache')
    wp.init(); wp.set_device(args.device)
    example, solvers, arguments = make_example(args.demo, reference=True, preserve_defaults=True,iterations=args.iterations)
    if len(solvers) != 1:raise ValueError('Single VBD subsystem required')
    solver, configuration = solvers[0]
    if args.factor != 1.:
        value = example.model.joint_target_ke.numpy() * args.factor
        example.model.joint_target_ke.assign(value)
        solver.notify_model_changed(newton.ModelFlags.JOINT_DOF_PROPERTIES)
    static, metadata = export_model(example.model, configuration, args.demo, newton, example.sim_dt)
    ids, fixed = static['source_joint_ids'], static['kinematic_ids']
    trajectory = []; memories = []; controls_x = []; controls_q = []; forces = []; torques = []
    original = solver.step
    def record(state_in, state_out, control, contacts, dt, **kwargs):
        if contacts is not None:raise ValueError('This no-contact rod contract cannot discard contacts')
        if dt != example.sim_dt:raise ValueError('Authored timestep changed')
        fields = {'input__' + key: value for key, value in arrays(state_in).items()}
        fields.update({'aux_before__' + key: value for key, value in auxiliary(solver).items()})
        if not trajectory:
            trajectory.append(state(fields, 'input')); memories.append(history(fields, 'aux_before', ids))
        before = state(fields, 'input')
        mobile = static['movable']
        for key in ('x', 'q'):
            if np.max(np.abs(before[key][mobile] - trajectory[-1][key][mobile])) > 2e-6:raise ValueError('Unexpected dynamic pose edit')
        if control is not None and np.any(control.joint_f.numpy()):raise ValueError('Joint actuation needs a wrench adapter')
        controls_x.append(before['x'][fixed].copy()); controls_q.append(before['q'][fixed].copy())
        wrench = fields['input__body_f']; forces.append(wrench[:, :3]); torques.append(wrench[:, 3:])
        original(state_in, state_out, control, contacts, dt, **kwargs)
        fields.update({'output__' + key: value for key, value in arrays(state_out).items()})
        fields.update({'aux_after__' + key: value for key, value in auxiliary(solver).items()})
        trajectory.append(state(fields, 'output')); memories.append(history(fields, 'aux_after', ids))
    solver.step = record
    for frame in range(args.frames):
        example.step()
        if (frame + 1) % 60 == 0:print(args.demo, args.factor, frame + 1, flush=True)
    result = {key: np.stack([s[key] for s in trajectory]) for key in trajectory[0]}
    result.update({'history__' + key: np.stack([s[key] for s in memories]) for key in memories[0]})
    result.update(input_x=np.stack(controls_x), input_q=np.stack(controls_q), input_force=np.stack(forces), input_torque=np.stack(torques))
    if not all(np.isfinite(value).all() for value in result.values()):raise ValueError('Nonfinite teacher episode')
    target='Original authored iteration budget' if args.iterations is None else 'Explicit reference iteration-budget audit, not a convergence certificate'
    metadata.update(index=0, parameter_factor=args.factor, teacher_frames=args.frames, teacher_substeps=len(forces), teacher_iterations=solver.iterations, teacher_device=args.device, arguments=arguments, teacher_target=target+', rescaled physical stiffness with caches explicitly refreshed')
    final_check='NOT_RUN';task_metrics=None
    if args.report:
        try:example.test_final();final_check='PASS'
        except Exception as error:final_check='FAIL: '+str(error)
        if hasattr(example,'_measure_twist_profile'):
            task_metrics={case['label']:example._measure_twist_profile(case).tolist() for case in example.cases}
        metadata['reference_final_check']=final_check
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **static, **result, metadata=np.asarray(json.dumps(metadata, default=str)))
    if args.report:
        import hashlib
        report={'status':'REFERENCE_EPISODE_RECORDED','demo':args.demo,'frames':args.frames,'physical_substeps':len(forces),'dt':example.sim_dt,'iterations':solver.iterations,'device':args.device,'parameter_factor':args.factor,'test_final':final_check,'twist_profiles_radians':task_metrics,'dataset':str(args.output),'dataset_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'scope':'Raw source reference and original assertions. Original terminal checks and reference convergence certification are distinct.'}
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print('saved', args.output, len(forces), flush=True)


if __name__ == '__main__':main()
