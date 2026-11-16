"""Check authored data reconstruction, causal controls and stale-label rejection."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--review',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    import numpy as np
    import torch
    import warp as wp
    import newton
    sys.path.append('/home/oem/code/engine/newton/.venv/lib/python3.12/site-packages')
    from wp_vbd.franka_reference_data import load_reference,prepare
    from wp_vbd.mimic_constraint import rows as mimic_rows
    from wp_vbd.data import State
    from wp_vbd.surface_joint_data import load_reference as load_course_reference
    from review_franka_ablation import review
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    def forbidden(*unused,**kwargs):raise AssertionError('Data verification must not execute a dynamics solver')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    scene,states=load_reference(args.dataset,wp,newton)
    dispatched,course_states=load_course_reference(args.dataset,wp,newton)
    assert all(torch.equal(value,course_states[key]) for key,value in states.items())
    assert dispatched.mimic_spec is not None
    with np.load(args.recording/'trajectory.npz') as z:archive={k:z[k].copy() for k in z.files}
    assert len(states['x'])==len(archive['physical_time'])
    assert np.array_equal(states['input_joint_target'][:-1,7,0].numpy(),archive['input_joint_target_raw'][:,7])
    assert np.array_equal(states['input_joint_target'][:-1,3,3].numpy(),archive['input_joint_target_raw'][:,3])
    assert not bool(states['input_force'].any()) and not bool(states['input_torque'].any())
    with torch.no_grad():
        constraints=mimic_rows(scene,State(*(states[k] for k in ('x','q','v','omega'))))
    assert bool(scene.mimic_spec.linear.all())
    error=float((constraints.value*scene.length).abs().max());assert error<1e-6,error
    with tempfile.TemporaryDirectory(dir=ROOT/'.local') as temporary:
        temporary=Path(temporary);fixture=temporary/args.recording.name;fixture.mkdir()
        for name in ('run.json','source_static.npz','implementation_snapshot_manifest.json','implementation_snapshot'):
            (fixture/name).symlink_to((args.recording/name).resolve(),target_is_directory=(args.recording/name).is_dir())
        changed={k:v.copy() for k,v in archive.items()}
        changed['input_joint_target_raw']=np.roll(changed['input_joint_target_raw'],-1,axis=0)
        np.savez_compressed(fixture/'trajectory.npz',**changed)
        rejected=[]
        try:prepare(fixture,args.review,temporary/'stale.npz',wp,newton)
        except ValueError as e:
            assert 'changed after' in str(e);rejected.append('stale_screen')
        else:raise AssertionError('A modified tape passed its old screen')
        fresh,_=review(fixture,args.recording.parent/(args.recording.name+'.log'))
        assert fresh['automatic_task_and_mesh_screen_passed']
        renewed=temporary/'renewed.json';renewed.write_text(json.dumps({'cases':{args.recording.name:fresh}}))
        try:prepare(fixture,renewed,temporary/'future_input.npz',wp,newton)
        except ValueError as e:
            assert 'causality' in str(e);rejected.append('future_shifted_control')
        else:raise AssertionError('Future-shifted drive targets passed the causal input check')
        assert not (temporary/'stale.npz').exists() and not (temporary/'future_input.npz').exists()
    result={'status':'PASS','physical_transitions':len(states['x'])-1,
            'maximum_native_mimic_error_in_loaded_coordinates_m':error,
            'native_control_indexing_verified':True,'complete_source_asset_reconstruction':True,
            'existing_surface_course_loader_dispatch_verified':True,
            'rejected_cases':rejected,'full_vbd_calls':0,'featherstone_calls':0,'new_optimizer_updates':0}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(args.report.read_text())


if __name__=='__main__':main()
