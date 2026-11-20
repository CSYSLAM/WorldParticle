"""Force one diagnostic rejection in a real native/model step and verify recovery.

This is a fault-injection regression, never a demo or a training trajectory.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--reject-all',action='store_true',help='Verify rollback after every candidate is marked rejected')
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    (args.output/'fault_injection.json').write_text(json.dumps({
        'synthetic_rejection':True,'purpose':'recovery regression only',
        'policy':'All real evaluations are marked rejected.' if args.reject_all else
                 'First real model evaluation is marked rejected; subsequent evaluations use original admission.'},indent=2)+'\n')
    import torch
    import numpy as np
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    from wp_vbd.mixed_contact_runtime import MixedContactExampleSolver
    from wp_vbd.mixed_solver import MixedSolver
    original=MixedSolver.forward
    original_write=MixedContactExampleSolver._write_native_output
    native_writes=[]
    def observed_write(self,*values):
        native_writes.append(True)
        return original_write(self,*values)
    MixedContactExampleSolver._write_native_output=observed_write
    starts=[];evaluations=[]
    def injected(self,scene,start,dt,**kwargs):
        if starts:
            baseline=starts[0]
            for key in ('x','q','v','omega'):
                torch.testing.assert_close(getattr(start,key),getattr(baseline,key),rtol=0,atol=0)
            assert start.history.keys()==baseline.history.keys()
            for key in start.history:
                torch.testing.assert_close(start.history[key],baseline.history[key],rtol=0,atol=0)
        starts.append(start.clone())
        result,info=original(self,scene,start,dt,**kwargs)
        evaluations.append({'dt':dt,'rounds':len(info.get('iterations',())),
            'learned':kwargs.get('learned',True),'original_accepted':bool(torch.as_tensor(info['accepted']).all())})
        if len(evaluations)==1 or args.reject_all:
            info={**info,'accepted':torch.zeros_like(torch.as_tensor(info['accepted']),dtype=torch.bool)}
        return result,info
    MixedSolver.forward=injected
    saved_argv=sys.argv
    sys.argv=[str(ROOT/'scripts/run_franka_vbd_shared.py'),
        '--checkpoint',str(args.checkpoint),'--output',str(args.output/'recording'),
        '--frames','1','--request-dt',str(1/600),'--motion-budget','.012',
        '--material-motion-modes','--articulation-modes','--force-aware-anticipation',
        '--tangent-support-scale','8','--elastic-local-heads','2','--learned-contact-memory',
        '--device',args.device,'--gpu-memory-gib','12']
    try:
        try:runpy.run_path(str(ROOT/'scripts/run_franka_vbd_shared.py'),run_name='__main__')
        except SystemExit as error:
            if not args.reject_all or error.code!=1:raise
    finally:
        MixedSolver.forward=original
        MixedContactExampleSolver._write_native_output=original_write
        sys.argv=saved_argv
    run=json.loads((args.output/'recording/run.json').read_text())
    if args.reject_all:
        assert run['status']=='FAILED' and run['physical_seconds']==0 and run['physical_substeps']==0
        assert not native_writes
        attempts=run['last_diagnostic']['adaptive']['attempts']
        assert len(attempts)==len(evaluations)==6
        assert all(not row['accepted'] for row in attempts)
        assert sum(row['network_rounds'] for row in attempts)==sum(row['rounds'] for row in evaluations)
        with np.load(args.output/'recording/final_runtime_state.npz') as final, np.load(args.output/'recording/trajectory.npz') as tape:
            np.testing.assert_array_equal(final['particle_q'],tape['particle_position'][0])
            np.testing.assert_array_equal(final['body_q'],tape['body_pose'][0])
            np.testing.assert_array_equal(final['particle_qd'],tape['particle_velocity'][0])
            np.testing.assert_array_equal(final['body_qd'],tape['body_velocity'][0])
            for key,value in starts[0].history.items():
                np.testing.assert_array_equal(final['history__'+key],value.cpu().numpy())
        assert all(not value for value in run['forbidden_solver_calls'].values())
        report={'status':'PASS','device':args.device,'expected_exhausted_recovery':True,
            'native_state_velocity_history_and_clock_unchanged':True,'native_output_writes':0,
            'solver_attempts':len(attempts),'response_rounds':sum(row['rounds'] for row in evaluations),
            'native_dynamics_calls':0,'scope':'All-candidates-rejected fault injection; verifies bounded rollback, not a successful physical rollout.'}
        (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2));return
    assert run['status']=='COMPLETE' and run['adaptive_accepted_substeps']==1
    assert run['adaptive_retries']==1 and run['adaptive_recovery_attempts']==1
    assert run['adaptive_recovery_modes']=={'seed_off':1}
    assert len(evaluations)==run['adaptive_solver_attempts']==2
    assert run['adaptive_response_rounds']==sum(row['rounds'] for row in evaluations)
    assert run['adaptive_response_rounds']==run['nn_backbone_calls']
    assert all(row['learned'] and row['dt']>=1/600-1e-14 for row in evaluations)
    assert all(not value for value in run['forbidden_solver_calls'].values())
    assert len(native_writes)==1
    report={'status':'PASS','device':args.device,'evaluations':evaluations,
        'rejected_input_and_history_reused_exactly':True,'physical_seconds':run['physical_seconds'],
        'minimum_dt_s':run['adaptive_min_accepted_dt_s'],'retries':run['adaptive_retries'],
        'response_rounds':run['adaptive_response_rounds'],'native_dynamics_calls':0,
        'teacher_reads':run['teacher_trajectory_reads'],'optimizer_updates':0,
        'scope':'Fault-injection integration test on a real Franka/model/native-state path; not a grasp-quality or performance claim.'}
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
