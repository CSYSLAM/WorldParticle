"""Exercise rejected-step recovery, bounded work, and transactional history on CPU.

Synthetic proposals deliberately fail; this does not certify grasp quality.
The production runtime recovery method is used without native dynamics.
"""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    from wp_vbd.adaptive_step import AdaptiveStepper,AdaptiveStepConfig,AdaptiveStepFailure
    from wp_vbd.data import State
    from wp_vbd.mixed_contact_runtime import MixedContactExampleSolver

    torch.set_num_threads(1)
    floor=1/600
    state=State(torch.zeros(1,2,3,dtype=torch.float64),
                torch.tensor([0.,0.,0.,1.],dtype=torch.float64).expand(1,2,4).clone(),
                torch.zeros(1,2,3,dtype=torch.float64),torch.zeros(1,2,3,dtype=torch.float64),
                {'elapsed':torch.zeros(1,dtype=torch.float64)})
    config={'unroll':3,'learned_contact_tangent_seed':True}
    runtime=SimpleNamespace(network=SimpleNamespace(config=config,elastic_shape_head=SimpleNamespace(local_heads=2)),
                            scene=SimpleNamespace(length=1.),learned=True)
    stepper=runtime.adaptive_stepper=AdaptiveStepper(AdaptiveStepConfig())
    attempted=[]

    def solve(current,h,time,learned):
        # The production recovery ladder must supply a fresh clone each time.
        torch.testing.assert_close(current.history['elapsed'],torch.tensor([time],dtype=torch.float64))
        current.history['elapsed'].add_(h)
        current.x.add_(h)
        rounds=config['unroll']
        attempted.append({'h':h,'time':time,'rounds':rounds,'learned':learned})
        accepted=not config['learned_contact_tangent_seed']
        return current,{'accepted':accepted,'iterations':[{}]*rounds}

    def attempt(current,h,time):
        return solve(current,h,time,True)

    def repair(current,h,time,info,allowance):
        return MixedContactExampleSolver._repair_adaptive_proposal(
            runtime,current,h,time,info,allowance,solve)

    # Ordinary proposals shrink to the floor, then a learned seed-off solve
    # succeeds without calling the analytic fallback or carrying rejected history.
    result,report=stepper.advance(state,1/60,attempt,repair_step=repair)
    torch.testing.assert_close(result.x,torch.full_like(result.x,1/60))
    torch.testing.assert_close(result.history['elapsed'],torch.tensor([1/60],dtype=torch.float64))
    assert config=={'unroll':3,'learned_contact_tangent_seed':True}
    assert runtime.network.elastic_shape_head.local_heads==2
    assert report['retries']>0
    assert report['network_rounds']==sum(row['rounds'] for row in attempted)
    assert len(report['attempts'])==len(attempted)
    assert min(row['h'] for row in attempted)>=floor-1e-14
    assert max(row['h'] for row in attempted)<=1/30+1e-14
    assert all(row['learned'] for row in attempted)
    assert sum(row['mode']=='seed_off' and row['accepted'] for row in report['attempts'])>0

    # An end/event-aligned remainder such as 1.5*floor cannot be halved;
    # repair that representable step instead of looping or using a sub-floor h.
    attempted.clear()
    aligned,aligned_report=stepper.advance(state,1.5*floor,attempt,repair_step=repair)
    torch.testing.assert_close(aligned.x,torch.full_like(aligned.x,1.5*floor))
    assert aligned_report['substeps']==1 and aligned_report['retries']==1

    # A solver-accepted proposal can still violate the independent velocity
    # or contact-depth admission rule; these rejections must enter repair too.
    for reason in ('relative_velocity_change','contact_depth','solver_rejected_or_nonfinite'):
        local_stepper=AdaptiveStepper(AdaptiveStepConfig(maximum_contact_depth=.001))
        def bad(current,h,time):
            if reason=='relative_velocity_change':current.v.fill_(1e4)
            elif reason=='solver_rejected_or_nonfinite':current.history['elapsed'].fill_(float('nan'))
            return current,{'accepted':True,'iterations':[{}]*3,
                'minimum_contact_normal_physical_separation':-.01 if reason=='contact_depth' else 0.}
        def fixed(current,h,time,info,allowance):
            current.x.add_(h)
            return current,{'accepted':True,'iterations':[{}]*5,'adaptive_recovery':{
                'attempts':[{'mode':'test_repair','accepted':True,'reason':'accepted',
                             'network_rounds':5,'relative_velocity_change':0.}]}}
        _,admission=local_stepper.advance(state,floor,bad,repair_step=fixed)
        assert admission['attempts'][0]['reason']==reason
        assert admission['substeps']==1 and admission['retries']==1 and admission['network_rounds']==8

    # Exhausted recovery stays bounded and leaves the caller's entire state
    # unchanged, even if proposals corrupt their private histories in place.
    def reject(current,h,time,learned):
        current.history['elapsed'].add_(100)
        return current,{'accepted':False,'iterations':[{}]*config['unroll']}
    def failed_repair(current,h,time,info,allowance):
        return MixedContactExampleSolver._repair_adaptive_proposal(
            runtime,current,h,time,info,allowance,reject)
    bounded=AdaptiveStepper(AdaptiveStepConfig(maximum_attempts=3))
    def failed(current,h,time):return reject(current,h,time,True)
    try:bounded.advance(state,floor,failed,repair_step=failed_repair)
    except AdaptiveStepFailure as error:
        assert len(error.report['attempts'])==3
        assert error.report['network_rounds']==3+3+3
    else:raise AssertionError('A wholly rejected interval was committed')
    assert not bool(state.x.any()) and not bool(state.history['elapsed'].any())
    assert config=={'unroll':3,'learned_contact_tangent_seed':True}
    assert runtime.network.elastic_shape_head.local_heads==2

    # A candidate exception must restore optional config keys and head count.
    def exception(*unused):raise RuntimeError('synthetic recovery exception')
    try:MixedContactExampleSolver._repair_adaptive_proposal(runtime,state,floor,0.,{'accepted':False},5,exception)
    except RuntimeError as error:assert str(error)=='synthetic recovery exception'
    else:raise AssertionError('Exception was swallowed')
    assert config=={'unroll':3,'learned_contact_tangent_seed':True}
    assert runtime.network.elastic_shape_head.local_heads==2

    # Reject impossible time partitions before any solver work.
    for duration,events in ((floor/2,()),(1/60,(floor/2,))):
        try:stepper.advance(state,duration,attempt,events=events)
        except ValueError:pass
        else:raise AssertionError('Sub-floor requested/event interval was accepted')
    narrow=AdaptiveStepper(AdaptiveStepConfig(minimum_dt=floor,maximum_dt=1.1*floor))
    try:narrow.advance(state,1.5*floor,attempt)
    except ValueError:pass
    else:raise AssertionError('Impossible min/max partition was accepted')

    summary={'status':'PASS','minimum_dt_s':floor,'learned_repair_completes_requested_interval':True,
        'all_recovery_rounds_and_retries_counted':True,'event_aligned_floor_recovery':True,
        'independent_admission_rejections_repaired':True,'private_history_per_recovery_candidate':True,
        'optional_configuration_restored_on_success_failure_exception':True,
        'work_budget_enforced':True,'impossible_time_partitions_rejected_before_solving':True,
        'native_dynamics_calls':0,'optimizer_updates':0,
        'scope':'Synthetic failure regression using the production stepper and runtime recovery ladder; not a physical grasp validation.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
