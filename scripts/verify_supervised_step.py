"""Check differentiable, atomic repairs without accessing future labels."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']=''


def main():
    import torch
    from wp_vbd.data import State
    from wp_vbd.supervised_step import advance_reference_span
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1)
    x=torch.zeros(1,1,3,dtype=torch.float64)
    initial=State(x,torch.tensor([[[0.,0.,0.,1.]]],dtype=torch.float64),x.clone(),x.clone(),{'elapsed':torch.zeros(1,dtype=torch.float64)})
    force=torch.arange(1.,4.,dtype=torch.float64).reshape(3,1,1).expand(3,1,3).clone().requires_grad_(True)
    states={'x':torch.full((4,1,3),float('nan')),'history__elapsed':torch.full((4,),float('nan')),'input_force':force}
    def stub(current,actions,dt):
        assert set(actions)=={'input_force'}
        accepted=dt<=.0035
        if not accepted:
            current.x.add_(100);current.history['elapsed'].add_(50)
        out=State(current.x+dt*actions['input_force'],current.q,current.v,current.omega,
                  {'elapsed':current.history['elapsed']+dt})
        return out,{'accepted':torch.tensor([accepted]),'iterations':[{'phase':'analytic_predictor'},{'phase':'neural_correction'}]}
    result,info,report=advance_reference_span(initial,states,0,3,1/600,stub)
    torch.testing.assert_close(result.x,force.sum(0,keepdim=True)/600)
    torch.testing.assert_close(result.history['elapsed'],torch.tensor([.005],dtype=torch.float64))
    assert report['neural_rounds']==3 and report['accepted_substeps']==2 and report['rejected_attempts']==1
    assert not bool(initial.x.any()) and not bool(initial.history['elapsed'].any())
    result.x.sum().backward();torch.testing.assert_close(force.grad,torch.ones_like(force)/600)
    def reject(current,actions,dt):
        current.x.add_(70);current.history['elapsed'].add_(80)
        return current,{'accepted':torch.tensor([False]),'iterations':[{'phase':'neural_correction'}]}
    try:advance_reference_span(initial,states,0,3,1/600,reject)
    except RuntimeError as error:assert 'floor' in str(error)
    else:raise AssertionError('A rejected floor step committed')
    assert not bool(initial.x.any()) and not bool(initial.history['elapsed'].any())
    summary={'status':'PASS','no_teacher_state_or_history_reads':True,'accepted_history_continuity':True,
        'original_state_unchanged_after_failed_mutation':True,'repaired_end_time_exact':True,
        'external_impulse_integral_exact':True,'finite_correct_gradient':True,
        'hard_floor_1_over_600s':True,'all_attempted_neural_calls_counted':True}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary))


if __name__=='__main__':main()
