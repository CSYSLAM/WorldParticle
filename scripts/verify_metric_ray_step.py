"""Check proposal curvature scales against an independent small dense form."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    import torch
    from wp_vbd.metric_ray_step import scale_proposal
    from wp_vbd.mixed_metric import MetricTerm
    from wp_vbd.rigid_contact_system import ContactMetric
    from wp_vbd.gpu_budget import GpuBudget
    budget=GpuBudget(args.device);torch.set_num_threads(1)
    cases=[]
    for dtype in (torch.float32,torch.float64):
        dof=torch.ones(4,6,device=args.device,dtype=torch.bool);dof[[0,2],3:]=False;dof[3]=False
        scene=SimpleNamespace(n=4,interaction_entities=[torch.tensor(ids,device=args.device) for ids in ([0,1],[2],[3])])
        local=torch.diag_embed(dof.to(dtype))[None]
        ids=torch.tensor([[0,1]],device=args.device)
        jac=torch.tensor([[[[[-1.,0,0,0,0,0],[1.,0,0,.25,0,0]]]]],device=args.device,dtype=dtype)
        metric=ContactMetric(local,[MetricTerm(ids,jac,torch.ones(1,1,1,device=args.device,dtype=dtype)*2)],dof)
        dense=torch.diag(dof.flatten().to(dtype))
        row=torch.zeros(24,device=args.device,dtype=dtype);row[:12]=jac.flatten()
        dense=dense+2*row[:,None]*row[None]
        raw=(torch.arange(24,device=args.device,dtype=dtype).reshape(1,4,6)/8)*dof
        image=(dense@raw.flatten()).reshape_as(raw)
        residual=-image;residual[:,:2]*=.25;residual[:,2]*=4
        residual.requires_grad_();update=raw.clone().requires_grad_()
        actual,valid,detail=scale_proposal(scene,metric,residual,update)
        expected_scales=torch.tensor([.25,.25,1,1],device=args.device,dtype=dtype)[None,:,None]
        assert bool(valid.all()) and torch.equal(actual,raw*expected_scales)
        assert not bool(actual[:,~dof].any())
        actual.sum().backward()
        assert torch.equal(update.grad,expected_scales.expand_as(update)) and residual.grad is None
        changed=residual.detach().clone();changed[:,2]*=32
        other,_,_=scale_proposal(scene,metric,changed,raw)
        assert torch.equal(other[:,:2],actual[:,:2])
        zero=torch.zeros_like(raw)
        assert torch.equal(scale_proposal(scene,metric,zero,zero)[0],zero)
        assert torch.equal(scale_proposal(scene,metric,zero,raw)[0],zero)
        bad,good,_=scale_proposal(scene,metric,-residual.detach(),raw)
        assert not bool(good.all()) and torch.equal(bad,zero)
        class NegativeMetric:
            terms=[]
            def __call__(self,value):return -value
        assert not bool(scale_proposal(scene,NegativeMetric(),residual.detach(),raw)[1].all())
        nonfinite=residual.detach().clone();nonfinite[0,0,0]=float('nan')
        assert not bool(scale_proposal(scene,metric,nonfinite,raw)[1].all())
        try:scale_proposal(scene,metric,raw[:,:-1],raw)
        except ValueError:pass
        else:raise AssertionError('Mismatched physical shapes accepted')
        # At the uncapped scale, this quadratic's ray minimizer is exactly .25.
        sub=torch.zeros_like(raw);sub[:,:2]=raw[:,:2]
        def first_energy(scale):
            step=sub*scale
            return (residual.detach()*step).sum()+(step.flatten()@(dense@step.flatten()))/2
        assert first_energy(.25)<first_energy(.125) and first_energy(.25)<first_energy(.5)
        cases.append({'dtype':str(dtype),'dense_form_and_mask_exact':True,'independent_components_exact':True,
                      'frozen_coefficient_backward_exact':True,'zero_residual_fixed_point_exact':True,
                      'non_descent_negative_curvature_nonfinite_and_bad_shape_rejected':True,'scales':detail})
    budget.observe(force=True)
    result={'status':'METRIC_RAY_DENSE_QUADRATIC_AND_FROZEN_BACKWARD_PASS','device':args.device,'cases':cases,
            'new_optimizer_updates':0,'new_training_admissions':0,'full_vbd_prediction_calls':0,'gpu_resource_budget':budget.report(),
            'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/wp_vbd/metric_ray_step.py',Path(__file__))},
            'scope':'Small independent physical quadratic, six-slot masks, component isolation, frozen decision gradients and invalid forms. This does not certify the nonlinear metric model, original demos or selected training.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
