"""Independent dense-metric tests of material labels and deployed readout."""
import argparse,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    import torch
    from wp_vbd.data import State
    from wp_vbd.shared_material_correction import SharedMaterialCorrection,teacher_loss,object_inputs
    from wp_vbd.material_correction_data import MaterialCorrectionRecorder
    from wp_vbd.analytic_material_response import correct
    torch.set_num_threads(1);torch.manual_seed(3101)
    n=13;dim=n*6;points=torch.randn(1,n,3,dtype=torch.float64)*.1
    mobile=torch.ones(n,dtype=torch.bool);mobile[-1]=False
    dof=torch.zeros(n,6,dtype=torch.bool);dof[:-1,:3]=True
    scene=SimpleNamespace(n=n,length=.3,kind=torch.zeros(n,dtype=torch.long),movable=mobile,dof=dof,
        object_id=torch.zeros(n,dtype=torch.long),mass=torch.rand(n,dtype=torch.float64)+1,
        rest_x=points[0].clone(),interaction_entities=[torch.arange(n)],
        metadata={'contract':'elastic_surface_joint_contact_v38'})
    matrix=torch.randn(dim,dim,dtype=torch.float64)*.05;matrix=matrix.T@matrix+torch.eye(dim,dtype=torch.float64)*3
    class Metric:
        def __init__(self,H):self.H=H;self.calls=[];self.terms=[]
        def __call__(self,value):self.calls.append(value.shape[-1]);return (self.H@value.reshape(dim,-1)).reshape_as(value)
    metric=Metric(matrix)
    raw=torch.zeros(dim,1,dtype=torch.float64);raw.reshape(n,6)[:-1,0]=1
    Q=raw/(raw.T@matrix@raw).sqrt()
    def coarse(value):return (Q@(Q.T@value.reshape(dim,-1))).reshape_as(value)
    state=State(points,torch.tensor([0.,0.,0.,1.],dtype=torch.float64).expand(1,n,4).clone(),torch.zeros_like(points),torch.zeros_like(points))
    source=torch.randn(1,n,6,dtype=torch.float64)*.01*dof;proposal=torch.randn_like(source)*.001*dof
    w=torch.eye(6,dtype=torch.float64).expand(1,n,6,6)
    baseline=torch.linalg.inv(matrix.reshape(n,6,n,6).permute(0,2,1,3)[torch.arange(n),torch.arange(n)])[None]
    latent=torch.randn(1,n,16,requires_grad=True)
    model=SimpleNamespace(protected_response=SimpleNamespace(last_latent=latent))
    with tempfile.TemporaryDirectory() as directory:
        recorder=MaterialCorrectionRecorder(directory,scene='dense_probe',stride=1)
        plain,plain_valid=correct(scene,state,metric,coarse,source,proposal)
        observed,observed_valid=correct(scene,state,metric,coarse,source,proposal,
            observer=recorder.observer(model,scene,state,metric,coarse,source,proposal,w,baseline,1/180))
        torch.testing.assert_close(plain,observed,rtol=0,atol=0)
        assert torch.equal(plain_valid,observed_valid)
        report=recorder.finish(complete=True);assert report['samples']==1
        row=recorder.rows[0];c=row['oracle'];gram=row['gram'];cross=row['cross'];norm=row['norm']
        oracle_error=(c@gram@c-2*c@cross+norm).clamp_min(0)
        assert float(oracle_error/(norm+1e-12))<1e-6
    head=SharedMaterialCorrection(16,hidden=32)
    zero,valid=head.correct(scene,state,metric,coarse,source,proposal,w,baseline,latent,1/180)
    torch.testing.assert_close(zero,proposal,rtol=0,atol=0);assert bool(valid.all())
    torch.nn.init.normal_(head.readout[-1].weight,std=.03)
    metric.calls.clear()
    prediction,valid=head.correct(scene,state,metric,coarse,source,proposal,w,baseline,latent,1/180)
    assert metric.calls==[1,1],metric.calls
    assert bool(valid.all()) and not bool(prediction[:,~mobile].any()) and not bool(prediction[...,3:].any())
    error=float((Q.T@matrix@(prediction-proposal).reshape(dim,1)).abs().max())
    assert error<1e-10
    prediction.square().sum().backward();assert latent.grad is not None and bool(torch.isfinite(latent.grad).all())
    assert float(head.readout[-1].weight.grad.norm())>0
    differentiable_state=State(points.clone().requires_grad_(True),state.q,state.v,state.omega)
    features,frozen_basis,scale=object_inputs(scene,differentiable_state,torch.arange(n-1),latent,
        source,proposal,w,baseline,1/180)
    assert not frozen_basis.requires_grad and features.requires_grad
    teacher_model=SimpleNamespace(training=True,config={})
    previous=scene._analytic_material_last_solves
    teacher_metric=Metric(matrix)
    loss=teacher_loss(teacher_model,scene,state,teacher_metric,coarse,source,proposal,plain)
    assert float(loss)==0. and teacher_model._material_training_teacher_solves==1
    assert scene._analytic_material_last_solves==previous
    head.zero_grad(set_to_none=True)
    predicted,_=head.correct(scene,state,metric,coarse,source,proposal,w,baseline,latent,1/180)
    teacher_loss(teacher_model,scene,state,metric,coarse,source,proposal,predicted).backward()
    assert float(head.readout[-1].weight.grad.norm())>0
    teacher_model.training=False
    try:teacher_loss(teacher_model,scene,state,metric,coarse,source,proposal,predicted)
    except RuntimeError:pass
    else:raise AssertionError('Eval must never call the training teacher')
    # Real spatial transform of the independent dense system.
    from wp_vbd import math3d as so3
    rotation=so3.matrix(so3.exp(torch.tensor([[.3,-.2,.1]],dtype=torch.float64)))[0]
    transform=torch.block_diag(*([rotation,rotation]*n));rotated_metric=Metric(transform@matrix@transform.T)
    rotated_Q=transform@Q
    def rotated_coarse(value):return (rotated_Q@(rotated_Q.T@value.reshape(dim,-1))).reshape_as(value)
    rotated_scene=SimpleNamespace(**vars(scene));rotated_scene.rest_x=scene.rest_x@rotation.T+.7
    rotated_state=State(points@rotation.T+.7,state.q,state.v,state.omega)
    vector=lambda value:(transform@value.reshape(dim,1)).reshape_as(value)
    block=torch.block_diag(rotation,rotation)
    rotated_baseline=block@baseline@block.T
    rotated,_=head.correct(rotated_scene,rotated_state,rotated_metric,rotated_coarse,vector(source),vector(proposal),w,rotated_baseline,latent.detach(),1/180)
    covariance=float((rotated-vector(prediction.detach())).abs().max())
    assert covariance<1e-7,covariance
    result={'status':'PASS','observer_preserves_teacher_exactly':True,'independent_mass_metric_oracle_relative_error':float(oracle_error/(norm+1e-12)),
        'zero_initialization_preserves_proposal':True,'runtime_metric_action_columns':[1,1],
        'no_runtime_material_gram_or_reduced_solve':True,'fixed_and_rotational_particle_dofs_preserved':True,
        'on_policy_teacher_exact_at_target':True,'on_policy_teacher_gradient_nonzero':True,
        'teacher_eval_forbidden':True,'teacher_counter_isolated_from_inference':True,
        'material_geometric_basis_frozen_but_shared_features_differentiable':True,
        'coarse_orthogonality_error':error,'shared_latent_gradient_nonzero':float(latent.grad.norm())>0,
        'world_rotation_translation_covariance_error':covariance,'native_solver_calls':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
