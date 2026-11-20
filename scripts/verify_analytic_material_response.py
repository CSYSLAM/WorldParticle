"""Independent energy, rank, coupling and gradient checks of material correction."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    import torch
    from wp_vbd.analytic_material_response import reduced_correction,correct
    torch.set_num_threads(1);torch.manual_seed(20261120)
    n=21;D=n*6;rank=11
    A=torch.randn(D,D,dtype=torch.float64);H=A.T@A+torch.eye(D,dtype=torch.float64)*2
    columns=torch.randn(1,n,6,rank,dtype=torch.float64)
    columns=torch.cat((columns,columns[...,:2]),-1)
    image=(H@columns.reshape(D,-1)).reshape_as(columns)
    r=torch.randn(1,n,6,dtype=torch.float64)
    delta,valid=reduced_correction(columns,image,r)
    C=columns.reshape(D,-1);rhs=C.T@r.flatten()
    expected=-C@torch.linalg.pinv(C.T@H@C,rtol=1e-10,hermitian=True)@rhs
    torch.testing.assert_close(delta.flatten(),expected,rtol=1e-9,atol=1e-10)
    projected=(C.T@(r.flatten()+H@delta.flatten())).norm()
    assert bool(valid.all()) and float(projected)<1e-9
    # Real entry: two interacting materials, static point, and analytic coarse.
    scene=SimpleNamespace(n=n,kind=torch.zeros(n,dtype=torch.long),movable=torch.ones(n,dtype=torch.bool),
        object_id=torch.cat((torch.zeros(10,dtype=torch.long),torch.ones(11,dtype=torch.long))),
        mass=torch.rand(n,dtype=torch.float64)+1,interaction_entities=[torch.arange(n)],
        dof=torch.zeros(n,6,dtype=torch.bool))
    scene.dof[:,:3]=True;scene.movable[-1]=False;scene.dof[-1]=False
    points=torch.randn(1,n,3,dtype=torch.float64)
    class Metric:
        def __call__(self,value):return (H@value.reshape(D,-1)).reshape_as(value)
    metric=Metric()
    # Include a physical coherent translation in the already represented space.
    B=torch.zeros(D,1,dtype=torch.float64);B.reshape(n,6)[:-1,0]=1
    Q=B/torch.sqrt(B.T@H@B)
    def coarse(value):
        shape=value.shape
        return (Q@(Q.T@value.reshape(D,-1))).reshape(shape)
    raw=torch.randn(1,n,6,dtype=torch.float64,requires_grad=True)
    proposed=raw*scene.dof
    fixed,good=correct(scene,SimpleNamespace(x=points),metric,coarse,r,proposed)
    def energy(value):
        u=value.flatten();return r.flatten()@u+.5*u@H@u
    assert bool(good.all()) and float(energy(fixed).detach())<=float(energy(proposed).detach())+1e-8
    assert not bool(fixed[:,~scene.movable].any()) and not bool(fixed[...,3:].any())
    # Material correction remains H-orthogonal to the original coarse space.
    assert float((Q.T@H@(fixed-proposed).flatten()).detach().abs().max())<1e-8
    fixed.square().sum().backward()
    assert raw.grad is not None and bool(torch.isfinite(raw.grad).all()) and float(raw.grad.norm())>0
    from wp_vbd import math3d as so3
    rotation=so3.matrix(so3.exp(torch.tensor([[.21,-.37,.13]],dtype=torch.float64)))[0]
    transform=torch.block_diag(*([rotation,rotation]*n))
    rotated_metric_matrix=transform@H@transform.T
    class RotatedMetric:
        def __call__(self,value):return (rotated_metric_matrix@value.reshape(D,-1)).reshape_as(value)
    rotated_Q=transform@Q
    def rotated_coarse(value):
        return (rotated_Q@(rotated_Q.T@value.reshape(D,-1))).reshape_as(value)
    rotated_r=(transform@r.flatten()).reshape_as(r)
    rotated_proposal=(transform@proposed.detach().flatten()).reshape_as(proposed)
    rotated,rotated_good=correct(scene,SimpleNamespace(x=points@rotation.T+1.),
        RotatedMetric(),rotated_coarse,rotated_r,rotated_proposal)
    expected_rotated=(transform@fixed.detach().flatten()).reshape_as(fixed)
    torch.testing.assert_close(rotated,expected_rotated,rtol=1e-7,atol=1e-8)
    assert bool(rotated_good.all())
    # The work cap applies per physical interaction, not across unrelated
    # objects that merely happen to share one simulation world.
    count=45
    independent=SimpleNamespace(n=count,kind=torch.zeros(count,dtype=torch.long),
        movable=torch.ones(count,dtype=torch.bool),object_id=torch.arange(count)//9,
        mass=torch.ones(count,dtype=torch.float64),interaction_entities=list(torch.arange(count).split(9)),
        dof=torch.tensor([True,True,True,False,False,False]).expand(count,-1))
    class IdentityMetric:
        def __call__(self,value):return value
    zero=torch.zeros(1,count,6,dtype=torch.float64)
    force=torch.randn_like(zero)*independent.dof
    response,independent_good=correct(independent,SimpleNamespace(x=torch.randn(1,count,3,dtype=torch.float64)),
        IdentityMetric(),torch.zeros_like,force,zero,maximum_columns=24)
    assert bool(independent_good.all())
    assert float((force*response+.5*response.square()).sum())<0
    # More objects than a block admits still share the physical quadratic.
    # Check bounded sequential blocks against a dense coupled SPD metric.
    independent.interaction_entities=[torch.arange(count)]
    matrix=torch.randn(count*6,count*6,dtype=torch.float64)*.025
    matrix=matrix.T@matrix+torch.eye(count*6,dtype=torch.float64)
    class CoupledMetric:
        def __call__(self,value):return (matrix@value.reshape(count*6,-1)).reshape_as(value)
    raw=torch.randn_like(zero,requires_grad=True);proposal=raw*independent.dof
    response,blocked_good=correct(independent,SimpleNamespace(x=torch.randn(1,count,3,dtype=torch.float64)),
        CoupledMetric(),torch.zeros_like,force,proposal,maximum_columns=24,blocked=True)
    def coupled_energy(value):
        v=value.flatten();return force.flatten()@v+.5*v@matrix@v
    assert bool(blocked_good.all()) and float((coupled_energy(response)-coupled_energy(proposal)).detach())<=1e-8
    response.square().sum().backward()
    assert raw.grad is not None and bool(torch.isfinite(raw.grad).all())
    summary={'status':'PASS','rank_deficient_solve_matches_independent_pseudoinverse':True,
        'projected_residual_norm':float(projected),'coupled_quadratic_energy_does_not_increase':True,
        'fixed_and_rotational_dofs_unchanged':True,'coarse_energy_orthogonality_preserved':True,
        'world_rotation_and_translation_covariance_error':float((rotated-expected_rotated).abs().max()),
        'independent_objects_do_not_share_the_material_work_budget':True,
        'bounded_blocks_keep_all_coupled_objects_and_nonincreasing_energy':True,
        'shared_proposal_gradient_norm':float(raw.grad.norm()),'native_dynamics_calls':0,'optimizer_updates':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(summary,indent=2)+'\n');print(summary)


if __name__=='__main__':main()
