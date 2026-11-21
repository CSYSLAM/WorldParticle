"""Dense energy, projector, rank and training checks for fused response kernels."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cpu',choices=('cpu','cuda:0'))
    p.add_argument('--baseline',type=Path,required=True,help='Pre-change source snapshot root')
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--spatial-products',action='store_true')
    a=p.parse_args()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    from wp_vbd.compiled_contact import small_spatial_products
    with small_spatial_products(a.spatial_products):verify(a)


def verify(a):
    import torch
    from wp_vbd.mixed_metric import MetricTerm
    from wp_vbd.rigid_contact_system import ContactMetric
    from wp_vbd.protected_response import ProtectedCoarse
    from wp_vbd.heterogeneous_backbone import build_plan
    torch.set_num_threads(1);torch.manual_seed(7121)
    torch.use_deterministic_algorithms(True)

    def old_module(filename):
        name='wp_vbd._device_baseline_'+Path(filename).stem
        spec=importlib.util.spec_from_file_location(name,a.baseline/'src/wp_vbd'/filename)
        module=importlib.util.module_from_spec(spec);sys.modules[name]=module
        spec.loader.exec_module(module)
        return module
    old_metric=old_module('mixed_metric.py').FrozenMixedMetric
    old_coarse=old_module('protected_response.py').ProtectedCoarse
    old_plan=old_module('heterogeneous_backbone.py').build_plan
    device=torch.device(a.device);dtype=torch.float64;n=14;batch=2
    random=lambda *shape:torch.randn(*shape,device=device,dtype=dtype)
    matrix=random(batch,n,6,6)
    local=matrix.transpose(-1,-2)@matrix+torch.eye(6,device=device,dtype=dtype)
    dof=torch.ones(n,6,device=device,dtype=torch.bool);dof[:7,3:]=False;dof[-1]=False
    ids=torch.tensor([[0,1,1],[3,5,8],[10,13,0]],device=device)
    jacobian=random(batch,3,4,3,6);weight=torch.rand(batch,3,4,device=device,dtype=dtype)+.2
    terms=[MetricTerm(ids,jacobian,weight)]
    reference=old_metric.__new__(old_metric)
    reference.local=local;reference.terms=terms;reference.dof=dof
    candidate=ContactMetric(local,terms,dof,compiled_response_kernels=True)
    scene=SimpleNamespace(device=device,n=n,dof=dof,interaction_entities=[torch.arange(n,device=device)],
        metadata={},compiled_response_kernels=True)
    # Build H independently by adding every owner role before J^T W J.
    dense=torch.block_diag(*local[0]).expand(batch,-1,-1).clone()
    for b in range(batch):
        dense[b]=torch.block_diag(*local[b])
        for row in range(len(ids)):
            j=random(0).new_zeros(4,n,6)
            for slot in range(ids.shape[1]):j[:,ids[row,slot]]+=jacobian[b,row,:,slot]
            j=j.flatten(1)
            dense[b]+=j.T@(weight[b,row,:,None]*j)
    mask=dof.flatten().to(dtype)
    dense=dense*mask[None,:,None]*mask[None,None,:]
    checks={};max_errors={}
    with torch.no_grad():
        torch.use_deterministic_algorithms(False)
        for columns in (1,6,24):
            value=random(batch,n,6,columns)
            expected=(dense@value.flatten(1,2)).reshape_as(value)
            original=reference(value);actual=candidate(value)
            torch.testing.assert_close(original,expected,rtol=1e-10,atol=1e-10)
            torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
            max_errors[str(columns)]=float((actual-expected).abs().max())
        torch.testing.assert_close(candidate.diagonal,reference.diagonal,rtol=1e-12,atol=1e-12)
        checks['dense_metric_including_repeated_owner_cross_terms']=True
        checks['original_slot_diagonal_convention_preserved']=True
        modes=random(batch,n,6,12);modes[...,9:11]=modes[...,:2];modes[...,11]=0.
        blocks=[(SimpleNamespace(ids=torch.arange(n,device=device)),modes)]
        original=old_coarse(scene,reference,blocks)
        compiled=ProtectedCoarse(scene,candidate,blocks)
        assert torch.equal(original.valid,compiled.valid) and bool(compiled.valid.all())
        assert all(torch.equal(x,y) for x,y in zip(original.ranks,compiled.ranks))
        q=compiled.groups[0][1];gram=torch.einsum('bnip,bniq->bpq',q,candidate(q))
        active=q.square().sum((1,2))>0
        torch.testing.assert_close(gram,torch.diag_embed(active.to(dtype)),rtol=1e-8,atol=1e-9)
        value=random(batch,n,6)
        torch.testing.assert_close(compiled(value),original(value),rtol=1e-9,atol=1e-10)
        torch.testing.assert_close(compiled(candidate(compiled(value)[...,None])[...,0]),compiled(value),rtol=1e-8,atol=1e-9)
        checks['redundant_and_zero_columns_keep_original_rank']=True
        checks['energy_orthonormal_basis_and_projector']=True
        checks['fixed_dofs_have_zero_response']=not bool(compiled(value)[:,~dof].any())
        # Negative weights must remain rejected even if local inertia is SPD.
        bad=ContactMetric(local,[MetricTerm(ids,jacobian,-weight)],dof,compiled_response_kernels=True)
        rejected=ProtectedCoarse(scene,bad,blocks)
        assert not bool(rejected.valid.any())
        checks['invalid_metric_not_repaired_into_acceptance']=True
        empty=ContactMetric(local,[],dof,compiled_response_kernels=True)
        torch.testing.assert_close(empty(value[...,None])[...,0],(local@(value*dof)[...,None])[...,0]*dof)
        checks['empty_sparse_terms_supported']=True
    # New flags must not compile away the original gradient-enabled path.
    x=random(batch,n,6,1).requires_grad_()
    energy=(x*candidate(x)).sum()*.5;energy.backward()
    expected=(dense@x.detach().flatten(1,2)).reshape_as(x)
    torch.testing.assert_close(x.grad,expected,rtol=1e-10,atol=1e-10)
    y=random(batch,n,6).requires_grad_();compiled(y).sum().backward()
    torch.testing.assert_close(y.grad,original(torch.ones_like(y)),rtol=1e-9,atol=1e-10)
    checks['gradient_enabled_paths_preserved']=True
    # Metadata changes get a fresh exact partition, including noncontiguous
    # component labels, object IDs, all four types and fixed entities.
    graph=SimpleNamespace(interaction_id=torch.arange(n,device=device)%3*7,
        movable=dof.any(-1),object_id=torch.arange(n,device=device)%5-2,
        kind=torch.arange(n,device=device)%4,topology_generation=0)
    for generation in range(3):
        graph.topology_generation=generation
        left=old_plan(graph,3);right=build_plan(graph,3)
        assert (left.generation,left.maximum_super_tokens,left.buckets)==(right.generation,right.maximum_super_tokens,right.buckets)
        assert len(left.components)==len(right.components)
        for u,v in zip(left.components,right.components):
            assert torch.equal(u.ids,v.ids) and u.merge_rounds==v.merge_rounds and u.coarse_count==v.coarse_count
            assert len(u.groups)==len(v.groups) and all(torch.equal(i,j) for i,j in zip(u.groups,v.groups))
        assert all(torch.equal(u,v) for u,v in zip(left.type_ids,right.type_ids))
        graph.kind=(graph.kind+1)%4;graph.object_id=graph.object_id.flip(0)
    checks['batched_metadata_exact_same_hierarchy_plan']=True
    result={'status':'PASS' if all(checks.values()) else 'FAIL','device':a.device,'small_spatial_products':a.spatial_products,'checks':checks,
        'dense_metric_max_abs_errors':max_errors,'coarse_rank':compiled.ranks[0].tolist(),
        'native_dynamics_calls':0,'optimizer_updates':0}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    assert result['status']=='PASS'


if __name__=='__main__':main()
