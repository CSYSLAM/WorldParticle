"""Index-map parity, mutation invalidation and bounded ownership of topology caches."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    import torch
    from wp_vbd.protected_response import topology
    from wp_vbd.response_topology import IndexWorkspace,workspace,mobile_components
    path=a.baseline/'src/wp_vbd/protected_response.py'
    spec=importlib.util.spec_from_file_location('wp_vbd._topology_reference',path)
    reference=importlib.util.module_from_spec(spec);sys.modules[spec.name]=reference;spec.loader.exec_module(reference)
    device=torch.device(a.device);n=11;make=lambda value:torch.tensor(value,device=device,dtype=torch.long)
    groups=[make([0,1,2,3]),make([4,5,6]),make([7,8,9]),make([10])]
    scene=SimpleNamespace(n=n,device=device,interaction_entities=groups,compiled_response_kernels=True,
                          movable=make([1]*10+[0]).bool())
    ids=[make([[0,1,1],[2,3,-1],[5,6,7],[10,10,-1]]),make([[4,5],[8,9]])]
    terms=[SimpleNamespace(ids=value) for value in ids]
    metric=SimpleNamespace(terms=terms)
    blocks=[(SimpleNamespace(ids=make([0,2,3])),torch.zeros(1,3,6,6,device=device)),
            (SimpleNamespace(ids=make([4,5,6])),torch.zeros(1,3,6,3,device=device)),
            (SimpleNamespace(ids=make([7,8,9])),torch.zeros(1,3,6,9,device=device))]
    def same(left,right):
        assert len(left)==len(right)
        for u,v in zip(left,right):
            assert torch.equal(u.ids,v.ids) and u.diagonal_exact==v.diagonal_exact
            assert len(u.blocks)==len(v.blocks) and len(u.terms)==len(v.terms)
            for (i,x),(j,y) in zip(u.blocks,v.blocks):assert i==j and torch.equal(x,y)
            for urow,vrow in zip(u.terms,v.terms):
                assert urow[0]==vrow[0]
                assert all(torch.equal(x,y) for x,y in zip(urow[1:],vrow[1:]))
    with torch.no_grad():
        for mutation in range(5):
            scene._protected_topology=None
            expected=reference.topology(scene,metric,blocks);actual=topology(scene,metric,blocks)
            same(expected,actual);assert topology(scene,metric,blocks) is actual
            if mutation==0:ids[0][0,1]=2 # same shapes, different owner and repetition
            elif mutation==1:
                # Same row count and values in a different order.
                ids[1].copy_(ids[1].flip(0))
            elif mutation==2:
                groups[0].copy_(groups[0].flip(0))
            elif mutation==3:
                blocks[0][0].ids.copy_(make([1,2,3]))
        # Basis values/column counts and physical coefficients are not index
        # dependencies. Reusing a map must not keep any numerical response.
        old=topology(scene,metric,blocks)
        blocks[0]=(blocks[0][0],torch.randn(1,3,6,12,device=device))
        assert topology(scene,metric,blocks) is old
        for term in terms:term.jacobian=torch.randn(1,device=device);term.weight=torch.randn(1,device=device)
        assert topology(scene,metric,blocks) is old
        first=mobile_components(scene);scene.movable[0]=False
        second=mobile_components(scene);assert not torch.equal(first[0],second[0])
        assert all(torch.equal(m,group[scene.movable[group]]) for m,group in zip(second,groups))
        # Eviction and whole backing-storage accounting, including a one-item view.
        bounded=IndexWorkspace(maximum_entries=3,maximum_bytes=16*1024)
        for i in range(20):
            source=make([i]*4);result=bounded.get('copy',(source,),lambda:source.clone())
            assert torch.equal(result,source) and bounded.bytes<=bounded.maximum_bytes and len(bounded.entries)<=3
        assert bounded.evictions>0
        huge=IndexWorkspace(maximum_entries=3,maximum_bytes=1024);large=torch.zeros(4096,device=device)
        view=large[:1];calls=[0]
        def compute():calls[0]+=1;return view.clone()
        huge.get('view',(view,),compute);huge.get('view',(view,),compute)
        assert calls[0]==2 and not huge.entries
        fresh=IndexWorkspace();calls[0]=0
        with torch.inference_mode():unversioned=make([1,2])
        def unknown():calls[0]+=1;return unversioned.clone()
        fresh.get('unknown',(unversioned,),unknown);fresh.get('unknown',(unversioned,),unknown)
        assert calls[0]==2 and fresh.unversioned==2 and not fresh.entries
    # Autograd-enabled calls retain the original planner and do not populate
    # an inference workspace on an independent training scene.
    training=SimpleNamespace(n=n,device=device,interaction_entities=groups,compiled_response_kernels=True)
    reference.topology(training,metric,blocks);topology(training,metric,blocks)
    assert not hasattr(training,'_response_index_workspace')
    result={'status':'PASS','device':a.device,'same_ordered_maps_and_repeated_owner_flags':True,
            'same_shape_mutations_and_reordering_invalidate':True,'basis_and_metric_values_not_cached':True,
            'mobility_mutations_invalidate':True,'bounded_eviction':bounded.report(),
            'large_view_backing_storage_and_unversioned_inputs_bypass':True,
            'gradient_enabled_training_planner_unchanged':True,'workspace':workspace(scene).report()}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
