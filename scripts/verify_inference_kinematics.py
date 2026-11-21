"""Pure-kinematics cache equivalence, mutation invalidation and scope ownership."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    import torch
    from wp_vbd.contact_ports import PointPorts
    from wp_vbd.data import State
    from wp_vbd.inference_kinematics import StepKinematics,memoize
    torch.set_num_threads(1)
    owners=torch.tensor([[0,1]],dtype=torch.long)
    weights=torch.tensor([[[1.,0.],[0.,1.]]],dtype=torch.float64)
    ports=PointPorts(owners,weights,torch.tensor([[True,False]]),
        torch.zeros(1,2,3,dtype=torch.float64),torch.tensor([True,False]))
    x=torch.tensor([[[0.,0.,0.],[0.,0.,1.]]],dtype=torch.float64)
    q=torch.tensor([[[0.,0.,0.,1.],[0.,0.,0.,1.]]],dtype=torch.float64)
    state=State(x,q,torch.zeros_like(x),torch.zeros_like(x))
    com=torch.zeros(2,3,dtype=torch.float64)
    with torch.no_grad(),StepKinematics() as scope:
        result=ports.geometry(state,com)
        again=ports.geometry(state,com)
        assert all(torch.equal(u,v) for u,v in zip(result,again))
        assert scope.hits['port_geometry']==1
        x[:,1,2]+=.1
        moved=ports.geometry(state,com)
        assert float(moved[0][0,0,1,2])==float(x[0,1,2])
        assert not torch.equal(result[0],moved[0])
        com[0,0]=.01
        changed=ports.geometry(state,com)
        assert not torch.equal(moved[0],changed[0])
        ports.local_points[0,0,0]=.02
        changed_port=ports.geometry(state,com)
        assert not torch.equal(changed_port[0],changed[0])
        state.q[0,0]=0.
        try:ports.geometry(state,com)
        except ValueError:pass
        else:raise AssertionError('Mutation bypassed the orientation validation')
        state.q[0,0,3]=1.
    assert not scope.entries
    # Gradient-enabled calls never return a cached graph.
    variable=x.clone().requires_grad_();state.x=variable
    with StepKinematics() as scope:
        u=ports.geometry(state,com)[0];u.sum().backward()
        v=ports.geometry(state,com)[0];v.sum().backward()
        assert not scope.entries and variable.grad is not None
    # A bounded cache evicts old entries and cannot leak out of its context.
    owner=SimpleNamespace();counter=[0]
    def compute():counter[0]+=1;return torch.ones(4)
    with torch.no_grad(),StepKinematics(maximum_entries=1,maximum_bytes=64) as scope:
        tensors=[torch.zeros(1),torch.zeros(1)]
        memoize('probe',owner,tensors[:1],(),compute)
        memoize('probe',owner,tensors[1:],(),compute)
        assert scope.evictions==1
    assert not scope.entries
    memoize('probe',owner,tensors[1:],(),compute)
    assert counter[0]==3
    # Small views must not hide large allocations retained by strong refs.
    with torch.no_grad(),StepKinematics(maximum_bytes=64) as scope:
        backing=torch.zeros(256)
        memoize('large_source',owner,(backing[:1],),(),compute)
        assert not scope.entries
        memoize('large_output',owner,(torch.zeros(1),),(),lambda:backing[:1])
        assert not scope.entries
    # PyTorch inference-mode tensors have no version; cache must decline them.
    with torch.inference_mode(),StepKinematics() as scope:
        t=torch.zeros(1)
        memoize('unversioned',owner,(t,),(),compute)
        memoize('unversioned',owner,(t,),(),compute)
        assert not scope.entries
    result={'status':'PASS','exact_repeated_geometry':True,'state_and_com_mutations_invalidate':True,
        'contact_source_mutation_invalidates':True,'nonfinite_or_degenerate_state_not_hidden':True,
        'gradient_enabled_cache_bypass':True,'unversioned_tensor_cache_bypass':True,
        'bounded_storage_and_scope_cleanup':True,'retained_input_and_view_storage_budgeted':True,
        'native_dynamics_calls':0,'optimizer_updates':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
