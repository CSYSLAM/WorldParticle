"""Verify accepted tangent cache identity, release, bounds and head gradients."""
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
    from torch import nn
    from wp_vbd.surface_tangent_memory import initial_history,match_force,commit,seed_learned
    torch.manual_seed(803);dtype=torch.float64
    state=SimpleNamespace(x=torch.zeros(1,8,3,dtype=dtype),v=torch.zeros(1,8,3,dtype=dtype),
        omega=torch.zeros(1,8,3,dtype=dtype))
    history=initial_history(state,12)
    class Ports:
        node_count=8
        ids=torch.tensor([[0,4,5,6],[1,4,5,6],[2,4,5,6]],dtype=torch.long)
        weights=torch.tensor([[[1.,0.,0.,0.],[0.,.5,.5,0.]]]*3,dtype=dtype)
        local_points=torch.tensor([[[0.,0.,0.],[0.,0.,0.]],
                                   [[.03,0.,0.],[0.,0.,0.]],
                                   [[.10,0.,0.],[0.,0.,0.]]],dtype=dtype)
    class Snapshot:
        shape=torch.tensor([2,2,3])
        corners=torch.tensor([[0,1,-1],[0,1,-1],[2,3,-1]])
        normal=torch.tensor([[0.,0.,1.]]*3,dtype=dtype)
        ports=Ports()
    snap=Snapshot()
    dual=torch.tensor([[[1.,0.,0.],[0.,1.,0.],[0.,1.,0.]]],dtype=dtype)
    history.update(commit(snap,dual,torch.tensor([True]),history))
    carried=match_force(snap,history,torch.tensor([[2.,2.,2.]],dtype=dtype),torch.tensor([.5,.5,.5],dtype=dtype))
    torch.testing.assert_close(carried.norm(dim=-1),torch.tensor([[1.,1.,1.]],dtype=dtype))
    duplicate=SimpleNamespace(shape=torch.tensor([2,2]),corners=torch.tensor([[0,1,-1],[0,1,-1]]),
        normal=torch.tensor([[0.,0.,1.],[0.,0.,1.]],dtype=dtype),
        ports=SimpleNamespace(node_count=8,weights=Ports.weights[:2],local_points=Ports.local_points[[0,0]]))
    duplicate_force=match_force(duplicate,history,torch.tensor([[2.,2.]],dtype=dtype),torch.tensor([.5,.5],dtype=dtype))
    assert int((duplicate_force.norm(dim=-1)>0).sum())==1
    # A normal rotation retensions memory in the new tangent plane and a
    # reduced load enforces the current, rather than stale, friction disk.
    changed=SimpleNamespace(shape=torch.tensor([2]),corners=torch.tensor([[0,1,-1]]),
        normal=torch.tensor([[1.,0.,0.]],dtype=dtype),ports=SimpleNamespace(
            node_count=8,weights=Ports.weights[:1],local_points=Ports.local_points[:1]))
    projected=match_force(changed,history,torch.tensor([[.4]],dtype=dtype),torch.tensor([.25],dtype=dtype))
    assert float(projected.norm())<=.1+1e-12
    released=match_force(snap,history,torch.zeros(1,3,dtype=dtype),torch.tensor([.5,.5,.5],dtype=dtype))
    assert not bool(released.any())
    failed=commit(snap,torch.zeros_like(dual),torch.tensor([False]),history)
    for key in history:torch.testing.assert_close(failed[key],history[key])

    class SeedSnapshot:
        shape=torch.tensor([3,3,3])
        shape_margin=torch.ones(3,dtype=dtype)
        ports=SimpleNamespace(ids=Ports.ids,weights=Ports.weights)
        normal=torch.tensor([[0.,0.,1.]]*3,dtype=dtype)
        def geometry(self,start,body_com):
            point=start.x.new_zeros(1,3,2,3)
            jac=start.x.new_zeros(1,3,2,4,3,6)
            jac[..., :3,:3]=torch.eye(3,dtype=dtype)
            return point,jac,start.x.new_full((1,3),.01)
    seed_snapshot=SeedSnapshot()
    scene=SimpleNamespace(body_com=torch.zeros(8,3,dtype=dtype),length=.1,
        values={'gravity':torch.tensor([0.,0.,-9.81],dtype=dtype)},
        mass=torch.ones(8,dtype=dtype))
    start=SimpleNamespace(x=state.x,v=torch.randn_like(state.v),omega=state.omega)
    network=SimpleNamespace(contact_tangent_head=nn.Sequential(nn.Linear(7,16),nn.Tanh(),nn.Linear(16,2)),
        config={'contact_tangent_seed_scale':.1})
    soft=SimpleNamespace(surface_velocity=torch.zeros(3,3,dtype=dtype),
        material=SimpleNamespace(stiffness=torch.full((3,),1000.,dtype=dtype),friction=torch.full((3,),.5,dtype=dtype)),
        dual_t=torch.zeros(1,3,3,dtype=dtype))
    context=SimpleNamespace(contacts=SimpleNamespace(soft=seed_snapshot),soft=soft,dt=1/30,
        rigid=SimpleNamespace(dual_t=torch.zeros(1,0,3,dtype=dtype)),dual_t=soft.dual_t)
    seed_learned(network,scene,start,context,True)
    context.soft.dual_t.square().sum().backward()
    gradient=float(network.contact_tangent_head[-1].weight.grad.norm())
    limit=soft.material.friction[None]*1000*.01
    assert gradient>0 and bool(torch.isfinite(context.soft.dual_t).all())
    assert bool((context.soft.dual_t.norm(dim=-1)<=limit+1e-12).all())
    result={'status':'PASS','feature_exact_match_and_duplicate_suppression':True,
        'new_normal_projection_and_current_coulomb_bound':True,'release_clears_carried_force':True,
        'rejected_step_does_not_commit':True,'learned_seed_respects_coulomb_disk':True,
        'learned_seed_gradient_norm':gradient,'optimizer_updates':0}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
