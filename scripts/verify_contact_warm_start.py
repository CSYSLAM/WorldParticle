"""Verify contact-aware initialization preserves tangents, opening and forces."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    import torch
    from wp_vbd.contact_warm_start import project_closing,prepare
    torch.set_num_threads(1);dtype=torch.float64
    warm=torch.tensor([[[.1,.2,-.3,0.,0.,0.],[.2,.3,-.4,0.,0.,0.],[.4,.5,-.6,.1,.2,.3],[0.,0.,0.,0.,0.,0.]]],dtype=dtype,requires_grad=True)
    frozen=warm.detach().clone();normal=torch.tensor([[0.,0.,1.]],dtype=dtype)
    value,count=project_closing(warm,torch.tensor([0]),normal)
    assert count==1 and torch.equal(value[0,0,:2],warm[0,0,:2]) and float(value[0,0,2])==0
    assert torch.equal(value[:,1:],warm[:,1:]) and torch.equal(warm,frozen)
    value.sum().backward();assert torch.equal(warm.grad[0,0],torch.tensor([1.,1.,0.,1.,1.,1.],dtype=dtype))
    # Several contact normals at a corner: remove both inward components,
    # leave the independent tangent, and do not amplify displacement.
    normals=torch.tensor([[-1.,0.,0.],[0.,0.,1.],[-1.,0.,0.]],dtype=dtype)
    corner,_=project_closing(warm.detach(),torch.tensor([0,0,0]),normals)
    assert torch.allclose(corner[0,0,:3],torch.tensor([0.,.2,0.],dtype=dtype),atol=1e-14)
    assert float(corner.norm())<=float(warm.norm())
    # An opening displacement is never projected, including a duplicated row.
    upward=warm.detach().clone();upward[0,0,2]=.3
    untouched,count=project_closing(upward,torch.tensor([0]),normal)
    assert count==0 and torch.equal(untouched,upward)
    owners=torch.tensor([[3,0,-1,-1],[3,1,-1,-1],[2,1,-1,-1]])
    weights=torch.zeros(3,2,4,dtype=dtype);weights[:,0,0]=1;weights[:,1,1]=1
    class Ports:
        def point_velocity(self,jacobian,update):
            return torch.stack((update[:,owners[:,0],:3],update[:,owners[:,1],:3]),2)
    ports=Ports();ports.owners=owners;ports.weights=weights
    scene=SimpleNamespace(world=3,kind=torch.tensor([0,0,2,2]),movable=torch.tensor([True,True,True,False]))
    snapshot=SimpleNamespace(ports=ports,normal=normal.expand(3,-1))
    context=SimpleNamespace(contacts=SimpleNamespace(soft=snapshot),soft=SimpleNamespace(material=SimpleNamespace(stiffness=torch.ones(3,dtype=dtype)*1e4)))
    depth=torch.tensor([[-1e-5,-1.,-1e-5]],dtype=dtype)
    row={'jacobian':None,'raw_depth':depth,'force_eligible':torch.ones(1,3,dtype=torch.bool)}
    result,info=prepare(scene,None,warm.detach(),context,{'soft_rows':row})
    assert info['projected_particles']==1 and float(result[0,0,2])==0
    assert torch.equal(result[:,1:],warm.detach()[:,1:])
    assert torch.equal(depth,row['raw_depth'])
    row['force_eligible'][0,0]=False
    result,info=prepare(scene,None,warm.detach(),context,{'soft_rows':row})
    assert info['projected_particles']==0 and torch.equal(result,warm.detach())
    report={'status':'PASS','tangential_motion_preserved':True,'separating_motion_preserved':True,
        'independent_corner_normals_supported':True,'dynamic_and_far_contacts_unchanged':True,
        'ineligible_rows_unchanged':True,'input_not_mutated':True,'warm_gradient_verified':True,
        'native_dynamics_calls':0,'optimizer_updates':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n');print(a.report.read_text())


if __name__=='__main__':main()
