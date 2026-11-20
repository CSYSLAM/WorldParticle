"""Continuity and energy/force consistency of a hinge crossing atan2's cut."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    import torch
    from wp_vbd.bending_angle import difference
    from wp_vbd.material_damping import dihedral,damping_system,damping_potential
    from wp_vbd.physics import elastic_nodes
    scene=SimpleNamespace(periodic_bending=True)
    angle=torch.tensor([-torch.pi+1e-6,torch.pi-1e-6],dtype=torch.float64,requires_grad=True)
    reference=torch.tensor([torch.pi-1e-6,-torch.pi+1e-6],dtype=torch.float64)
    delta=difference(scene,angle,reference)
    assert torch.allclose(delta,torch.tensor([2e-6,-2e-6],dtype=torch.float64),rtol=1e-9,atol=1e-15)
    gradient=torch.autograd.grad(delta.sum(),angle)[0];assert torch.allclose(gradient,torch.ones_like(angle))
    scene.periodic_bending=False;assert torch.equal(difference(scene,angle,reference),angle-reference)
    # A real four-point hinge with nearly opposing normals, on either side
    # of the signed-angle representation boundary.
    x=torch.tensor([[[.5,1.,0.],[.5,1.,1e-6],[0.,0.,0.],[1.,0.,0.]]],dtype=torch.float64)
    old=x.clone();old[:,1,2]=-1e-6
    empty=torch.empty(0,4,dtype=torch.long);values={}
    physical=SimpleNamespace(periodic_bending=True,n=4,values=values,tets=empty,triangles=torch.empty(0,3,dtype=torch.long),
        bend_edges=torch.tensor([[0,1,2,3]]),bending_indices=torch.tensor([0]),bend_material=torch.tensor([[.3,.02]],dtype=torch.float64),
        bend_length=torch.tensor([1.],dtype=torch.float64),bend_angle=dihedral(old[:,None]).detach()[0],
        material_damping_flags=(False,False,True),redistribute_structural_energy=False)
    values['bend_edges']=physical.bend_edges
    dt=1/180;variable=x.clone().requires_grad_();force,_,_=damping_system(physical,variable,old,dt)
    potential=damping_potential(physical,variable,old,dt).sum();automatic=torch.autograd.grad(potential,variable)[0]
    error=float((force-automatic).abs().max());assert error<1e-12,error
    assert float(potential)<1e-10,float(potential)
    elastic,blocks=elastic_nodes(physical,x)
    assert bool(torch.isfinite(elastic).all()) and float(elastic.abs().max())<1e-4
    # The same geometry under the legacy raw subtraction has a large jump.
    physical.periodic_bending=False
    legacy=float(damping_potential(physical,x,old,dt).sum())
    assert legacy>1.,legacy
    result={'status':'PASS','angle_difference':delta.detach().tolist(),
        'damping_force_energy_gradient_error':error,'periodic_damping_energy':float(potential.detach()),
        'legacy_damping_energy':legacy,'native_dynamics_calls':0,'optimizer_updates':0,
        'scope':'Local continuity at the representation boundary, unchanged legacy option, real hinge force/potential gradient. A relative turn of pi remains a local chart boundary.'}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(a.report.read_text())


if __name__=='__main__':main()
