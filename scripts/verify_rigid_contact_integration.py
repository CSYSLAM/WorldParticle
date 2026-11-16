"""CPU checks of rigid-contact assembly, support, graph and rollback contracts."""
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import torch
from wp_vbd import math3d as so3
from wp_vbd.data import State
from wp_vbd.rigid_collision import RigidContacts,supports
from wp_vbd.rigid_contact_scene import RigidContactScene
from wp_vbd.rigid_contact_solver import move
from wp_vbd.rigid_contact_system import begin,rows,system,merit
from wp_vbd.response_operator import skew


def fixture():
    values={'mass':np.array([2.,3.]),'inertia':np.array([np.diag([.2,.3,.4]),np.diag([.4,.5,.6])]),'inverse_inertia':np.array([np.diag([5.,1/.3,2.5]),np.diag([2.5,2.,1/.6])]),'inverse_mass':np.array([.5,1/3]),'body_com':np.array([[.01,-.02,.03],[-.02,.01,0.]]),'rest_pose':np.array([[0.,0.,0.,0.,0.,0.,1.],[.2,.1,.2,0.,0.,0.,1.]]),'body_radius':np.array([.3,.3]),'reference_length':np.array(1.),'gravity':np.zeros(3)}
    meta={'body_count':2,'contract':'rigid_contact_primal_dual_v21','contact_alpha':0.,'friction_epsilon':.01,'body_position_representation':'com'}
    scene=RigidContactScene(values,meta,'cpu')
    q=so3.exp(torch.tensor([[[.1,-.2,.05],[-.15,.07,.2],[0.,0.,0.]]],dtype=torch.float64))
    start=State(scene.rest_x[None].clone(),q,torch.zeros(1,3,3,dtype=torch.float64),torch.zeros(1,3,3,dtype=torch.float64))
    owners=torch.tensor([[0,1],[0,2],[1,2]])
    normal=so3.normalize(torch.tensor([[.2,.1,1.],[0.,0.,1.],[0.,1.,0.]],dtype=torch.float64))
    points=torch.tensor([[[.02,0.,-.03],[-.01,.01,.02]],[[.03,.02,-.01],[0.,0.,0.]],[[0.,.03,.01],[.1,0.,0.]]],dtype=torch.float64)
    origin=start.x-so3.rotate(q,scene.body_com)
    skeleton=origin[:,owners]+so3.rotate(q[:,owners],points)
    margins=(((skeleton[:,:,1]-skeleton[:,:,0])*normal).sum(-1)[0]+torch.tensor([.003,.002,-.01],dtype=torch.float64))[:,None].expand(-1,2)/2
    c=RigidContacts(owners,owners.clone(),points,torch.zeros_like(points),normal,margins,torch.zeros_like(normal),torch.tensor([1000.,2000.,500.],dtype=torch.float64),torch.tensor([.1,0.,0.],dtype=torch.float64),torch.tensor([.5,.7,.3],dtype=torch.float64),torch.full((3,),.01,dtype=torch.float64),torch.arange(3),torch.arange(3),1,3,3)
    return scene,start,c


def main():
    torch.set_num_threads(1)
    scene,start,c=fixture();dt=1/300;context=begin(scene,start,c,dt)
    # Independent 12-DOF dense Delassus construction checks both endpoints,
    # nonzero COM offsets and rotation; fixed world contributes exactly zero.
    normal_support=[];tangent_support=[]
    _,q,skeleton,surface,_=c.geometry(scene,start)
    eye=torch.eye(3,dtype=torch.float64)
    for r in range(3):
        wn=0.;wt=torch.zeros(3,3,dtype=torch.float64)
        for s in range(2):
            owner=c.owners[r,s]
            if not scene.movable[owner]:continue
            rotation=so3.matrix(q[0,r,s]);inverse=rotation@scene.inverse_inertia[owner]@rotation.T
            mass_inverse=1/scene.mass[owner]
            ns=skew(skeleton[0,r,s]-start.x[0,owner]);ts=skew(surface[0,r,s]-start.x[0,owner])
            wnormal=mass_inverse*eye-ns@inverse@ns
            wn=wn+c.normal[r]@wnormal@c.normal[r]
            wt=wt+mass_inverse*eye-ts@inverse@ts
        p=eye-c.normal[r,:,None]*c.normal[r,None,:]
        normal_support.append(1/(dt*dt*wn));tangent_support.append(1/(dt*dt*torch.linalg.eigvalsh(p@wt@p)[-1]))
    torch.testing.assert_close(context.rho_n[0],torch.stack(normal_support),rtol=1e-12,atol=1e-8)
    torch.testing.assert_close(context.rho_t[0],torch.stack(tangent_support),rtol=1e-12,atol=1e-8)
    residual,metric,frozen=system(scene,start,start,start.x,start.q,context)
    assert bool(frozen['valid'].all())
    assert bool((frozen['tangent_force'].norm(dim=-1)<=c.mu[None]*frozen['normal_load']+1e-12).all())
    dofs=6*scene.n
    basis=torch.eye(dofs,dtype=torch.float64).reshape(1,scene.n,6,dofs)
    dense=metric(basis).reshape(dofs,dofs)
    torch.testing.assert_close(dense,dense.T,rtol=1e-12,atol=1e-8)
    assert float(torch.linalg.eigvalsh(dense)[0])>=-1e-8
    diagonal=dense.reshape(scene.n,6,scene.n,6).diagonal(dim1=0,dim2=2).permute(2,0,1)
    torch.testing.assert_close(metric.diagonal[0],diagonal,rtol=1e-12,atol=1e-8)
    # The merit derivative freezes the Coulomb normal load, exactly as the
    # contact residual does. Local physical normal K is not a numerical rho.
    increment=torch.zeros(1,scene.n,6,dtype=torch.float64,requires_grad=True)
    trial=move(scene,start,increment,torch.ones(1,scene.n,dtype=torch.float64))
    energy=merit(scene,trial,start,start.x,start.q,context,frozen).sum()
    derivative,=torch.autograd.grad(energy,increment)
    torch.testing.assert_close(derivative,residual,rtol=1e-9,atol=1e-9)
    # Actual active body/body edges merge mobile components, while two
    # independent contacts with the same static world never merge them.
    scene.update_graph(c.owners,torch.tensor([False,True,True]));g0=scene.topology_generation
    assert scene.interaction_id[0]!=scene.interaction_id[1]
    scene._protected_topology='stale';scene.update_graph(c.owners,torch.tensor([True,True,True]))
    assert scene.interaction_id[0]==scene.interaction_id[1] and not hasattr(scene,'_protected_topology')
    scene.update_graph(c.owners,torch.tensor([False,True,True]));assert scene.topology_generation==g0+2
    assert scene.interaction_id[0]!=scene.interaction_id[1]
    from wp_vbd.mixed_solver import MixedSolver
    checkpoint=ROOT/'checkpoints/shared_hierarchy_v20/shared_strict.pt'
    stored=torch.load(checkpoint,weights_only=True,map_location='cpu')
    model=MixedSolver(stored['config']['model']);model.load_state_dict(stored['model'])
    def motion(before,after):
        travel=(after.x-before.x).norm(dim=-1)+scene.body_radius*so3.log(so3.multiply(after.q,so3.conjugate(before.q))).norm(dim=-1)
        return torch.isfinite(travel).all(-1)&(travel.amax(-1)<=.05*scene.length),travel
    scene.geometry_oracle=SimpleNamespace(motion=motion);scene.candidates=c
    output,diagnostic=model(scene,start,dt)
    assert bool(diagnostic['accepted'].all()),diagnostic['iterations']
    loss=output.x.square().sum()+output.omega.square().sum()*dt**2
    loss.backward()
    gradients=[p.grad for p in model.protected_response.parameters() if p.grad is not None]
    assert gradients and all(bool(torch.isfinite(g).all()) for g in gradients)
    norm=sum(float(g.double().square().sum()) for g in gradients)**.5
    assert norm>0,'The coupled contact fixture must exercise shared neural parameters'
    # Rejected physical input returns the full original state and cannot
    # commit an intermediate tangent multiplier as cross-step history.
    boosted=State(start.x,start.q,start.v+100.,start.omega)
    with torch.no_grad():rolled,rejected=model(scene,boosted,dt)
    assert not bool(rejected['accepted'].any())
    for key in ('x','q','v','omega'):assert torch.equal(getattr(rolled,key),getattr(boosted,key))
    assert rolled.history is None and boosted.history is None
    report={'status':'PASS','support_dense_delassus_relative_tolerance':1e-12,'metric_symmetric_psd_and_diagonal':True,'frozen_coulomb_merit_gradient_max_error':float((derivative-residual).abs().max()),'active_graph_join_split_and_static_world_isolation':True,'newton_full_solver_calls':0,'new_training_admissions':0,'scope':'Local COM assembly, inertial support and graph lifecycle. Full trajectory quality and trained neural effectiveness are separate gates.'}
    report['implementation_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/wp_vbd').glob('rigid_contact*.py')}
    report.update(shared_contact_forward_backward_finite=True,contact_fixture_gradient_norm=norm,whole_state_rejection_rollback=True,checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())
    target=ROOT/'reports/rigid_contact_integration_v21/assembly_verification.json';target.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
