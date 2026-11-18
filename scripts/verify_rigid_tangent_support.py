"""Check numerical friction support without changing physical Coulomb bounds."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from verify_rigid_contact_integration import fixture
    from wp_vbd.rigid_contact_system import begin,system,merit
    from wp_vbd.rigid_contact_solver import move
    torch.set_num_threads(1)
    scene,start,contacts=fixture();dt=1/600;base=begin(scene,start,contacts,dt)
    normal_material=contacts.material.clone();friction=contacts.mu.clone();cases=[]
    displacement=start.x.new_zeros(1,scene.n,6);displacement[0,0,0]=3e-5
    current=move(scene,start,displacement,torch.ones(1,scene.n,dtype=torch.float64))
    for scale in (1.,16.,64.):
        scene.metadata['rigid_tangent_support_scale']=scale;context=begin(scene,start,contacts,dt)
        assert torch.equal(context.rho_n,base.rho_n) and torch.equal(context.rho_t,base.rho_t*scale)
        assert torch.equal(contacts.material,normal_material) and torch.equal(contacts.mu,friction)
        residual,metric,frozen=system(scene,current,start,current.x,current.q,context)
        assert bool(frozen['valid'].all())
        assert bool((frozen['tangent_force'].norm(dim=-1)<=contacts.mu[None]*frozen['normal_load']+1e-12).all())
        increment=torch.zeros_like(displacement,requires_grad=True)
        trial=move(scene,current,increment,torch.ones(1,scene.n,dtype=torch.float64))
        energy=merit(scene,trial,start,current.x,current.q,context,frozen).sum()
        derivative,=torch.autograd.grad(energy,increment)
        torch.testing.assert_close(derivative,residual,rtol=1e-8,atol=1e-8)
        size=6*scene.n;basis=torch.eye(size,dtype=torch.float64).reshape(1,scene.n,6,size)
        dense=metric(basis).reshape(size,size)
        torch.testing.assert_close(dense,dense.T,rtol=1e-12,atol=1e-8)
        assert float(torch.linalg.eigvalsh(dense).min())>=-1e-8
        cases.append({'scale':scale,'coulomb_cap_preserved':True,'normal_material_preserved':True,
                      'merit_gradient_error':float((derivative-residual).abs().max()),'metric_symmetric_psd':True})
    for invalid in (0.,-1.,float('nan'),float('inf')):
        scene.metadata['rigid_tangent_support_scale']=invalid
        try:begin(scene,start,contacts,dt)
        except ValueError:pass
        else:raise AssertionError('Invalid numerical support admitted')
    del scene.metadata['rigid_tangent_support_scale']
    assert torch.equal(begin(scene,start,contacts,dt).rho_t,base.rho_t)
    result={'status':'PASS','device':'cpu','cases':cases,'default_response_unchanged':True,
            'invalid_support_rejected':True,'native_dynamics_calls':0}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
