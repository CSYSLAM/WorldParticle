"""Verify numerical surface friction support and unchanged Coulomb limits."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.soft_surface_ports import SoftSurfacePorts,native_state
    from wp_vbd import surface_contact_rows as physics
    from wp_vbd.data import State
    from wp_vbd.rigid_contact_system import ContactMetric
    torch.set_num_threads(1);wp.init();wp.set_device('cpu')
    def forbidden(*args,**kwargs):raise AssertionError('No dynamics solver in numerical row verification')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    b=newton.ModelBuilder(gravity=(0.,0.,0.));b.add_particle(wp.vec3(0.,0.,.005),wp.vec3(0.),.01,radius=.01)
    b.add_ground_plane();b.color();model=b.finalize(device='cpu')
    model.soft_contact_ke=1000.;model.soft_contact_kd=.1;model.soft_contact_mu=.8
    start_native=model.state();pipeline=newton.CollisionPipeline(model,soft_contact_gap=.02,enable_rigid_soft_full_surface_contact=True)
    contacts=pipeline.contacts();pipeline.collide(start_native,contacts)
    ports=SoftSurfacePorts.snapshot(model,contacts,wp);start,com,_=native_state(model,start_native,wp)
    inverse=start.x.new_zeros(1,len(com),6,6);inverse[0,0,:3,:3]=torch.eye(3,dtype=torch.float64)/float(model.particle_mass.numpy()[0])
    material=physics.export_material(model,ports,wp,newton);dt=.01
    current=State(start.x.clone(),start.q,start.v,start.omega);current.x[0,0,0]+=.0001
    base=physics.begin(ports,start,com,inverse,dt,material,.001)
    rows=[];normal=None
    for scale in (1.,64.,512.):
        context=physics.begin(ports,start,com,inverse,dt,material,.001,tangent_support_scale=scale)
        torch.testing.assert_close(context.rho_t,base.rho_t*scale,rtol=0.,atol=0.)
        frozen=physics.rows(ports,current,com,context)
        assert bool(frozen['valid'].all()) and bool(frozen['active'].any())
        if normal is None:normal=frozen['normal_load'].clone()
        else:assert torch.equal(normal,frozen['normal_load'])
        assert bool((frozen['tangent_force'].norm(dim=-1)<=material.friction[None]*normal+1e-12).all())
        perturb=torch.zeros_like(start.x,requires_grad=True)
        trial=State(current.x+perturb,current.q,current.v,current.omega)
        energy=physics.merit(ports,trial,com,context,frozen).sum();gradient,=torch.autograd.grad(energy,perturb)
        torch.testing.assert_close(gradient,frozen['residual'][...,:3],rtol=1e-9,atol=1e-9)
        dof=torch.zeros(len(com),6,dtype=torch.bool);dof[0,:3]=True
        metric=ContactMetric(start.x.new_zeros(1,len(com),6,6),[frozen['metric_term']],dof)
        size=6*len(com);basis=torch.eye(size,dtype=torch.float64).reshape(1,len(com),6,size)
        dense=metric(basis).reshape(size,size);torch.testing.assert_close(dense,dense.T,rtol=1e-12,atol=1e-9)
        assert float(torch.linalg.eigvalsh(dense).min())>=-1e-9
        rows.append({'scale':scale,'normal_load_unchanged':True,'coulomb_cap_preserved':True,
                     'merit_gradient_error':float((gradient-frozen['residual'][...,:3]).abs().max()),'metric_symmetric_psd':True})
    result={'status':'PASS','device':'cpu','cases':rows,'native_dynamics_calls':0}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
