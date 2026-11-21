"""Compare actual tet/cloth physical metrics with and without neural-only assembly."""
import argparse
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd.franka_vbd_scene import build as build_tet
    from wp_vbd.franka_cloth_corner_scene import build as build_cloth
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.data import State
    from wp_vbd.mixed_physics import predictor,residual_system
    from wp_vbd.mixed_solver import relation_inputs
    from wp_vbd.mixed_metric import FrozenMixedMetric
    torch.set_num_threads(1);torch.manual_seed(87);wp.init();wp.set_device('cpu')
    bound=inspect.signature(newton.solvers.SolverVBD).bind(None,integrate_with_external_rigid_solver=False,
        particle_enable_self_contact=False);bound.apply_defaults()
    cfg={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    def forbidden(*args,**kwargs):raise AssertionError('No native dynamics in metric verification')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    rows=[]
    for name,builder in [('tet',build_tet),('cloth',build_cloth)]:
        e=builder(wp,newton,device='cpu',initialize_controls=False)
        scene=export_model(e.model,cfg,'research.'+name,newton,1/180,'cpu')
        x=scene.rest_x[None].clone();q=scene.rest_q[None].clone();v=torch.zeros_like(x)
        start=State(x,q,v,v)
        for perturb in (0.,.0002):
            current=start.clone();current.x[:,:e.model.particle_count]+=perturb*torch.randn_like(current.x[:,:e.model.particle_count])
            with torch.no_grad():
                tx,tq=predictor(scene,start,1/180)
                _,_,extras,base=residual_system(scene,current,start,tx,tq,1/180,return_metric_base=True)
                jac,_=relation_inputs(scene.elastic_scene,current,start,extras,1/180,torch.ones_like(x[...,0]))
                old=FrozenMixedMetric(scene,current,start,tx,tq,1/180,extras,jac,base=base,elastic_mode='objective')
                new=FrozenMixedMetric(scene,current,start,tx,tq,1/180,extras,None,base=base,elastic_mode='objective')
                assert torch.equal(old.local,new.local)
                assert len(old.terms)==len(new.terms)
                for u,w in zip(old.terms,new.terms):
                    assert torch.equal(u.ids,w.ids) and torch.equal(u.jacobian,w.jacobian) and torch.equal(u.weight,w.weight)
                direction=torch.randn(len(x),scene.n,6,4,dtype=x.dtype)
                error=float((old(direction)-new(direction)).abs().max())
                assert error==0.
                assert torch.equal(old.diagonal,new.diagonal)
            rows.append({'material':name,'perturbation_m':perturb,'term_count':len(old.terms),'metric_action_max_error':error})
    result={'status':'PASS','cases':rows,'exact_sparse_metric_terms_and_diagonal':True,
            'native_dynamics_calls':0,'optimizer_updates':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
