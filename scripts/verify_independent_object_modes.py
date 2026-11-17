"""Check independent payload/cloth modes against the real finite joint metric."""
import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    from wp_vbd.newton_source import activate_newton_source,source_root
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    sys.path.append(str(source_root()/'.venv/lib/python3.12/site-packages'))
    from wp_vbd.franka_rigid_cloth_scene import build
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd import articulation_coarse as tree,surface_joint_system as physics,rigid_contact_system as rigid
    from wp_vbd.protected_response import ProtectedCoarse,metric_action
    from wp_vbd.mixed_physics import predictor
    torch.set_num_threads(1);torch.manual_seed(313)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    bound=inspect.signature(newton.solvers.SolverVBD).bind(None);bound.apply_defaults()
    cfg={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    def forbidden(*args,**kwargs):raise AssertionError('A dynamics solver is forbidden in basis verification')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    e=build(wp,newton,device='cpu')
    scene=export_model(e.model,cfg,'research.franka_vbd_rigid_cloth',newton,e.sim_dt,'cpu')
    scene.metadata['detection_motion_budget']=.003
    solver=SurfaceJointExampleSolver(SimpleNamespace(config={}),scene,e.sim_dt,wp,newton,e.model,e.state_0)
    solver.refresh_control(e.control);start=solver.read(e.state_0)
    scene.candidates=solver.oracle.query(start);scene.geometry_oracle=solver.oracle
    scene.candidate_kind=scene.candidates.relation_kind
    context=physics.begin(scene,start,scene.candidates,e.sim_dt)
    tx,tq=predictor(scene,start,e.sim_dt)
    residual,metric,result=physics.system(scene,start,start,tx,tq,context)
    # Deliberately test the hardest allocation: all three object families
    # in one interaction. These test-only graph edges prescribe no motion.
    payload=e.model.particle_count+e.payload_id
    robot=e.model.particle_count+e.helper.endeffector_id
    edges=torch.tensor([[40,payload],[payload,robot]],dtype=torch.long)
    scene.candidate_kind=torch.zeros(2,dtype=torch.long)
    scene.update_graph(edges,torch.ones(2,dtype=torch.bool))
    base=rigid.coarse(scene,start)
    residual=torch.randn_like(residual)*scene.dof
    old=tree.augment(scene,start,base,metric,residual)
    assert max(z.shape[-1] for _,z,*_ in old.blocks)<=12
    scene.metadata['independent_object_motion_modes']=True
    new=tree.augment(scene,start,base,metric,residual)
    projection=ProtectedCoarse(scene,metric,new.blocks,maximum_columns=18)
    assert bool(projection.valid.all())
    chosen=next(z for ports,z,*_ in new.blocks if payload in ports.ids.tolist() and 40 in ports.ids.tolist())
    assert chosen.shape[-1]==18,chosen.shape
    assert not bool(projection(torch.zeros_like(residual)).any())
    u=torch.randn_like(residual)*scene.dof;v=torch.randn_like(residual)*scene.dof
    symmetry=abs(float((u*projection(v)).sum()-(v*projection(u)).sum()))
    assert symmetry<1e-8,symmetry
    energy=float((u*projection(u)).sum());assert energy>=0.
    ports=next(ports for ports,z,*_ in new.blocks if z is chosen)
    z=chosen.new_zeros(1,scene.n,6,chosen.shape[-1]).index_copy(1,ports.ids,chosen)
    error=float((projection(metric_action(metric,z))-z).abs().max())
    assert error<1e-7,error
    # FREE payload directions have no derivatives in the robot constraints.
    free=chosen[:,:,:,:12]
    ids=ports.ids.tolist();payload_local=ids.index(payload)
    assert any(bool(free[0,payload_local,:,i].abs().sum()) and
               not bool(free[0,[j for j,k in enumerate(ids) if k>=e.model.particle_count and k!=payload],:,i].any())
               for i in range(12))
    from wp_vbd.protected_response import protected_update
    scene.metadata['joint_tree_coarse_modes']='scalar_tree_twelve_columns_v307'
    scene.component_batch=True;scene.fast_rigid_contact=True
    identity=torch.eye(6,dtype=torch.float64).expand(1,scene.n,6,6)
    routed=protected_update(SimpleNamespace(config={}),scene,start,metric,new,residual,
                            identity,identity,None,torch.zeros_like(residual),e.sim_dt,False)
    assert bool(routed[2].all()) and bool(torch.isfinite(routed[0]).all())
    assert float((residual*routed[0]).sum())<=0.
    report={'status':'PASS','maximum_columns':18,'actual_columns':chosen.shape[-1],
            'coarse_exactness_error':error,'symmetry_error':symmetry,'positive_energy':energy,
            'zero_residual_preserved':True,'old_twelve_column_path_preserved':True,
            'free_payload_has_independent_directions':True,'selection':scene.articulation_coarse_diagnostic,
            'component_batch_dispatch_honors_eighteen_column_budget':True,
            'vbd_calls':0,'featherstone_calls':0,'new_network_parameters':0}
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(report,indent=2)+'\n');print(a.report.read_text())


if __name__=='__main__':main()
