"""Verify actual basic-joint graph and combined contact/joint SPD assembly."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import activate_newton_source
activate_newton_source()
import torch
import warp as wp
import newton
import newton.viewer
from wp_vbd.official import make_example
from wp_vbd.general_joint_export import read_controls
from wp_vbd.general_joint_scene import export_model
from wp_vbd.general_joint_system import begin,system,joint_rows,relation_features
from wp_vbd.rigid_contact_runtime import CollisionOracle
from wp_vbd.mixed_physics import predictor
from wp_vbd.data import State
from wp_vbd import math3d as so3


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/general_joint_rows_v24/combined_system.json')
    args=parser.parse_args()
    torch.set_num_threads(1);torch.manual_seed(267)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    example,solvers,_=make_example('basic.example_basic_joints',reference=False)
    def forbidden(*args,**kwargs):raise AssertionError('Full SolverVBD forbidden')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden
    scene=export_model(example.model,solvers[0][1],'basic.example_basic_joints',newton,example.sim_dt,'cpu')
    start=scene.initial_state(example.model,wp);scene.joint_controls=read_controls(scene.joint_spec,example.control)
    candidates=CollisionOracle(example.model,scene,wp,newton).query(start)
    context=begin(scene,start,candidates,example.sim_dt)
    scene.use_exact_inertia=True;tx,tq=predictor(scene,start,example.sim_dt)
    residual,metric,result=system(scene,start,start,tx,tq,context)
    scene.update_graph(candidates.owners,result['active'][0])
    assert len(torch.unique(scene.interaction_id[scene.movable]))==3,'Static world/anchors must not unite the three mechanisms'
    features=relation_features(scene,start,start,context,result,torch.ones_like(start.x[...,0]))
    assert features.shape==(1,len(scene.layout.incidence),32) # 9 value + 9 material + 9 rate + 5 scalar
    assert bool(torch.isfinite(features).all()) and bool(result['valid'].all())
    # Matrix-free bilinear symmetry and positivity on the actual mobile DOFs.
    a=torch.randn(1,scene.n,6,1,dtype=torch.float64)*scene.dof[None,:,:,None]
    b=torch.randn_like(a)*scene.dof[None,:,:,None]
    lhs=(a*metric(b)).sum();rhs=(b*metric(a)).sum();error=float(abs(lhs-rhs)/max(float(abs(lhs)),float(abs(rhs)),1.))
    assert error<1e-12 and float((a*metric(a)).sum())>0
    # Joint part only: exact pose derivative of objective material/drive merit.
    delta=torch.zeros(1,scene.n,6,dtype=torch.float64,requires_grad=True)
    trial=State(start.x+delta[...,:3],so3.multiply(so3.exp(delta[...,3:]),start.q),start.v,start.omega,start.history)
    joint=joint_rows(scene,trial,context);gradient=torch.autograd.grad(joint['energy'].sum(),delta)[0]
    assembled=torch.zeros_like(delta).index_add(1,scene.joint_spec.owners.flatten(),joint['pair_residual'].flatten(1,2))
    derivative_error=float((gradient-assembled).abs().max());assert derivative_error<1e-7,derivative_error
    # Deactivating structural/drive/limit rows removes both forces and graph edges.
    scene.joint_spec.enabled.fill_(False);disabled=joint_rows(scene,start,context)
    assert not bool(disabled['pair_residual'].any()) and not bool(disabled['weight'].any())
    scene.update_graph(candidates.owners,result['active'][0]);assert not bool(scene.joint_potential().any())
    report={'status':'PASS','demo':'basic.example_basic_joints','joint_count':len(scene.joint_spec.owners),'general_joint_relation_rows':3*len(scene.joint_spec.owners),'separate_dynamic_mechanisms':3,'physical_residual_finite':bool(torch.isfinite(residual).all()),'relation_features_finite':True,'matrix_free_metric_symmetry_relative_error':error,'joint_energy_gradient_max_error':derivative_error,'disabled_force_metric_and_graph_zero':True,'independent_collision_queries':1,'full_vbd_calls':0,'new_training_admissions':0,'implementation_sha256':{name:hashlib.sha256((ROOT/'src/wp_vbd'/name).read_bytes()).hexdigest() for name in ('general_joint_scene.py','general_joint_system.py','general_joint_rows.py','general_joint_export.py')},'scope':'Actual model graph, combined residual and sparse metric only. Fixed-budget joint step, winding commit/rollback, live model/control updates and native task remain unimplemented. Shared coarse gain must acquire a finite-joint scope before claiming isolated-joint neural training gradients.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
