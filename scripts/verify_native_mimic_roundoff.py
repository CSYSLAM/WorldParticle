"""Verify bounded native precision repair on real W1 poses, without dynamics."""
import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset',type=Path,default=Path.home()/'下载/assets/w1-030/robot.urdf')
    parser.add_argument('--report',type=Path,required=True)
    a=parser.parse_args()
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd.w1_dual_grasp_scene import build,controller
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd.native_mimic_roundoff import project
    from wp_vbd.mimic_constraint import rows
    from wp_vbd.data import State
    from wp_vbd import math3d as so3
    torch.set_num_threads(1);wp.init();wp.set_device('cpu')
    bound=inspect.signature(newton.solvers.SolverVBD).bind(None);bound.apply_defaults()
    cfg={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    def forbidden(*args,**kwargs):raise AssertionError('No dynamics solver in precision verification')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    e=build(wp,newton,asset=a.asset,output=a.report.parent/'precision_asset',device='cpu',style='front',objects='cylinder_ball',friction=2.5)
    scene=export_model(e.model,cfg,'research.w1_precision',newton,e.sim_dt,'cpu')
    solver=SurfaceJointExampleSolver(SimpleNamespace(config={}),scene,e.sim_dt,wp,newton,e.model,e.state_0)
    results=[]
    for time in (0.,.65,1.25,2.,2.8):
        controller(e,time,wp);q=e.model.joint_q.numpy();q[:e.robot_coordinates]=e.control.joint_target_q.numpy()[:e.robot_coordinates]
        newton.eval_fk(e.model,wp.array(q,dtype=wp.float32,device='cpu'),e.model.joint_qd,e.state_0)
        before=e.state_0.body_q.numpy().copy();raw=solver.read(e.state_0);fixed,detail=project(scene,raw)
        assert bool((rows(scene,fixed).value.abs()<=1e-9).all())
        assert torch.equal(raw.x[:,:e.model.particle_count],fixed.x[:,:e.model.particle_count])
        assert torch.equal(raw.v,fixed.v) and torch.equal(raw.omega,fixed.omega)
        assert np.array_equal(before,e.state_0.body_q.numpy())
        scene.metadata['repair_native_mimic_roundoff']=True
        routed=solver.read(e.state_0);assert torch.equal(routed.x,fixed.x) and torch.equal(routed.q,fixed.q)
        scene.metadata['repair_native_mimic_roundoff']=False
        assert torch.equal(solver.read(e.state_0).x,raw.x)
        results.append({'time':time,**detail})
    joint=int(scene.mimic_spec.pairs[0,0]);parent,child=scene.joint_spec.owners[joint]
    frame=so3.multiply(raw.q[:,parent],scene.joint_spec.frames[joint,0,3:])
    axis=so3.rotate(frame,scene.joint_spec.linear_axes[joint,0])
    bad=State(raw.x.clone(),raw.q,raw.v,raw.omega,raw.history);bad.x[:,child]+=axis*.001
    rejected=False
    try:project(scene,bad)
    except ValueError:rejected=True
    assert rejected,'A real one-millimeter violation must not be hidden'
    result={'status':'PASS','cases':results,'particles_and_velocities_unchanged':True,'native_input_unchanged':True,
            'default_dispatch_preserved':True,'genuine_millimeter_violation_rejected':True,'native_dynamics_calls':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(a.report.read_text())


if __name__=='__main__':main()
