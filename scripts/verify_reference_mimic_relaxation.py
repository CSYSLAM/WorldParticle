"""Check the reference-only adapter preserves the native mimic fixed point."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    import numpy as np
    import torch
    import warp as wp
    import newton
    sys.path.append('/home/oem/code/engine/newton/.venv/lib/python3.12/site-packages')
    from newton._src.solvers.vbd import solver_vbd
    from wp_vbd.franka_vbd_scene import build
    from wp_vbd.reference_mimic_relaxation import ReferenceMimicRelaxation
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    def forbidden(*unused,**kwargs):raise AssertionError('No complete dynamics solver in the local projection check')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    e=build(wp,newton,device='cpu',start_time=1.3,grasp_x=0.,grasp_yaw=-np.pi/4)
    model=e.model
    qout=wp.empty_like(model.joint_q);qdout=wp.empty_like(model.joint_qd)
    def error(state):
        newton.eval_ik(model,state,qout,qdout)
        q=qout.numpy();return float(q[8]-q[7])
    pose=e.state_0.body_q.numpy().copy()
    def rotate(q,v):return v+2*np.cross(q[:3],np.cross(q[:3],v)+q[3]*v)
    parent=int(model.joint_parent.numpy()[8]);axis_index=int(model.joint_qd_start.numpy()[8])
    axis=rotate(pose[parent,3:],rotate(model.joint_X_p.numpy()[8,3:],model.joint_axis.numpy()[axis_index]))
    pose[8,:3]+=axis*.001
    def state():
        value=model.state();value.body_q.assign(pose);value.body_qd.zero_();return value
    original=solver_vbd.project_joint_mimics
    with ReferenceMimicRelaxation(1.) as default:
        assert solver_vbd.project_joint_mimics is original
    for invalid in (0.,-1.,1.01,float('nan')):
        try:ReferenceMimicRelaxation(invalid)
        except ValueError:pass
        else:raise AssertionError('Invalid relaxation accepted')
    modified=state();before=error(modified)
    deltas=wp.zeros_like(model.body_qd)
    with ReferenceMimicRelaxation(.1) as adapted:
        solver_vbd.project_joint_mimics(model,modified.body_q,modified.body_qd,
                                       model.body_inv_mass,model.body_inv_inertia,deltas,1/600)
        after_one=error(modified)
        assert abs(after_one-.9*before)<1e-6,(before,after_one)
        for _ in range(49):
            solver_vbd.project_joint_mimics(model,modified.body_q,modified.body_qd,
                                           model.body_inv_mass,model.body_inv_inertia,deltas,1/600)
        after_fifty=error(modified)
        assert abs(after_fifty)<1e-5 and adapted.calls==50,after_fifty
    assert solver_vbd.project_joint_mimics is original
    stock=state();original(model,stock.body_q,stock.body_qd,model.body_inv_mass,model.body_inv_inertia,deltas,1/600)
    stock_error=error(stock)
    assert abs(stock_error)<1e-6,stock_error
    scheduled=state()
    with ReferenceMimicRelaxation(1.,projection_interval=3) as schedule:
        for _ in range(2):
            solver_vbd.project_joint_mimics(model,scheduled.body_q,scheduled.body_qd,
                                           model.body_inv_mass,model.body_inv_inertia,deltas,1/600)
        assert abs(error(scheduled)-before)<1e-9 and schedule.calls==0
        solver_vbd.project_joint_mimics(model,scheduled.body_q,scheduled.body_qd,
                                       model.body_inv_mass,model.body_inv_inertia,deltas,1/600)
        assert abs(error(scheduled))<1e-6 and schedule.calls==1 and schedule.opportunities==3
    assert solver_vbd.project_joint_mimics is original
    result={'status':'PASS','initial_constraint_error_m':before,'relaxed_error_after_one_m':after_one,
            'relaxed_error_after_fifty_m':after_fifty,'stock_error_after_one_m':stock_error,
            'default_does_not_patch_native_function':True,'native_function_restored':True,
            'physical_mimic_relation_retained':True,'full_vbd_steps':0,'featherstone_calls':0,
            'scheduled_native_projection_and_count_checked':True,
            'new_optimizer_updates':0,'scope':'Local native projection on a perturbed actual Franka finger; not a whole-scene convergence guarantee.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(args.report.read_text())


if __name__=='__main__':main()
