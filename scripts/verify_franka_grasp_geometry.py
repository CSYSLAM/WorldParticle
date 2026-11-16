"""Check an open, side-aligned Franka grasp before any dynamics is run."""
import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.report.exists():raise FileExistsError(a.report)
    import torch
    import warp as wp
    import newton
    sys.path.append('/home/oem/code/engine/newton/.venv/lib/python3.12/site-packages')
    from wp_vbd.franka_vbd_scene import build,controller
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd.soft_surface_ports import native_state
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    bound=inspect.signature(newton.solvers.SolverVBD).bind(None);bound.apply_defaults()
    cfg={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    def forbidden(*unused,**kwargs):raise AssertionError('Geometry verification must not construct or execute a dynamics solver')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    e=build(wp,newton,device='cpu',start_time=1.3,material_mu=1e6,material_lambda=1e6,arm_drive_ke=20000.,
            grasp_x=0.,grasp_yaw=-.7853981633974483,grip_closed=.024)
    scene=export_model(e.model,cfg,'research.franka_vbd_block',newton,e.sim_dt,'cpu');scene.metadata['detection_motion_budget']=.003
    solver=SurfaceJointExampleSolver(SimpleNamespace(config={}),scene,e.sim_dt,wp,newton,e.model,e.state_0)
    rows=[]
    for time in (1.3,2.1,3.3):
        controller(e,time,wp);scratch=e.model.state()
        newton.eval_fk(e.model,e.control.joint_target_q,e.model.joint_qd,scratch)
        state,com,_=native_state(e.model,scratch,wp);contacts=solver.oracle.query(state)
        rigid_depth=contacts.rigid.geometry(scene,state)[-1][0]
        depth=contacts.soft.geometry(state,com)[-1][0]
        callback=contacts.soft.mesh_force_filter
        eligible=callback(state,com)[0] if callback is not None else torch.ones_like(depth,dtype=torch.bool)
        active=(depth>1e-5)&eligible;owners=contacts.soft.ports.owners[:,0]-e.model.particle_count
        counts={str(owner):int((active&(owners==owner)).sum()) for owner in torch.unique(owners[active]).tolist()}
        dynamic=owners<e.model.body_count
        maximum=max(0.,float(depth[dynamic&eligible].max())) if bool((dynamic&eligible).any()) else 0.
        rows.append({'command_time':time,'rigid_penetrations_over_10um':int((rigid_depth>1e-5).sum()),
                     'soft_contact_counts_by_body':counts,'max_robot_soft_penetration_m':maximum})
    assert rows[0]['max_robot_soft_penetration_m']<=1e-5,rows[0]
    assert rows[1]['soft_contact_counts_by_body'].get('7',0)>0 and rows[1]['soft_contact_counts_by_body'].get('8',0)>0,rows[1]
    assert all(row['rigid_penetrations_over_10um']==0 for row in rows),rows
    result={'status':'PASS','independent_initial_state_open_and_clear':True,'samples':rows,
            'full_vbd_calls':0,'featherstone_calls':0,
            'scope':'Offline geometry of initial, closed-command and lifted-command poses. Closing contacts are deliberate; this is not a dynamics/grasp success check.'}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(a.report.read_text())


if __name__=='__main__':main()
