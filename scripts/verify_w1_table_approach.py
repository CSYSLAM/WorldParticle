"""Check lateral W1 target geometry, including original visual meshes."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset',type=Path,default=Path.home()/'下载/assets/w1-030/robot.urdf')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import numpy as np
    import warp as wp
    import newton
    from wp_vbd.w1_dual_grasp_scene import build,controller
    from wp_vbd.w1_table_clearance import TableClearance,clip_to_table
    # A triangle crossing the table edge must be measured even if none of
    # its original vertices lies inside the table footprint.
    crossed=clip_to_table(np.array([[-1.,.5,0.],[2.,.5,3.],[.5,2.,1.5]]),[0.,1.,0.,1.])
    assert len(crossed)>0 and abs(crossed[:,2].min()-1.)<1e-12
    def forbidden(*args,**kwargs):raise AssertionError('Geometry validation cannot execute dynamics')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    wp.init();wp.set_device('cpu')
    e=build(wp,newton,asset=args.asset,output=args.output,device='cpu',style='side',objects='cylinder_ball',friction=2.5)
    clearance=TableClearance(e.model,payload_id=e.payload_id)
    rows=[];bodies=[];particles=[];times=np.linspace(0.,3.,61)
    finger_ids=[[i for i,name in enumerate(e.model.body_label) if side+'_finger' in name] for side in ('left','right')]
    for time in times:
        controller(e,float(time),wp);q=e.model.joint_q.numpy()
        q[:e.robot_coordinates]=e.control.joint_target_q.numpy()[:e.robot_coordinates]
        newton.eval_fk(e.model,wp.array(q,dtype=wp.float32,device='cpu'),e.model.joint_qd,e.state_0)
        pose=e.state_0.body_q.numpy();row=clearance.measure(pose);row['time']=float(time);rows.append(row)
        bodies.append(pose.copy());particles.append(e.state_0.particle_q.numpy().copy())
        for ids in finger_ids:
            assert len(ids)==2
            axis=pose[ids[0],:3]-pose[ids[1],:3];axis=axis/np.linalg.norm(axis)
            assert abs(float(axis[2]))<.01 and abs(float(axis[1]))>.99,'Finger closing direction is not horizontal'
    minimum=min(r['minimum_robot_visual_table_clearance_m'] for r in rows if r['minimum_robot_visual_table_clearance_m'] is not None)
    report={'status':'PASS' if minimum>=.002 else 'FAIL','minimum_planned_visual_table_clearance_m':minimum,
            'closing_axis':'horizontal lateral','supports':False,'rows':rows,'native_dynamics_calls':0,
            'scope':'Target geometry only; actual model rollout is verified separately.'}
    (args.output/'geometry_check.json').write_text(json.dumps(report,indent=2)+'\n')
    run={'mode':'asset_geometry_preview','asset':str(args.asset.resolve()),'asset_sha256':hashlib.sha256(args.asset.read_bytes()).hexdigest(),
         'style':'side','objects':'cylinder_ball','friction':2.5,'soft_lame_lambda':e.soft_lambda,'release':False,'full_authored_timeline_recorded':False}
    (args.output/'run.json').write_text(json.dumps(run,indent=2)+'\n')
    np.savez_compressed(args.output/'trajectory.npz',body_pose=bodies,particle_position=particles,physical_time=times)
    print(json.dumps({k:v for k,v in report.items() if k!='rows'},indent=2))
    if report['status']!='PASS':sys.exit(1)


if __name__=='__main__':main()
