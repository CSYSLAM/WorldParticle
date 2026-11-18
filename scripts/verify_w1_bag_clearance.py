"""Audit original gripper visual meshes against every recorded paper surface."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--stride',type=int,default=1)
    p.add_argument('--spacing',type=float,default=.002)
    p.add_argument('--tolerance',type=float,default=.003)
    a=p.parse_args()
    if a.stride<1 or a.spacing<=0 or a.tolerance<0:p.error('Invalid sampling settings')
    os.environ['CUDA_VISIBLE_DEVICES']=''
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    import numpy as np
    import warp as wp
    import newton
    from wp_vbd.w1_bag_scene import build
    from wp_vbd.w1_bag_clearance import GripperBagClearance
    def forbidden(*args,**kwargs):raise AssertionError('Geometry audit cannot execute dynamics')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    folder=a.recording.resolve();run=json.loads((folder/'run.json').read_text())
    if hashlib.sha256(Path(run['asset']).read_bytes()).hexdigest()!=run['asset_sha256']:raise ValueError('Changed source asset')
    if run['newton_source']['revision']!=newton_revision():raise ValueError('Use the recorded Newton source revision')
    wp.init();wp.set_device('cpu')
    e=build(wp,newton,asset=run['asset'],output=folder/'clearance_asset',device='cpu',initialize_controls=False,
            bag_profile=run.get('bag_profile','stiff'),bag_grip=run.get('bag_grip','horizontal'))
    audit=GripperBagClearance(e.model,spacing=a.spacing)
    tape=folder/('trajectory.npz' if (folder/'trajectory.npz').exists() else 'progress.npz')
    with np.load(tape) as z:particles=z['particle_position'];poses=z['body_pose'];times=z['physical_time'];faces=z['faces']
    rows=[]
    for i in sorted(set(range(0,len(times),a.stride))|{len(times)-1}):
        row=dict(frame=i,time=float(times[i]),**audit.measure(poses[i],particles[i],faces));rows.append(row)
        if len(rows)%30==0:print(json.dumps({'checked':len(rows),'last_frame':i,'depth':row['maximum_sampled_paper_inside_visual_depth_m']}),flush=True)
    worst=max(rows,key=lambda r:r['maximum_sampled_paper_inside_visual_depth_m'])
    unresolved=sorted({i for row in rows for i in row['unresolved_open_visual_shapes']})
    result=dict(status='PASS' if worst['maximum_sampled_paper_inside_visual_depth_m']<=a.tolerance and not unresolved else 'FAIL',
                scope='Original left wrist/palm and both finger visual solids; triangle interiors sampled. No dynamics.',
                maximum_sampled_depth_m=worst['maximum_sampled_paper_inside_visual_depth_m'],worst_frame=worst['frame'],
                checked_frames=len(rows),sample_spacing_m=a.spacing,allowed_sampled_depth_m=a.tolerance,
                watertight_visual_meshes=sum(mesh.is_watertight for _,_,mesh in audit.meshes),
                unresolved_open_visual_shapes=unresolved,
                trajectory_sha256=hashlib.sha256(tape.read_bytes()).hexdigest(),rows=rows)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if result['status']!='PASS':sys.exit(1)


if __name__=='__main__':main()
