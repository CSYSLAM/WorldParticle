"""Independently verify grasp, release and containment of a model bag tape."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def inside_shell(point,vertices,faces):
    """Odd ray intersections with an explicit cap over the open bag mouth."""
    import numpy as np
    direction=np.array([.317,.173,1.]);direction/=np.linalg.norm(direction)
    a=vertices[faces[:,0]];ab=vertices[faces[:,1]]-a;ac=vertices[faces[:,2]]-a
    cross=np.cross(np.broadcast_to(direction,ac.shape),ac);det=(ab*cross).sum(-1)
    valid=abs(det)>1e-12;inverse=np.where(valid,1/np.where(valid,det,1.),0.)
    offset=point-a;u=inverse*(offset*cross).sum(-1);q=np.cross(offset,ab)
    v=inverse*(q*direction).sum(-1);t=inverse*(q*ac).sum(-1)
    selected=valid&(u>=-1e-9)&(v>=-1e-9)&(u+v<=1+1e-9)&(t>1e-8)
    hits=np.sort(t[selected]);unique=hits[np.r_[True,np.diff(hits)>1e-7]] if len(hits) else hits
    return len(unique)%2==1


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    from scipy.spatial.transform import Rotation
    from wp_vbd.w1_bag_scene import paper_shell
    folder=args.recording.resolve();run=json.loads((folder/'run.json').read_text());quality=json.loads((folder/'quality.json').read_text())
    assert run['mode']=='shared_network_inference' and run['status']=='COMPLETE' and quality['accepted']
    assert not any(run['forbidden_solver_calls'].values()) and run['teacher_trajectory_reads']==0 and run['optimizer_updates']==0
    assert run['full_authored_timeline_recorded']
    with np.load(folder/'trajectory.npz') as tape:
        particles=tape['particle_position'];poses=tape['body_pose'];times=tape['physical_time'];targets=tape['motor_target']
    assert len(times)==len(particles)==len(poses)==run['frames_completed']+1
    assert np.isfinite(particles).all() and np.isfinite(poses).all()
    rest,faces,bottom,rim=paper_shell();order=np.argsort(np.arctan2(rest[rim,1],rest[rim,0]))
    ring=rim[order];center=len(rest)
    caps=np.column_stack((ring,np.roll(ring,-1),np.full(len(ring),center)))
    closed=np.concatenate((faces,caps));inside=[];speeds=[]
    # Allow a 2 mm surface/contact tolerance by checking a shrunken cube.
    corners=np.array([[a,b,c] for a in (-.023,.023) for b in (-.023,.023) for c in (-.023,.023)])
    for i in np.flatnonzero(times>=4.2):
        vertices=np.vstack((particles[i],particles[i,ring].mean(0)))
        pose=poses[i,run['payload_body_id']];points=Rotation.from_quat(pose[3:]).apply(corners)+pose[:3]
        inside.append(all(inside_shell(point,vertices,closed) for point in points))
        if i:speeds.append(float(np.linalg.norm(poses[i,run['payload_body_id'],:3]-poses[i-1,run['payload_body_id'],:3])/(times[i]-times[i-1])))
    forces=json.loads((folder/'grasp_forces.json').read_text());hold=[r for r in forces if r['time']>=1.]
    bilateral=[min(r['left_finger_normal_load_N'])>1e-4 for r in hold]
    fraction=float(np.mean(bilateral)) if bilateral else 0.
    # Paper shell edges must be exactly shared and the mouth the sole boundary.
    edges=np.sort(np.concatenate((closed[:,[0,1]],closed[:,[1,2]],closed[:,[2,0]])),axis=1)
    _,counts=np.unique(edges,axis=0,return_counts=True);assert (counts==2).all()
    result={'status':'PASS' if inside and all(inside) and fraction>.5 else 'FAIL',
            'all_checked_cube_corners_inside_actual_capped_shell':bool(inside and all(inside)),
            'final_interval_frames_checked':len(inside),'cube_surface_tolerance_m':.002,
            'left_bilateral_contact_fraction_after_grasp':fraction,
            'contact_samples_after_grasp':len(hold),
            'contact_sample_interval_s':[hold[0]['time'],hold[-1]['time']] if hold else None,
            'maximum_contact_sample_gap_s':max((b['time']-a['time'] for a,b in zip(hold,hold[1:])),default=None),
            'minimum_robot_visual_table_clearance_m':quality['minimum_robot_visual_table_clearance_m'],
            'last_interval_maximum_cube_speed_m_s':max(speeds) if speeds else None,
            'source_trajectory_sha256':hashlib.sha256((folder/'trajectory.npz').read_bytes()).hexdigest(),
            'geometry_cap_is_verification_only':True,'no_physics_steps_during_verification':True}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['status']!='PASS':sys.exit(1)


if __name__=='__main__':main()
