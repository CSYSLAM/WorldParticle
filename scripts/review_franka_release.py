"""Qualify an independently executed Franka hold-then-release VBD teacher."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):value.update(chunk)
    return value.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    import numpy as np
    import torch
    import warp as wp
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    import newton
    from wp_vbd.franka_vbd_scene import build
    root=Path(__file__).resolve().parents[1]
    run=json.loads((args.recording/'run.json').read_text())
    if run['status']!='COMPLETE' or run['mode']!='offline_vbd_reference' or run['teacher_release'] is None:
        raise ValueError('A complete native-VBD hold/release recording is required')
    if run['newton_source']['revision']!=newton_revision() or run['reference_vbd_steps']!=run['physical_substeps']:
        raise ValueError('Reference solver/source provenance is incomplete')
    audit=run.get('native_substep_audit')
    if not audit or not audit['enabled'] or audit['steps']!=run['physical_substeps'] or audit['nonpositive_tet_steps']:
        raise ValueError('Every VBD physical step must pass the material orientation audit')
    if digest(args.recording/'source_static.npz')!=run['source_static_sha256']:
        raise ValueError('Source asset archive changed')
    implementation=json.loads((args.recording/'implementation_snapshot_manifest.json').read_text())
    for relative,expected in implementation.items():
        if digest(args.recording/'implementation_snapshot'/relative)!=expected:
            raise ValueError('Reference implementation snapshot changed: '+relative)
    with np.load(args.recording/'trajectory.npz') as z:
        x=z['particle_position'].astype(np.float64);pose=z['body_pose'].astype(np.float64)
        times=z['physical_time'].astype(np.float64);phases=z['phase'];target=z['joint_target'].astype(np.float64)
        tets=z['tets'];rest=z['particle_rest'].astype(np.float64);velocity=z['particle_velocity'].astype(np.float64)
    if len(times)!=run['physical_substeps']+1 or not np.allclose(np.diff(times),run['physical_dt'],rtol=0,atol=1e-12):
        raise ValueError('Complete per-substep physical states are required')
    if not all(np.isfinite(value).all() for value in (x,pose,target,velocity)):
        raise ValueError('Nonfinite state or control in release reference')
    tetra=x[:,tets]
    volume=np.linalg.det(np.transpose(tetra[:,:,1:]-tetra[:,:,:1],(0,1,3,2)))
    ratio=volume/volume[:1]
    if (ratio<=0).any():raise ValueError('Release teacher inverted a tetrahedron')
    release=run['teacher_release'];global_time=run['initial_command_time']+times
    hold_end=np.flatnonzero(global_time<=release['start_time_s']+1e-10)
    if not len(hold_end):raise ValueError('No pre-release hold interval')
    first=float(global_time[0]);hold_seconds=float(global_time[hold_end[-1]]-max(first,3.3))
    if hold_seconds<.49:raise ValueError('At least 0.49 seconds of held grasp before release is required')
    opening=run['physical_parameters']['grip_open'];closed=run['physical_parameters']['grip_closed']
    pre=target[hold_end[-1],-2:];post=target[-1,-2:]
    if np.max(np.abs(pre-closed))>.002 or np.min(post)<opening-.002:
        raise ValueError('Teacher drive must hold closed before release and finish open')
    wp.config.kernel_cache_dir=str(root/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    p=run['physical_parameters'];example=build(wp,newton,device='cpu',initialize_controls=False,
        material_mu=p['tet_material'][0],material_lambda=p['tet_material'][1],arm_drive_ke=p['arm_drive_ke'],
        grasp_x=p['grasp_x'],grasp_yaw=p['grasp_yaw'],grasp_height=p['grasp_height'],
        grip_open=p['grip_open'],grip_closed=p['grip_closed'],friction=p['friction_override'])
    body_names=list(example.model.body_label)
    hand=[i for i,name in enumerate(body_names) if name.endswith('/fr3_hand') or name.endswith('fr3_hand')]
    if not hand:hand=[i for i,name in enumerate(body_names) if name.endswith('/fr3_link7')]
    if len(hand)!=1:raise ValueError('A unique physical hand body is required')
    mass=example.model.particle_mass.numpy().astype(np.float64);center=(x*mass[None,:,None]).sum(1)/mass.sum()
    q=pose[:,hand[0],3:];q=q/np.maximum(np.linalg.norm(q,axis=-1,keepdims=True),1e-30)
    from scipy.spatial.transform import Rotation
    relative=Rotation.from_quat(q).inv().apply(center-pose[:,hand[0],:3])
    held=np.flatnonzero((global_time>=3.3)&(global_time<=release['start_time_s']))
    if not len(held):raise ValueError('No samples in the held-contact evaluation span')
    lift=center[:,2]-center[0,2]
    if float(lift[held].min())<.02:
        raise ValueError('Held-grasp teacher must actually lift the body at least 2 cm above its initial center')
    hold_shift=np.linalg.norm(relative[held]-relative[held[:1]],axis=-1)
    tracking_tolerance=max(.03,.5*float(np.linalg.norm(np.ptp(x[0],axis=0))))
    if float(hold_shift.max())>tracking_tolerance:
        raise ValueError('Soft body does not remain near the gripper during the held grasp')
    after=np.flatnonzero(global_time>=release['start_time_s']+release['duration_s']+.35)
    if not len(after):raise ValueError('The trajectory ends before release recovery can be inspected')
    final_z=float(center[after[-1],2])
    if final_z<.14:raise ValueError('Released body passed below the support surface')
    rest_center=(rest*mass[:,None]).sum(0)/mass.sum();origin=rest-rest_center
    nonrigid=[]
    for state in x:
        centered=state-(state*mass[:,None]).sum(0)/mass.sum()
        u,_,vt=np.linalg.svd((origin*mass[:,None]).T@centered)
        frame=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
        error=centered-origin@frame
        nonrigid.append(float(np.sqrt(((error**2).sum(-1)*mass).sum()/mass.sum())))
    peak=max(nonrigid)
    recovery=float(nonrigid[after[-1]])
    result={'status':'QUALIFIED_HOLD_RELEASE_TEACHER','recording':str(args.recording.resolve()),
        'run_sha256':digest(args.recording/'run.json'),'trajectory_sha256':digest(args.recording/'trajectory.npz'),
        'automatic_task_and_mesh_screen_passed':True,'training_admitted':False,'new_optimizer_updates':0,
        'newton_revision':newton_revision(),'physical_steps':run['physical_substeps'],'hold_seconds_before_release':hold_seconds,
        'release_time_s':release['start_time_s'],'release_duration_s':release['duration_s'],
        'hold_gripper_relative_shift_max_m':float(hold_shift.max()),'tracking_tolerance_m':tracking_tolerance,
        'peak_actual_com_lift_m':float(lift.max()),'minimum_held_com_lift_m':float(lift[held].min()),
        'final_released_body_com_height_m':final_z,'minimum_tet_volume_ratio':float(ratio.min()),
        'peak_nonrigid_mass_rms_m':peak,'post_release_nonrigid_mass_rms_m':recovery,
        'post_release_shape_recovery_fraction':(peak-recovery)/peak if peak>1e-12 else None,
        'source_snapshot_verified':True,'native_substep_audit_passed':True,
        'scope':'Native VBD reference qualification for held-grasp and unload/recovery windows. No neural optimizer update or W1 data.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
