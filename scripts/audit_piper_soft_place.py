"""Audit saved Piper task, deformation and native geometry without dynamics."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    run=json.loads((a.recording/'run.json').read_text())
    if run['mode']!='shared_network_inference' or any(run['forbidden_solver_calls'].values()) or run['teacher_trajectory_reads']:
        raise ValueError('An independent model recording is required')
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    if newton_revision()!=run['newton_source']['revision']:
        raise ValueError('Use the recorded Newton revision')
    import numpy as np
    import torch
    import warp as wp
    import newton
    from scipy.spatial.transform import Rotation
    from wp_vbd.piper_soft_place_scene import build,TABLE_TOP,CUBE_HALF,TOOL_OFFSET,DURATION
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointCollisionOracle
    from wp_vbd.soft_surface_ports import native_state
    from wp_vbd.w1_bag_scene import deformation_metrics
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache')
    wp.init();wp.set_device('cpu')
    def forbid(*args,**kwargs):
        raise AssertionError('Saved-state audit executes no dynamics')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=forbid;cls.step=forbid
    e=build(wp,newton,asset=run['asset'],output=a.recording/'audit_asset',device='cpu',
        initialize_controls=False,friction=run['physical_parameters']['friction'])
    scene=export_model(e.model,run['solver_configuration'],'research.piper_soft_place',newton,run['physical_dt'],'cpu')
    scene.metadata['detection_motion_budget']=.012
    oracle=SurfaceJointCollisionOracle(e.model,scene,wp,newton)
    with np.load(a.recording/'trajectory.npz') as z:
        x=z['particle_position'].astype(np.float64);bodies=z['body_pose'];times=z['physical_time']
        velocity=z['particle_velocity'];body_velocity=z['body_velocity']
    if len(times)<2:
        raise ValueError('Nonempty rollout required')
    vertices=x[:,e.model.tet_indices.numpy()]
    dm=vertices[:,:,1:]-vertices[:,:,:1]
    ratio=np.linalg.det(dm)/np.linalg.det(dm[:1])
    shape=[deformation_metrics(x[0],points) for points in x]
    center=x.mean(1)
    cube=bodies[:,e.payload_id]
    local=np.stack([(points-pose[:3])@Rotation.from_quat(pose[3:]).as_matrix() for points,pose in zip(x,cube)])
    after=times>=4.7-1e-8
    local_center=local.mean(1)
    gap=local[...,2].min(1)-.003-CUBE_HALF
    fingers=[i for i,label in enumerate(e.model.body_label) if label.rsplit('/',1)[-1] in ('link7','link8')]
    rows=[]
    for index,(points,body) in enumerate(zip(x,bodies)):
        e.state_0.particle_q.assign(points.astype(np.float32));e.state_0.body_q.assign(body)
        state,com,_=native_state(e.model,e.state_0,wp)
        contacts=oracle.query(state)
        depth=contacts.soft.geometry(state,com)[-1][0]
        eligible=contacts.soft.mesh_force_filter(state,com)[0] if contacts.soft.mesh_force_filter is not None else torch.ones_like(depth,dtype=torch.bool)
        owner=contacts.soft.ports.owners[:,0]-e.model.particle_count
        robot=eligible&(owner>=0)&(owner<e.robot_bodies)
        support=eligible&(owner==e.payload_id)
        overlap=depth-contacts.soft.shape_margin
        rigid=contacts.rigid.geometry(scene,state)[-1][0]-contacts.rigid.physical_margin
        rows.append(dict(time=float(times[index]),
            robot_soft_overlap_m=max(0.,float(overlap[robot].max())) if bool(robot.any()) else 0.,
            cube_soft_overlap_m=max(0.,float(overlap[support].max())) if bool(support.any()) else 0.,
            rigid_overlap_m=max(0.,float(rigid.max())) if rigid.numel() else 0.,
            finger_contacts=[int((eligible&(owner==finger)&(depth>0)).sum()) for finger in fingers],
            cube_contact_rows=int((support&(depth>0)).sum())))
        if index%60==0:print('geometry',index,'/',len(times)-1,flush=True)
    held=(times>=1.8)&(times<=2.9)
    stacked=bool(after.any() and (abs(local_center[after,:2])<.012).all()
        and (abs(local[after,:,:2])<CUBE_HALF+.005).all()
        and (abs(gap[after])<.005).all())
    tool_body=next(i for i,label in enumerate(e.model.body_label) if label.rsplit('/',1)[-1]=='link6')
    tool=Rotation.from_quat(bodies[-1,tool_body,3:]).apply(TOOL_OFFSET)+bodies[-1,tool_body,:3]
    reasons=[]
    if run['status']!='COMPLETE' or times[-1]<DURATION-1e-8:reasons.append('incomplete full task')
    if not np.isfinite(x).all() or not np.isfinite(bodies).all():reasons.append('nonfinite state')
    if ratio.min()<=0:reasons.append('inverted material')
    if not held.any() or float((x[held,:,2].min(1)-.003-TABLE_TOP).max())<.05:reasons.append('no whole-block lift above 5 cm')
    if not stacked:reasons.append('soft block is not stably above the rigid cube')
    if float(np.linalg.norm(velocity[-1],axis=-1).max())>.03:reasons.append('soft block has not settled')
    if np.linalg.norm(tool-center[-1])<.06:reasons.append('gripper has not withdrawn')
    if rows[-1]['cube_contact_rows']==0:reasons.append('no final support contact')
    if max(row['robot_soft_overlap_m'] for row in rows)>.004:reasons.append('large robot-soft overlap')
    if max(row['cube_soft_overlap_m'] for row in rows)>.004:reasons.append('large cube-soft overlap')
    result=dict(status='PASS' if not reasons else 'FAIL',reasons=reasons,recording=str(a.recording.resolve()),
        physical_seconds=float(times[-1]),saved_states_checked=len(times),
        whole_block_peak_minimum_surface_table_clearance_m=float((x[:,:,2].min(1)-.003-TABLE_TOP).max()),
        final_soft_center_in_cube_frame_m=local_center[-1].tolist(),
        final_soft_surface_cube_top_gap_m=float(gap[-1]),last_interval_stacked=stacked,
        final_maximum_soft_vertex_speed_m_s=float(np.linalg.norm(velocity[-1],axis=-1).max()),
        gripper_final_soft_center_distance_m=float(np.linalg.norm(tool-center[-1])),
        nonrigid_rms_peak_m=max(value['rms_m'] for value in shape),
        nonrigid_maximum_displacement_m=max(value['maximum_m'] for value in shape),
        minimum_tet_volume_ratio=float(ratio.min()),maximum_tet_volume_ratio=float(ratio.max()),
        maximum_robot_soft_surface_overlap_m=max(row['robot_soft_overlap_m'] for row in rows),
        maximum_cube_soft_surface_overlap_m=max(row['cube_soft_overlap_m'] for row in rows),
        maximum_rigid_surface_overlap_m=max(row['rigid_overlap_m'] for row in rows),
        final_cube_contact_rows=rows[-1]['cube_contact_rows'],
        cube_translation_from_initial_m=(cube[-1,:3]-cube[0,:3]).tolist(),
        trajectory_sha256=hashlib.sha256((a.recording/'trajectory.npz').read_bytes()).hexdigest(),
        native_dynamics_calls=0,teacher_reads=0,optimizer_updates=0,
        scope='Task and discrete native geometry at every saved state, not continuous collision proof or visual acceptance.',samples=rows)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':
    main()
