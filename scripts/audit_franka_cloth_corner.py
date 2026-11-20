"""Check accepted one-corner cloth lift, contact and deformation; no dynamics."""
import argparse
import hashlib
import json
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
        raise ValueError('An independent shared-model recording is required')
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    if newton_revision()!=run['newton_source']['revision']:
        raise ValueError('Use the recording Newton revision')
    import numpy as np
    import torch
    import warp as wp
    import newton
    from scipy.spatial.transform import Rotation
    from wp_vbd.franka_cloth_corner_scene import build,TABLE_TOP,DURATION,TOOL_OFFSET
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointCollisionOracle
    from wp_vbd.soft_surface_ports import native_state
    from wp_vbd.w1_bag_scene import deformation_metrics
    from wp_vbd.cloth_drape_metrics import drape_metrics
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache')
    wp.init();wp.set_device('cpu')
    def forbid(*args,**kwargs):raise AssertionError('Saved-state audit executes no dynamics')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=forbid;cls.step=forbid
    style=run.get('grasp_style','side')
    e=build(wp,newton,device='cpu',initialize_controls=False,friction=run['physical_parameters']['friction'],grasp_style=style,fold_profile=run.get('fold_profile','edge'))
    scene=export_model(e.model,run['solver_configuration'],'research.franka_cloth_corner',newton,run['physical_dt'],'cpu')
    scene.metadata['detection_motion_budget']=.012
    oracle=SurfaceJointCollisionOracle(e.model,scene,wp,newton)
    with np.load(a.recording/'trajectory.npz') as z:
        x=z['particle_position'].astype(np.float64);bodies=z['body_pose'];times=z['physical_time'];faces=z['faces']
    if len(times)<2:raise ValueError('Nonempty model rollout required')
    triangles=x[:,faces]
    edges=triangles[:,:,1:]-triangles[:,:,:1]
    area=np.linalg.norm(np.cross(edges[:,:,0],edges[:,:,1]),axis=-1)
    area_ratio=area/area[:1]
    rest_gram=edges[0]@edges[0].swapaxes(-1,-2)
    whitened=np.linalg.solve(np.linalg.cholesky(rest_gram)[None],edges)
    stretch=np.sqrt(np.linalg.eigvalsh(whitened@whitened.swapaxes(-1,-2)).clip(0))
    shapes=[deformation_metrics(x[0],value) for value in x]
    fingers=[i for i,label in enumerate(e.model.body_label) if 'finger' in label]
    wrist=next(i for i,label in enumerate(e.model.body_label) if label.rsplit('/',1)[-1]=='fr3_link7')
    tools=Rotation.from_quat(bodies[:,wrist,3:]).apply(np.broadcast_to(TOOL_OFFSET,(len(times),3)).copy())+bodies[:,wrist,:3]
    axis=Rotation.from_quat(bodies[:,wrist,3:]).apply(np.tile([0.,0.,1.],(len(times),1)))
    downward_cosine=-axis[:,2]
    held=times>=(2.1 if style=='side' else 2.0)-1e-8
    settled=times>=max(2.1 if style=='side' else 2.0,float(times[-1])-.4)-1e-8
    drape=[drape_metrics(x[0],points,faces,e.corner_id) for points in x]
    rows=[]
    for i,(points,body) in enumerate(zip(x,bodies)):
        e.state_0.particle_q.assign(points.astype(np.float32));e.state_0.body_q.assign(body)
        state,com,_=native_state(e.model,e.state_0,wp)
        contacts=oracle.query(state)
        depth=contacts.soft.geometry(state,com)[-1][0]
        eligible=contacts.soft.mesh_force_filter(state,com)[0] if contacts.soft.mesh_force_filter is not None else torch.ones_like(depth,dtype=torch.bool)
        owners=contacts.soft.ports.owners[:,0]-e.model.particle_count
        dynamic=eligible&(owners>=0)&(owners<e.model.body_count)
        overlap=depth-contacts.soft.shape_margin
        patch=torch.isin(contacts.soft.corners,torch.tensor(e.patch_ids)).any(-1)
        active=eligible&(depth>0)
        rows.append(dict(time=float(times[i]),
            maximum_robot_cloth_overlap_m=max(0.,float(overlap[dynamic].max())) if bool(dynamic.any()) else 0.,
            finger_contact_rows=[int((active&(owners==finger)).sum()) for finger in fingers],
            patch_finger_contact_rows=[int((active&patch&(owners==finger)).sum()) for finger in fingers],
            corner_tool_distance_m=float(np.linalg.norm(points[e.corner_id]-tools[i])),
            minimum_cloth_vertex_z_m=float(points[:,2].min())))
        if i%60==0:print('geometry',i,'/',len(times)-1,flush=True)
    bilateral=np.array([all(count>0 for count in row['patch_finger_contact_rows']) for row in rows])
    finite=bool(np.isfinite(x).all() and np.isfinite(bodies).all() and np.isfinite(stretch).all())
    no_pins=bool((e.model.particle_mass.numpy()>0).all())
    reasons=[]
    if run['status']!='COMPLETE' or times[-1]<e.duration-1e-8:reasons.append('incomplete authored timeline')
    if not finite:reasons.append('nonfinite recorded state')
    if not no_pins:reasons.append('cloth contains fixed particles')
    if style=='overhead' and downward_cosine.min()<np.cos(np.deg2rad(10.)):
        reasons.append('gripper did not maintain its downward orientation')
    if x[-1,e.corner_id,2]-x[0,e.corner_id,2]<.22:reasons.append('corner did not lift by 22 cm')
    if not held.any() or (x[held,:,2].min()-.0015-TABLE_TOP)<.005:reasons.append('whole cloth is not clear of the table during hold')
    if held.any() and bilateral[held].mean()<.9:reasons.append('corner is not held by both fingers')
    if held.any() and max(rows[i]['corner_tool_distance_m'] for i in np.flatnonzero(held))>.05:reasons.append('corner lost gripper tracking')
    if area_ratio.min()<.25 or stretch.max()>1.6:reasons.append('excessive cloth collapse or stretch')
    if max(row['maximum_robot_cloth_overlap_m'] for row in rows)>.004:reasons.append('large robot-cloth overlap')
    if max(value['rms_m'] for value in shapes)<.008:reasons.append('cloth deformation is not visible')
    # These are scenario-specific appearance checks, not universal cloth
    # admissibility conditions. A rolled sheet can preserve all its areas.
    drape_width=min((drape[i]['panel_width_ratio'] for i in np.flatnonzero(settled)),default=None)
    drape_coherence=min((drape[i]['panel_normal_coherence'] for i in np.flatnonzero(settled)),default=None)
    if drape_width is not None and drape_width<.55:reasons.append('late cloth panel is bundled into a narrow strip')
    if drape_coherence is not None and drape_coherence<.60:reasons.append('late cloth panel has excessive conflicting folds')
    result=dict(status='PASS' if not reasons else 'FAIL',reasons=reasons,
        recording=str(a.recording.resolve()),physical_seconds=float(times[-1]),saved_states_checked=len(times),
        grasp_style=style,minimum_downward_tool_axis_cosine=float(downward_cosine.min()),
        fixed_cloth_particles=int((e.model.particle_mass.numpy()==0).sum()),cloth_attachments=0,
        corner_lift_m=float(x[-1,e.corner_id,2]-x[0,e.corner_id,2]),
        final_cloth_vertical_span_m=float(np.ptp(x[-1,:,2])),
        minimum_whole_cloth_surface_table_clearance_during_hold_m=float(x[held,:,2].min()-.0015-TABLE_TOP) if held.any() else None,
        bilateral_corner_contact_fraction_during_hold=float(bilateral[held].mean()) if held.any() else None,
        maximum_held_corner_tool_distance_m=max((rows[i]['corner_tool_distance_m'] for i in np.flatnonzero(held)),default=None),
        nonrigid_rms_peak_m=max(value['rms_m'] for value in shapes),
        minimum_triangle_area_ratio=float(area_ratio.min()),maximum_triangle_area_ratio=float(area_ratio.max()),
        minimum_principal_stretch=float(stretch.min()),maximum_principal_stretch=float(stretch.max()),
        maximum_robot_cloth_surface_overlap_m=max(row['maximum_robot_cloth_overlap_m'] for row in rows),
        minimum_late_panel_width_ratio=drape_width,minimum_late_panel_normal_coherence=drape_coherence,
        final_severe_panel_hinges=drape[-1]['severe_panel_hinges'],
        drape_appearance_thresholds=dict(minimum_width_ratio=.55,minimum_normal_coherence=.60,
            interval='last 0.4 seconds after the authored lift; excludes initial fold and immediate grip patch'),
        trajectory_sha256=hashlib.sha256((a.recording/'trajectory.npz').read_bytes()).hexdigest(),
        native_dynamics_calls=0,teacher_reads=0,optimizer_updates=0,
        scope='Task and native geometry at saved frames. Simple cloth has no self-contact; this is not a continuous intersection-free guarantee.',samples=rows)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='samples'},indent=2))


if __name__=='__main__':main()
