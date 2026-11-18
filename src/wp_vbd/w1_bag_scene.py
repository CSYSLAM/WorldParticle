# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
# Bag dimensions and crease formula follow Newton assets/w1_paper_bag/build_blender.py.
"""Simplified W1 packing: dynamic open paper bag and one free cube.

Geometry follows the referenced paper-bag asset's dimensions and crease
formula. The shell is coarser and omits folded handles; no bag pins,
attachments, prescribed object poses or external dynamics are used.
"""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation

from .w1_dual_grasp_scene import prepare,solve_targets

TABLE_TOP=1.
TABLE_BOUNDS=(.445,.985,-.6,.6)
BAG_POSITION=np.array([.54,.08,TABLE_TOP+.0015])
DEPTH,WIDTH,HEIGHT=.13,.32,.25
PICK=np.array([.48,-.27,TABLE_TOP+.0265])
DURATION=4.8
HOLD_START=4.

BAG_PROFILES={
    'stiff':dict(membrane=5e4,membrane_damping=.2,bend=30.,bend_damping=1.,
                 rim_bend=120.,rim_damping=2.,crease_bend=8.,crease_damping=.3),
    'flexible':dict(membrane=300.,membrane_damping=.03,bend=.003,bend_damping=.0003,
                    rim_bend=.06,rim_damping=.002,crease_bend=.0006,crease_damping=.0001),
}


def paper_shell(nx=4,ny=8,nz=8):
    """Welded five-panel shell, open rim, inset gussets, no duplicated seams."""
    vertices=[];faces=[];lookup={}
    def vertex(i,j,k):
        key=(i,j,k)
        if key not in lookup:
            x=DEPTH*(i/nx-.5);y=WIDTH*(j/ny-.5);z=HEIGHT*k/nz
            if j in (0,ny):y-=np.copysign(.016*(1-abs(2*i/nx-1))*np.sin(np.pi*k/nz),y)
            if i in (0,nx):x+=np.copysign(.0015*np.sin(6*np.pi*j/ny+.5)*np.sin(np.pi*k/nz),x)
            lookup[key]=len(vertices);vertices.append((x,y,z))
        return lookup[key]
    def grid(rows,reverse=False):
        for a,b in zip(rows[:-1],rows[1:]):
            for j in range(len(a)-1):
                for tri in ((a[j],a[j+1],b[j+1]),(a[j],b[j+1],b[j])):faces.append(tri[::-1] if reverse else tri)
    grid([[vertex(i,j,0) for j in range(ny+1)] for i in range(nx+1)])
    for i in (0,nx):grid([[vertex(i,j,k) for j in range(ny+1)] for k in range(nz+1)],i==0)
    for j in (0,ny):grid([[vertex(i,j,k) for i in range(nx+1)] for k in range(nz+1)],j==ny)
    v=np.asarray(vertices);f=np.asarray(faces,dtype=np.int32)
    area=np.linalg.norm(np.cross(v[f[:,1]]-v[f[:,0]],v[f[:,2]]-v[f[:,0]]),axis=-1)/2
    if np.any(area<1e-10):raise ValueError('Degenerate paper triangle')
    return v,f,np.flatnonzero(v[:,2]<1e-8),np.flatnonzero(v[:,2]>HEIGHT-1e-8)


def smooth(time,a,b):
    u=float(np.clip((time-a)/(b-a),0.,1.));return u*u*(3-2*u)


def schedule(side,*,bag_profile='stiff',bag_grip='horizontal'):
    if bag_grip not in ('horizontal','oblique'):raise ValueError('Unknown bag grip')
    rim=BAG_POSITION+np.array([-.025,WIDTH/2,HEIGHT+.010])
    if bag_grip=='oblique':rim[2]+=.006
    pickup=PICK+np.array([0.,0.,.008])
    high=TABLE_TOP+HEIGHT+.125
    carry=np.array([BAG_POSITION[0],BAG_POSITION[1]-.025,high])
    drop=carry+np.array([0.,0.,-.065])
    if side=='left':
        if bag_profile=='flexible':
            # Only motor targets move. A gentle downward rim displacement
            # loads the panels against the supported bottom instead of
            # merely rotating the whole bag about its lower edge.
            raised=rim+np.array([0.,.010,-.025])
            loaded=raised+np.array([0.,0.,.015])
            return [(0.,rim+[0.,0.,.08],.035),(.45,rim,.035),(.85,rim,.0015),
                    (1.6,raised,.0015),(3.7,raised,.0015),(4.35,loaded,.0015),
                    (DURATION,loaded,.0015)]
        return [(0.,rim+[0.,0.,.08],.035),(.45,rim,.035),(.85,rim,.0015),
                (DURATION,rim,.0015)]
    if bag_profile=='flexible':
        high+=.025;carry[2]+=.025;drop[2]+=.025
    return [(0.,pickup+[-.04,0.,0.],.028),(.4,pickup,.028),(.85,pickup,.022),
            (1.75,np.array([pickup[0],pickup[1],high]),.022),
            (2.6,carry,.022),(3.,drop,.022),(3.35,drop,.045),
            (3.8,carry,.045),(4.,carry+[-.08,0.,0.],.045),
            (DURATION,carry+[-.08,0.,0.],.045)]


def command(time):
    return next((name for end,name in ((.45,'approach'),(.85,'grasp bag and cube'),
                 (1.75,'lift cube'),(2.6,'transfer over bag'),(3.,'lower to bag mouth'),
                 (3.35,'release cube'),(4.,'withdraw right hand')) if time<end),'settle in bag')


def controller(example,time,wp):
    targets=example.control.joint_target_q.numpy()
    for indices,angles,fingers,points in example.waypoints:
        value=angles[-1];opening=points[-1][2]
        for i,(a,b) in enumerate(zip(points[:-1],points[1:])):
            if time<=b[0]:
                weight=smooth(time,a[0],b[0]);value=angles[i]*(1-weight)+angles[i+1]*weight
                opening=a[2]*(1-weight)+b[2]*weight;break
        targets[indices]=value;targets[fingers]=opening
    example.control.joint_target_q.assign(targets)
    return command(time)


def build(wp,newton,*,asset,output,device='cuda:0',substeps=10,initialize_controls=True,friction=2.5,bag_profile='stiff',bag_grip='horizontal',**unused):
    if bag_profile not in BAG_PROFILES:raise ValueError('Unknown paper bag material profile')
    if bag_grip not in ('horizontal','oblique'):raise ValueError('Unknown bag grip')
    material=BAG_PROFILES[bag_profile]
    output=Path(output);source=prepare(Path(asset),output/'w1_prepared.urdf')
    b=newton.ModelBuilder(gravity=(0.,0.,-9.81))
    b.add_urdf(str(output/'w1_prepared.urdf'),floating=False,collapse_fixed_joints=True,enable_self_collisions=False,joint_damping_mode='passive')
    robot_coordinates=len(b.joint_q);ends=list(b.joint_qd_start[1:])+[len(b.joint_target_ke)]
    b.joint_target_ke[:]=[4000. if 'FINGER' in b.joint_label[j] else 60000.
                         for j in range(len(b.joint_label)) for _ in range(ends[j]-b.joint_qd_start[j])]
    b.joint_target_kd[:]=[30. if k==4000. else 200. for k in b.joint_target_ke]
    b.joint_target_mode[:]=[int(newton.JointTargetMode.POSITION)]*len(b.joint_target_ke)
    b.default_shape_cfg.ke=1e4;b.default_shape_cfg.kd=10.;b.default_shape_cfg.mu=friction
    b.add_shape_box(-1,xform=wp.transform((.715,0.,TABLE_TOP-.04),wp.quat_identity()),
                    hx=.27,hy=.6,hz=.04,color=(.44,.49,.54),label='packing_table')
    for x in (.47,.94):
        for y in (-.53,.53):
            b.add_shape_box(-1,xform=wp.transform((x,y,(TABLE_TOP-.08)/2),wp.quat_identity()),hx=.02,hy=.02,
                            hz=(TABLE_TOP-.08)/2,color=(.23,.27,.30),cfg=b.ShapeConfig(has_particle_collision=False))
    rest,faces,bottom,rim=paper_shell()
    b.add_cloth_mesh(pos=wp.vec3(*BAG_POSITION),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(0.),
                     vertices=rest.tolist(),indices=faces.flatten().tolist(),density=.18,
                     tri_ke=material['membrane'],tri_ka=material['membrane'],tri_kd=material['membrane_damping'],
                     edge_ke=material['bend'],edge_kd=material['bend_damping'],particle_radius=.0012)
    # The folded corners bend more easily than broad paper panels.
    for i,edge in enumerate(b.edge_indices):
        ids=np.asarray(edge);ids=ids[ids>=0];points=rest[ids];hinge=rest[np.asarray(edge)[2:]]
        corner=np.all(abs(hinge[:,1])>WIDTH/2-.001)&np.all(abs(hinge[:,0])>DEPTH/2-.003)
        gusset=np.all(abs(hinge[:,0])<1e-6)&np.all(abs(hinge[:,1])>WIDTH/2-.02)
        if points[:,2].min()>HEIGHT-.04:b.edge_bending_properties[i]=(material['rim_bend'],material['rim_damping'])
        elif corner or gusset or np.all(abs(hinge[:,2])<1e-6):b.edge_bending_properties[i]=(material['crease_bend'],material['crease_damping'])
    for i,face in enumerate(faces):
        if rest[face,2].min()>HEIGHT-.04:
            ke,ka,kd,drag,lift=b.tri_materials[i];b.tri_materials[i]=(2*ke,2*ka,2*kd,drag,lift)
    payload=b.add_body(xform=wp.transform(tuple(PICK),wp.quat_identity()),label='free_packing_cube')
    cfg=b.default_shape_cfg.copy();cfg.density=200.
    b.add_shape_box(payload,hx=.025,hy=.025,hz=.025,cfg=cfg,color=(1.,.55,.08),label='packing_cube')
    b.add_ground_plane(height=-.003);b.color()
    model=b.finalize(device=device,requires_grad=False)
    model.soft_contact_ke=1e4;model.soft_contact_kd=10.;model.soft_contact_mu=friction
    model.shape_material_ke.fill_(1e4);model.shape_material_kd.fill_(10.);model.shape_material_mu.fill_(friction)
    common=dict(model=model,payload_id=payload,robot_coordinates=robot_coordinates,
                sim_substeps=substeps,sim_dt=1/(60*substeps),frame_dt=1/60,bag_rest=rest,bag_faces=faces,
                bag_bottom=bottom,bag_rim=rim,friction=friction,bag_profile=bag_profile,bag_grip=bag_grip)
    if not initialize_controls:return SimpleNamespace(**common,state_0=model.state(),state_1=model.state())
    labels={name.rsplit('/',1)[-1]:j for j,name in enumerate(model.joint_label)}
    starts=model.joint_q_start.numpy();target_starts=model.joint_target_q_start.numpy()
    q=model.joint_q.numpy();waypoints=[];tools=[]
    for side in ('left','right'):
        points=schedule(side,bag_profile=bag_profile,bag_grip=bag_grip)
        # Approach the thin rim from above. The palm and finger mounting
        # brackets stay above the paper; only the opposed pads straddle it.
        pitch=150. if side=='left' and bag_grip=='oblique' else 90.
        rotations=[Rotation.from_euler('y',pitch,degrees=True).as_matrix()]*len(points)
        names,angles,offset=solve_targets(source,side,[p[1] for p in points],rotations)
        indices=np.asarray([target_starts[labels[name]] for name in names]);coords=[starts[labels[name]] for name in names]
        q[coords]=angles[0]
        fingers=[target_starts[labels[f'{side.upper()}_FINGER{i}_JOINT']] for i in (1,2)]
        for i in (1,2):q[starts[labels[f'{side.upper()}_FINGER{i}_JOINT']]]=points[0][2]
        waypoints.append((indices,angles,fingers,points));tools.append((int(model.joint_child.numpy()[labels[names[-1]]]),offset))
    model.joint_q.assign(q);state=model.state();newton.eval_fk(model,model.joint_q,model.joint_qd,state);wp.copy(model.body_q,state.body_q)
    control=model.control();target=control.joint_target_q.numpy();target[:robot_coordinates]=q[:robot_coordinates];control.joint_target_q.assign(target)
    return SimpleNamespace(**common,state_0=state,state_1=model.state(),control=control,waypoints=waypoints,tools=tools)


def fit_bag(rest,current):
    a=rest.mean(0);b=current.mean(0);u,_,vt=np.linalg.svd((rest-a).T@(current-b))
    c=np.eye(3);c[2,2]=np.linalg.det(u@vt);rotation=(u@c@vt).T;position=b-rotation@a
    return position,rotation,float(np.sqrt(np.mean(np.sum((current-(rest@rotation.T+position))**2,axis=1))))


def deformation_metrics(rest,current):
    """Remove best rigid motion before measuring actual material deformation."""
    position,rotation,rms=fit_bag(rest,current)
    residual=current-(rest@rotation.T+position);norm=np.linalg.norm(residual,axis=-1)
    return dict(rms_m=rms,maximum_m=float(norm.max()),p90_m=float(np.quantile(norm,.9)),
                upright_cosine=float(rotation[2,2]))


def assess_recording(recording,report):
    with np.load(Path(recording)/'trajectory.npz') as z:
        x=z['particle_position'];body=z['body_pose'];times=z['physical_time'];tools=z['tool_position']
    rest,faces,bottom,rim=paper_shell();payload=body[:,report['payload_body_id']]
    position,rotation,error=fit_bag(rest,x[-1]);corners=np.array([[a,b,c] for a in (-.025,.025) for b in (-.025,.025) for c in (-.025,.025)])
    world=Rotation.from_quat(payload[-1,3:]).apply(corners)+payload[-1,:3];local=(world-position)@rotation
    contained=bool((abs(local[:,0])<DEPTH/2-.002).all() and (abs(local[:,1])<WIDTH/2-.005).all() and
                   (local[:,2]>-.006).all() and (local[:,2]<HEIGHT-.01).all())
    peak=float(payload[:,2].max()-payload[0,2]);released=times>=3.8
    separation=float(np.linalg.norm(tools[-1,1]-payload[-1,:3]));left_distance=float(np.linalg.norm(x[-1,rim]-tools[-1,0],axis=-1).min())
    result={'full_timeline':report['full_authored_timeline_recorded'],'cube_peak_lift_m':peak,'cube_contained_in_fitted_bag':contained,
            'cube_final_local_bounds':[local.min(0).tolist(),local.max(0).tolist()],
            'bag_upright_cosine':float(rotation[2,2]),'bag_shape_rms_m':error,'right_hand_cube_distance_m':separation,
            'left_hand_rim_distance_m':left_distance,'finite':bool(np.isfinite(x).all() and np.isfinite(body).all()),
            'minimum_triangle_area_ratio':report['minimum_recorded_triangle_area_ratio'],
            'forbidden_solver_calls':report['forbidden_solver_calls']}
    deformation=[deformation_metrics(rest,points) for points in x]
    peak_index=int(np.argmax([row['rms_m'] for row in deformation]))
    result.update(bag_peak_nonrigid_rms_m=deformation[peak_index]['rms_m'],
                  bag_peak_nonrigid_rms_time_s=float(times[peak_index]),
                  bag_peak_nonrigid_displacement_m=max(row['maximum_m'] for row in deformation),
                  bag_peak_nonrigid_p90_m=max(row['p90_m'] for row in deformation))
    loaded=[row for time,row in zip(times,deformation) if time>=3.35]
    result['bag_loaded_peak_nonrigid_rms_m']=max((row['rms_m'] for row in loaded),default=0.)
    if released.any():result['cube_final_speed_m_s']=float(np.linalg.norm((payload[-1,:3]-payload[-7,:3])/(times[-1]-times[-7]))) if len(times)>7 else None
    result['accepted']=bool(result['full_timeline'] and result['finite'] and peak>.10 and contained and separation>.08 and
                            left_distance<.045 and rotation[2,2]>.85 and error<.04 and
                            result['minimum_triangle_area_ratio']>.2 and not any(report['forbidden_solver_calls'].values()))
    if report.get('bag_profile')=='flexible':
        result['visible_material_deformation']=bool(result['bag_peak_nonrigid_rms_m']>.003 and
                                                   result['bag_peak_nonrigid_displacement_m']>.012 and
                                                   result['bag_loaded_peak_nonrigid_rms_m']>.003)
        result['accepted']&=result['visible_material_deformation']
    return result
