"""Small W1 dual grasp. IK plans motor targets; both payloads remain dynamic."""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

DURATION=1.4
OPENING=.028
LIFT=.08
CENTERS=np.array([[.40,.32,.9265],[.40,-.32,.9265]])
TOOL_OFFSET=np.array([0.,0.,.125])


def transform(origin):
    out=np.eye(4)
    if origin is not None:
        out[:3,3]=np.fromstring(origin.get('xyz','0 0 0'),sep=' ')
        out[:3,:3]=Rotation.from_euler('xyz',np.fromstring(origin.get('rpy','0 0 0'),sep=' ')).as_matrix()
    return out


def prepare(asset,output):
    """Keep original visuals/inertias, lock torso, and use simple collision boxes."""
    import trimesh
    source=ET.parse(asset).getroot()
    frozen={'ANKLE','KNEE','BUTTOCK','WAIST','NECK1','NECK2'}
    for joint in source.findall('joint'):
        if joint.get('name') in frozen:joint.set('type','fixed')
    for link in source.findall('link'):
        for mesh in link.findall('.//mesh'):
            mesh.set('filename',str((asset.parent/mesh.get('filename')).resolve()))
        for collision in link.findall('collision'):
            geometry=collision.find('geometry');mesh=geometry.find('mesh')
            if mesh is None:continue
            value=trimesh.load(mesh.get('filename'),force='mesh',process=False)
            bounds=np.asarray(value.bounds)*np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
            # The lower finger pad is the useful contact region. Avoid an
            # AABB filling the open space around its mounting bracket.
            if 'finger' in link.get('name','').lower():
                pad=np.asarray(value.vertices);pad=pad[pad[:,1]>-.040]
                bounds=np.array([pad.min(0),pad.max(0)])
            center=bounds.mean(0);pose=transform(collision.find('origin'))
            pose[:3,3]+=pose[:3,:3]@center
            origin=collision.find('origin')
            if origin is None:origin=ET.SubElement(collision,'origin')
            origin.set('xyz',' '.join(map(str,pose[:3,3])))
            geometry.remove(mesh);ET.SubElement(geometry,'box',size=' '.join(map(str,bounds[1]-bounds[0])))
    output.parent.mkdir(parents=True,exist_ok=True)
    ET.ElementTree(source).write(output,encoding='unicode')
    return source


def plan(source,side):
    joints={j.find('child').get('link'):j for j in source.findall('joint')}
    chain=[];node=f'{side}_gripper_base'
    while node in joints:
        j=joints[node];chain.append(j);node=j.find('parent').get('link')
    chain.reverse()
    names=[f'{side.upper()}_J{i}' for i in range(1,8)]
    limits={j.get('name'):j.find('limit') for j in chain if j.get('name') in names}
    lower=np.array([float(limits[n].get('lower')) for n in names]);upper=np.array([float(limits[n].get('upper')) for n in names])
    def fk(q):
        p=np.eye(4);values=dict(zip(names,q))
        for j in chain:
            p=p@transform(j.find('origin'))
            if j.get('name') in values:
                axis=np.fromstring(j.find('axis').get('xyz'),sep=' ')
                move=np.eye(4);move[:3,:3]=Rotation.from_rotvec(axis*values[j.get('name')]).as_matrix();p=p@move
        return p
    target_rotation=Rotation.from_euler('x',np.pi).as_matrix()
    initial=np.array([-.25,-.65,-.3,-1.2,.1,0.,.5])
    if side=='right':initial=np.array([-.25,.65,-.3,1.2,.1,0.,-.5])
    initial=np.clip(initial,lower+1e-5,upper-1e-5)
    result=[]
    center=CENTERS[0 if side=='left' else 1]
    for height in (0.,LIFT):
        target=center+np.array([0.,0.,height])
        seed=initial.copy()
        def residual(q):
            p=fk(q);point=p[:3,3]+p[:3,:3]@TOOL_OFFSET
            angle=Rotation.from_matrix(target_rotation@p[:3,:3].T).as_rotvec()
            return np.concatenate((point-target,.25*angle,.001*(q-seed)))
        fit=least_squares(residual,seed,bounds=(lower+1e-6,upper-1e-6),max_nfev=180,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        p=fk(fit.x);error=np.linalg.norm(p[:3,3]+p[:3,:3]@TOOL_OFFSET-target)
        if error>.002:raise ValueError(f'{side} arm IK target error {error:.4f} m')
        result.append(fit.x);initial=fit.x
    # Fixed chain from the actuated wrist to the gripper tool frame.
    relative=np.eye(4);after=False
    for j in chain:
        if after:relative=relative@transform(j.find('origin'))
        if j.get('name')==names[-1]:after=True
    return names,np.asarray(result),relative[:3,3]+relative[:3,:3]@TOOL_OFFSET


def command(time):
    def smooth(a,b):
        u=float(np.clip((time-a)/(b-a),0.,1.));return u*u*(3-2*u)
    opening=OPENING+(.020-OPENING)*smooth(.05,.45)
    lift=smooth(.50,1.10)
    phase='close' if time<.5 else 'lift' if time<1.1 else 'hold'
    return opening,lift,phase


def build(wp,newton,*,asset,output,device='cuda:0',substeps=4,initialize_controls=True):
    asset=Path(asset).expanduser().resolve();output=Path(output)
    prepared=output/'w1_prepared.urdf'
    source=prepare(asset,prepared)
    b=newton.ModelBuilder(gravity=(0.,0.,-9.81))
    b.add_urdf(str(prepared),floating=False,collapse_fixed_joints=True,enable_self_collisions=False,
               joint_damping_mode='passive')
    robot_coordinates=len(b.joint_q)
    ends=list(b.joint_qd_start[1:])+[len(b.joint_target_ke)]
    b.joint_target_ke[:]=[2000. if 'FINGER' in b.joint_label[j] else 20000.
                          for j in range(len(b.joint_label)) for _ in range(ends[j]-b.joint_qd_start[j])]
    b.joint_target_kd[:]=[20. if k==2000. else 100. for k in b.joint_target_ke]
    b.joint_target_mode[:]=[int(newton.JointTargetMode.POSITION)]*len(b.joint_target_ke)
    b.default_shape_cfg.ke=1e4;b.default_shape_cfg.kd=10.;b.default_shape_cfg.mu=1.5
    b.add_shape_box(-1,xform=wp.transform((.44,0.,.81),wp.quat_identity()),hx=.32,hy=.60,hz=.05,color=(.30,.37,.45))
    for center in CENTERS:
        b.add_shape_box(-1,xform=wp.transform((center[0],center[1],.88),wp.quat_identity()),
                        hx=.055,hy=.055,hz=.02,color=(.50,.56,.62))
    b.add_soft_grid(pos=wp.vec3(*(CENTERS[1]-.025+np.array([0.,0.,.002]))),rot=wp.quat_identity(),vel=wp.vec3(0.),
                    dim_x=2,dim_y=2,dim_z=2,cell_x=.025,cell_y=.025,cell_z=.025,
                    density=200.,k_mu=5000.,k_lambda=5000.,k_damp=1.,particle_radius=.002,
                    color=(.14,.63,.95),label='right_soft_block')
    payload=b.add_body(xform=wp.transform(tuple(CENTERS[0]),wp.quat_identity()),label='left_free_rigid')
    cfg=b.default_shape_cfg.copy();cfg.density=200.
    b.add_shape_box(payload,hx=.025,hy=.025,hz=.025,cfg=cfg,color=(1.,.62,.10))
    b.add_ground_plane(height=-.003);b.color()
    model=b.finalize(device=device,requires_grad=False)
    model.soft_contact_ke=1e4;model.soft_contact_kd=10.;model.soft_contact_mu=1.5
    model.shape_material_ke.fill_(1e4);model.shape_material_kd.fill_(10.);model.shape_material_mu.fill_(1.5)
    common=dict(model=model,payload_id=payload,robot_coordinates=robot_coordinates,
                sim_substeps=substeps,sim_dt=1/(60*substeps),frame_dt=1/60)
    if not initialize_controls:return SimpleNamespace(**common,state_0=model.state(),state_1=model.state())
    labels={name.rsplit('/',1)[-1]:j for j,name in enumerate(model.joint_label)}
    starts=model.joint_q_start.numpy();targets=model.joint_target_q_start.numpy()
    q=model.joint_q.numpy();waypoints=[];tools=[]
    for side in ('left','right'):
        names,angles,offset=plan(source,side)
        indices=np.array([starts[labels[n]] for n in names]);target_indices=np.array([targets[labels[n]] for n in names])
        q[indices]=angles[0]
        fingers=[targets[labels[f'{side.upper()}_FINGER{i}_JOINT']] for i in (1,2)]
        for i in (1,2):q[starts[labels[f'{side.upper()}_FINGER{i}_JOINT']]]=OPENING
        waypoints.append((target_indices,angles,fingers))
        tools.append((int(model.joint_child.numpy()[labels[names[-1]]]),offset))
    model.joint_q.assign(q)
    state=model.state();newton.eval_fk(model,model.joint_q,model.joint_qd,state);wp.copy(model.body_q,state.body_q)
    control=model.control();target=control.joint_target_q.numpy();target[:robot_coordinates]=q[:robot_coordinates];control.joint_target_q.assign(target)
    return SimpleNamespace(**common,state_0=state,state_1=model.state(),control=control,waypoints=waypoints,tools=tools)


def controller(example,time,wp):
    opening,blend,phase=command(time)
    values=example.control.joint_target_q.numpy()
    for indices,angles,fingers in example.waypoints:
        values[indices]=angles[0]+blend*(angles[1]-angles[0]);values[fingers]=opening
    example.control.joint_target_q.assign(values)
    return phase


def assess_recording(recording,report):
    """Measure lift and hold clearance directly from the recorded states."""
    with np.load(Path(recording)/'trajectory.npz') as tape:
        time=tape['physical_time'];x=tape['particle_position'];body=tape['body_pose']
    payload=report['payload_body_id'];hold=time>=1.1-1e-9
    corners=np.array([[u,v,w] for u in (-.025,.025) for v in (-.025,.025) for w in (-.025,.025)])
    rigid_low=np.array([(pose[:3]+Rotation.from_quat(pose[3:]).apply(corners))[:,2].min()
                         for pose in body[:,payload]])
    result=dict(recording=str(Path(recording).resolve()),full_timeline=report['full_authored_timeline_recorded'],
                rigid_final_lift_m=float(body[-1,payload,2]-body[0,payload,2]),
                soft_final_lift_m=float(x[-1,:,2].mean()-x[0,:,2].mean()),
                rigid_minimum_hold_clearance_m=float(rigid_low[hold].min()-.9) if hold.any() else None,
                soft_minimum_hold_clearance_m=float(x[hold,:,2].min()-.9) if hold.any() else None,
                minimum_tet_ratio=report['minimum_recorded_tet_ratio'],
                native_solver_calls=report['forbidden_solver_calls'],backbone_calls=report['nn_backbone_calls'],
                finite=bool(np.isfinite(x).all() and np.isfinite(body).all()))
    result['both_objects_lifted']=bool(result['rigid_final_lift_m']>.015 and result['soft_final_lift_m']>.015 and
        hold.any() and result['rigid_minimum_hold_clearance_m']>.005 and result['soft_minimum_hold_clearance_m']>.005)
    result['accepted']=bool(result['both_objects_lifted'] and result['finite'] and result['full_timeline'] and
        result['minimum_tet_ratio']>.2 and not any(result['native_solver_calls'].values()))
    return result
