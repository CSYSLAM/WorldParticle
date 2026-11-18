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


def scene_settings(style='overhead'):
    if style=='side':
        return dict(centers=np.array([[.48,.32,1.1065],[.48,-.32,1.1065]]),
                    table_top=1.08,support_top=1.08,duration=3.,hold_start=2.6)
    if style=='front':
        return dict(centers=np.array([[.50,.32,1.1465],[.50,-.32,1.1365]]),
                    table_top=1.08,support_top=1.12,duration=2.8,hold_start=2.4)
    if style!='overhead':raise ValueError('Unknown grasp style')
    return dict(centers=CENTERS,table_top=.86,support_top=.9,duration=DURATION,hold_start=1.1)


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


def solve_targets(source,side,targets,rotations,*,tolerance=.002):
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
    initial=np.array([-.25,-.65,-.3,-1.2,.1,0.,.5])
    if side=='right':initial=np.array([-.25,.65,-.3,1.2,.1,0.,-.5])
    initial=np.clip(initial,lower+1e-5,upper-1e-5)
    result=[]
    for target,target_rotation in zip(targets,rotations,strict=True):
        seed=initial.copy()
        def residual(q):
            p=fk(q);point=p[:3,3]+p[:3,:3]@TOOL_OFFSET
            angle=Rotation.from_matrix(target_rotation@p[:3,:3].T).as_rotvec()
            return np.concatenate((point-target,.25*angle,.001*(q-seed)))
        fit=least_squares(residual,seed,bounds=(lower+1e-6,upper-1e-6),max_nfev=180,xtol=1e-9,ftol=1e-9,gtol=1e-9)
        p=fk(fit.x);error=np.linalg.norm(p[:3,3]+p[:3,:3]@TOOL_OFFSET-target)
        if error>tolerance:raise ValueError(f'{side} arm IK target error {error:.4f} m')
        result.append(fit.x);initial=fit.x
    # Fixed chain from the actuated wrist to the gripper tool frame.
    relative=np.eye(4);after=False
    for j in chain:
        if after:relative=relative@transform(j.find('origin'))
        if j.get('name')==names[-1]:after=True
    return names,np.asarray(result),relative[:3,3]+relative[:3,:3]@TOOL_OFFSET


def plan(source,side,*,style='overhead'):
    target_rotation=(Rotation.from_euler('y',np.pi/2)*Rotation.from_euler('z',np.pi/2 if side=='left' else -np.pi/2)).as_matrix() if style=='front' else Rotation.from_euler('x',np.pi).as_matrix()
    if style=='side':target_rotation=Rotation.from_euler('y',np.pi/2).as_matrix()
    center=scene_settings(style)['centers'][0 if side=='left' else 1]
    offsets=([[-.005,0.,.002],[0.,0.,.002],[0.,0.,.092]]+
             [[-d,0.,.092] for d in (.025,.05,.075,.10)]) if style=='front' else [[0.,0.,0.],[0.,0.,LIFT]]
    if style=='side':offsets=([[-.04,0.,.008],[0.,0.,.008],[0.,0.,.098]]+
                              [[-d,0.,.098] for d in (.025,.05,.075,.10)])
    return solve_targets(source,side,center+np.asarray(offsets),[target_rotation]*len(offsets))

def command(time,*,release=False,style='overhead'):
    def smooth(a,b):
        u=float(np.clip((time-a)/(b-a),0.,1.));return u*u*(3-2*u)
    if style=='side':
        if release:raise ValueError('Side grasp uses sustained holding')
        opening=.028+(.022-.028)*smooth(.35,.75)
        phase=next((name for end,name in ((.4,'reach forward'),(.85,'close'),(1.6,'lift'),(2.6,'draw back')) if time<end),'hold')
        draw=smooth(1.6,2.6)*4
        return opening,np.array([smooth(0.,.4),smooth(.85,1.45),*[float(np.clip(draw-i,0.,1.)) for i in range(4)]]),phase
    if style=='front':
        if release:raise ValueError('Front grasp uses sustained holding')
        opening=.028+(.024-.028)*smooth(.05,.50)
        # Retain contact while easing preload after capture; excessive
        # squeeze can extrude a soft sphere toward a flat pad's edge.
        opening+=(.0265-.024)*smooth(.8,1.2)
        phase=next((name for end,name in ((.25,'reach forward'),(.65,'close'),(1.4,'lift'),(2.4,'draw back')) if time<end),'hold')
        draw=smooth(1.4,2.4)*4
        return opening,np.array([smooth(0.,.25),smooth(.65,1.25),*[float(np.clip(draw-i,0.,1.)) for i in range(4)]]),phase
    opening=OPENING+(.020-OPENING)*smooth(.05,.45)
    lift=smooth(.50,1.10)
    phase='close' if time<.5 else 'lift' if time<1.1 else 'hold'
    if release:
        opening+=(.038-.020)*smooth(1.4,1.8)
        if time>=1.4:phase='relax grip' if time<1.8 else 'settle'
    return opening,lift,phase


def sphere_tets():
    """A closed icosphere with a conforming center-fan volume mesh."""
    import trimesh
    surface=trimesh.creation.icosphere(subdivisions=1,radius=.025)
    vertices=np.vstack((np.zeros((1,3)),np.asarray(surface.vertices)))
    tets=np.column_stack((np.zeros(len(surface.faces),dtype=np.int32),np.asarray(surface.faces)+1))
    p=vertices[tets];det=np.linalg.det(p[:,1:]-p[:,:1])
    flip=det<0;tets[flip,1:3]=tets[flip,1:3][:,::-1]
    p=vertices[tets]
    if np.any(np.linalg.det(p[:,1:]-p[:,:1])<=1e-15):raise ValueError('Degenerate sphere volume mesh')
    return vertices,tets


def build(wp,newton,*,asset,output,device='cuda:0',substeps=4,initialize_controls=True,
          objects='blocks',friction=1.5,release=False,style='overhead',soft_lambda=None):
    if objects not in ('blocks','cylinder_ball'):raise ValueError('Unknown payload geometry')
    if not np.isfinite(friction) or friction<0:raise ValueError('Finite nonnegative friction required')
    if soft_lambda is None:soft_lambda=5e5 if style in ('front','side') else 5000.
    if not np.isfinite(soft_lambda) or soft_lambda<0:raise ValueError('Finite nonnegative Lame lambda required')
    layout=scene_settings(style);centers=layout['centers'];support_top=layout['support_top']
    asset=Path(asset).expanduser().resolve();output=Path(output)
    prepared=output/'w1_prepared.urdf'
    source=prepare(asset,prepared)
    b=newton.ModelBuilder(gravity=(0.,0.,-9.81))
    b.add_urdf(str(prepared),floating=False,collapse_fixed_joints=True,enable_self_collisions=False,
               joint_damping_mode='passive')
    robot_coordinates=len(b.joint_q)
    ends=list(b.joint_qd_start[1:])+[len(b.joint_target_ke)]
    finger_ke=4000. if style in ('front','side') else 2000.;arm_ke=60000. if style in ('front','side') else 20000.
    b.joint_target_ke[:]=[finger_ke if 'FINGER' in b.joint_label[j] else arm_ke
                          for j in range(len(b.joint_label)) for _ in range(ends[j]-b.joint_qd_start[j])]
    b.joint_target_kd[:]=[(30. if style in ('front','side') else 20.) if k==finger_ke else (200. if style in ('front','side') else 100.) for k in b.joint_target_ke]
    b.joint_target_mode[:]=[int(newton.JointTargetMode.POSITION)]*len(b.joint_target_ke)
    b.default_shape_cfg.ke=1e4;b.default_shape_cfg.kd=10.;b.default_shape_cfg.mu=friction
    b.add_shape_box(-1,xform=wp.transform((.715 if style=='side' else .59 if style=='front' else .44,0.,layout['table_top']-.05),wp.quat_identity()),
                    hx=.27 if style in ('front','side') else .32,hy=.60,hz=.05,color=(.30,.37,.45))
    for center in (() if style=='side' else centers):
        if style=='front':
            # Open central channel lets the lower jaw enter horizontally.
            for offset in (-.025,.025):
                b.add_shape_box(-1,xform=wp.transform((center[0],center[1]+offset,support_top-.02),wp.quat_identity()),
                                hx=.020,hy=.003,hz=.02,color=(.50,.56,.62))
        else:b.add_shape_box(-1,xform=wp.transform((center[0],center[1],support_top-.02),wp.quat_identity()),
                             hx=.055,hy=.055,hz=.02,color=(.50,.56,.62))
    if objects=='blocks':b.add_soft_grid(pos=wp.vec3(*(centers[1]-.025+np.array([0.,0.,.002]))),rot=wp.quat_identity(),vel=wp.vec3(0.),
                    dim_x=2,dim_y=2,dim_z=2,cell_x=.025,cell_y=.025,cell_z=.025,
                    density=200.,k_mu=5000.,k_lambda=soft_lambda,k_damp=1.,particle_radius=.002,
                    color=(.14,.63,.95),label='right_soft_block')
    else:
        vertices,tets=sphere_tets()
        b.add_soft_mesh(pos=wp.vec3(*(centers[1]+np.array([0.,0.,.002]))),rot=wp.quat_identity(),vel=wp.vec3(0.),
                        vertices=[wp.vec3(*p) for p in vertices],indices=tets.flatten().tolist(),scale=1.,
                        density=200.,k_mu=5000.,k_lambda=soft_lambda,k_damp=1.,particle_radius=.002,
                        color=(.14,.63,.95),label='right_soft_ball',validate_mesh=True)
    payload=b.add_body(xform=wp.transform(tuple(centers[0]),wp.quat_identity()),label='left_free_rigid')
    cfg=b.default_shape_cfg.copy();cfg.density=200.
    if objects=='blocks':b.add_shape_box(payload,hx=.025,hy=.025,hz=.025,cfg=cfg,color=(1.,.62,.10))
    else:b.add_shape_cylinder(payload,radius=.025,half_height=.025,cfg=cfg,color=(1.,.62,.10))
    b.add_ground_plane(height=-.003);b.color()
    model=b.finalize(device=device,requires_grad=False)
    model.soft_contact_ke=1e4;model.soft_contact_kd=10.;model.soft_contact_mu=friction
    model.shape_material_ke.fill_(1e4);model.shape_material_kd.fill_(10.);model.shape_material_mu.fill_(friction)
    common=dict(model=model,payload_id=payload,robot_coordinates=robot_coordinates,
                sim_substeps=substeps,sim_dt=1/(60*substeps),frame_dt=1/60,objects=objects,release=release,style=style,soft_lambda=soft_lambda)
    if not initialize_controls:return SimpleNamespace(**common,state_0=model.state(),state_1=model.state())
    labels={name.rsplit('/',1)[-1]:j for j,name in enumerate(model.joint_label)}
    starts=model.joint_q_start.numpy();targets=model.joint_target_q_start.numpy()
    q=model.joint_q.numpy();waypoints=[];tools=[]
    for side in ('left','right'):
        names,angles,offset=plan(source,side,style=style)
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
    opening,blend,phase=command(time,release=example.release,style=example.style)
    values=example.control.joint_target_q.numpy()
    for indices,angles,fingers in example.waypoints:
        value=angles[0]+np.sum(blend[:,None]*np.diff(angles,axis=0),axis=0) if example.style in ('front','side') else angles[0]+blend*(angles[1]-angles[0])
        values[indices]=value;values[fingers]=opening
    example.control.joint_target_q.assign(values)
    return phase


def assess_recording(recording,report):
    """Measure lift and hold clearance directly from the recorded states."""
    with np.load(Path(recording)/'trajectory.npz') as tape:
        time=tape['physical_time'];x=tape['particle_position'];body=tape['body_pose']
    release=report.get('release',False)
    layout=scene_settings(report.get('style','overhead'));top=layout['support_top']
    payload=report['payload_body_id'];hold=(time>=layout['hold_start']-1e-9)&((time<=1.4+1e-9) if release else True)
    corners=np.array([[u,v,w] for u in (-.025,.025) for v in (-.025,.025) for w in (-.025,.025)])
    rigid_low=np.array([(pose[:3]+Rotation.from_quat(pose[3:]).apply(corners))[:,2].min()
                         for pose in body[:,payload]])
    if report.get('objects')=='cylinder_ball':
        axis=Rotation.from_quat(body[:,payload,3:]).apply([0.,0.,1.])[:,2]
        rigid_low=body[:,payload,2]-.025*(np.abs(axis)+np.sqrt(np.maximum(0.,1-axis**2)))
    result=dict(recording=str(Path(recording).resolve()),full_timeline=report['full_authored_timeline_recorded'],
                rigid_final_lift_m=float(body[-1,payload,2]-body[0,payload,2]),
                soft_final_lift_m=float(x[-1,:,2].mean()-x[0,:,2].mean()),
                rigid_minimum_hold_clearance_m=float(rigid_low[hold].min()-top) if hold.any() else None,
                soft_minimum_hold_clearance_m=float(x[hold,:,2].min()-top) if hold.any() else None,
                minimum_tet_ratio=report['minimum_recorded_tet_ratio'],
                native_solver_calls=report['forbidden_solver_calls'],backbone_calls=report['nn_backbone_calls'],
                finite=bool(np.isfinite(x).all() and np.isfinite(body).all()))
    result['rigid_peak_lift_m']=float(body[:,payload,2].max()-body[0,payload,2])
    result['soft_peak_lift_m']=float(x[:,:,2].mean(1).max()-x[0,:,2].mean())
    rigid_lift=result['rigid_peak_lift_m'] if release else result['rigid_final_lift_m']
    soft_lift=result['soft_peak_lift_m'] if release else result['soft_final_lift_m']
    result['both_objects_lifted']=bool(rigid_lift>.015 and soft_lift>.015 and
        hold.any() and result['rigid_minimum_hold_clearance_m']>.005 and result['soft_minimum_hold_clearance_m']>.005)
    result['both_objects_dropped']=bool(result['rigid_peak_lift_m']-result['rigid_final_lift_m']>.015 and
                                     result['soft_peak_lift_m']-result['soft_final_lift_m']>.015)
    result['accepted']=bool(result['both_objects_lifted'] and result['finite'] and result['full_timeline'] and
        result['minimum_tet_ratio']>.2 and not any(result['native_solver_calls'].values()))
    if release:result['accepted']&=result['both_objects_dropped']
    if report.get('style') in ('front','side'):
        result['rigid_draw_back_m']=float(body[0,payload,0]-body[-1,payload,0])
        result['soft_draw_back_m']=float(x[0,:,0].mean()-x[-1,:,0].mean())
        result['accepted']&=result['rigid_draw_back_m']>.04 and result['soft_draw_back_m']>.04
    return result
