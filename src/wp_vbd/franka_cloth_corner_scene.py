"""Dynamic FR3 pinches one free cloth corner outside the table edge.

The default gripper approaches vertically from above. Motor targets lower,
close the pinch and lift; no cloth pins, attachments or pose overrides.
"""
from types import SimpleNamespace
import numpy as np
from scipy.spatial.transform import Rotation

TABLE_TOP = .20
TABLE_BOUNDS = (-.32, .28, -.55, -.15)
CLOTH_ORIGIN = np.array([-.10, -.60, TABLE_TOP + .002])
CLOTH_SIZE = .18
DURATION = 2.8
TOOL_OFFSET = np.array([0., 0., .22])
OPENING = .0015
CLOSED = .0003


def layout(grasp_style='overhead'):
    if grasp_style=='side':
        return dict(origin=CLOTH_ORIGIN.copy(),duration=2.8,opening=OPENING)
    if grasp_style!='overhead':raise ValueError('Unknown cloth grasp style')
    return dict(origin=np.array([-.10,-.575,TABLE_TOP+.002]),duration=DURATION,opening=.0015)


def command(time,grasp_style='overhead'):
    def ramp(a,b):
        u=float(np.clip((time-a)/(b-a),0.,1.))
        return u*u*(3.-2.*u)
    if grasp_style=='side':
        lift=ramp(.35,2.10)
        point=CLOTH_ORIGIN+np.array([0.,.020-.040*lift,.300*lift])
        opening=OPENING+(CLOSED-OPENING)*ramp(0.,.25)
        phase='pinch corner' if time<.35 else 'lift one corner' if time<2.10 else 'hold / free cloth drape'
        return point,opening,phase
    origin=layout(grasp_style)['origin']
    descent=ramp(0.,.20);lift=ramp(.30,2.0)
    point=origin+np.array([CLOTH_SIZE/8,-.025*lift,.012-.004*descent+.300*lift])
    opening=.0015+(CLOSED-.0015)*ramp(0.,.20)
    phase='vertical pregrasp / close fingers' if time<.30 else 'lift one corner vertically' if time<2.0 else 'hold / free cloth drape'
    return point,opening,phase


def build(wp,newton,*,device='cuda:0',substeps=3,initialize_controls=True,
          friction=2.,membrane=300.,bending=.0015,grasp_style='overhead',fold_profile='corner'):
    from newton.examples.softbody.example_softbody_franka import Example
    b=newton.ModelBuilder(gravity=(0.,0.,-9.81))
    helper=Example.__new__(Example)
    helper.create_articulation(b)
    b.joint_target_ke[:]=[20000.]*7+[2000.]*2
    b.joint_target_kd[:]=[100.]*7+[20.]*2
    b.joint_target_mode[:]=[int(newton.JointTargetMode.POSITION)]*9
    settings=layout(grasp_style);origin=settings['origin']
    b.joint_q[-2:]=[settings['opening']]*2
    b.default_shape_cfg.ke=1e4;b.default_shape_cfg.kd=10.;b.default_shape_cfg.mu=friction
    b.add_shape_box(-1,xform=wp.transform((-.02,-.35,.10),wp.quat_identity()),
        hx=.30,hy=.20,hz=.10,color=(.38,.45,.53),label='table')
    if grasp_style=='side':
        b.add_cloth_grid(pos=wp.vec3(*origin),rot=wp.quat_identity(),vel=wp.vec3(0.),
            dim_x=8,dim_y=8,cell_x=CLOTH_SIZE/8,cell_y=CLOTH_SIZE/8,mass=.00015,
            fix_left=False,fix_right=False,fix_top=False,fix_bottom=False,
            tri_ke=membrane,tri_ka=membrane,tri_kd=.03,edge_ke=bending,edge_kd=.0002,
            particle_radius=.0015,color=(.15,.62,.90),label='free_square_cloth')
    else:
        # A local free corner fold supplies the pinch, leaving the broad
        # panel uncreased. 'edge' preserves the earlier diagnostic asset.
        if fold_profile not in ('corner','edge'):raise ValueError('Unknown cloth fold profile')
        vertices=[];faces=[];cell=CLOTH_SIZE/8
        for i in range(9):
            for j in range(9):
                angle=np.pi/2*(1. if fold_profile=='edge' else max(0.,1.-j/3))
                vertices.append((cell-cell*np.cos(angle),j*cell,cell*np.sin(angle))
                                if i==0 else (i*cell,j*cell,0.))
        for i in range(8):
            for j in range(8):
                a=i*9+j;b0=(i+1)*9+j;c=(i+1)*9+j+1;d=i*9+j+1
                faces.extend((a,b0,c,a,c,d))
        b.add_cloth_mesh(pos=wp.vec3(*origin),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(0.),
            vertices=vertices,indices=faces,density=.375,
            tri_ke=membrane,tri_ka=membrane,tri_kd=.03,edge_ke=bending,edge_kd=.0002,
            particle_radius=.0015)
    b.add_ground_plane(height=-.12)
    b.color()
    model=b.finalize(device=device,requires_grad=False)
    model.soft_contact_ke=1e4;model.soft_contact_kd=10.;model.soft_contact_mu=friction
    model.shape_material_ke.fill_(1e4);model.shape_material_kd.fill_(10.);model.shape_material_mu.fill_(friction)
    points=model.particle_q.numpy()
    grasp_corner=origin+(np.array([CLOTH_SIZE/8,0.,CLOTH_SIZE/8]) if grasp_style=='overhead' else 0.)
    corner_id=int(np.argmin(np.linalg.norm(points-grasp_corner,axis=-1)))
    patch_ids=np.flatnonzero(np.linalg.norm(points-grasp_corner,axis=-1)<.035)
    common=dict(model=model,corner_id=corner_id,patch_ids=patch_ids,
        grasp_style=grasp_style,duration=settings['duration'],
        sim_substeps=substeps,sim_dt=1/(60*substeps),frame_dt=1/60)
    if not initialize_controls:
        return SimpleNamespace(**common,state_0=model.state(),state_1=model.state())
    helper.model=model
    point,opening,_=command(0.,grasp_style)
    # Cancel the imported hand's -pi/4 roll: approach along +Y and close
    # vertically, leaving the lower pad outside the table footprint.
    if grasp_style=='side':
        rotation=(Rotation.from_euler('x',-np.pi/2)*Rotation.from_euler('z',np.pi/4)).as_quat()
    else:
        # Tool +Z points down; the two fingers close along world X, both
        # outside the front table edge while surrounding the free corner.
        rotation=(Rotation.from_euler('x',np.pi)*Rotation.from_euler('z',-np.pi/4)).as_quat()
    helper.targets=np.array([[*point,*rotation,1.]],dtype=np.float32)
    helper.set_up_ik()
    helper.ik_joint_q=wp.clone(helper.ik_joint_q)
    helper.ik_solver.step(helper.ik_joint_q,helper.ik_joint_q,iterations=160)
    q=wp.to_torch(helper.ik_joint_q)[0];q[-2:]=opening
    wp.to_torch(model.joint_q).copy_(q)
    state=model.state();newton.eval_fk(model,model.joint_q,model.joint_qd,state)
    wp.copy(model.body_q,state.body_q)
    model.joint_target_q.assign(model.joint_q)
    return SimpleNamespace(**common,state_0=state,state_1=model.state(),control=model.control(),helper=helper)


def controller(example,time,wp):
    point,opening,phase=command(time,example.grasp_style)
    helper=example.helper
    helper.pos_obj.set_target_position(0,wp.vec3(*point))
    helper.ik_solver.step(helper.ik_joint_q,helper.ik_joint_q,iterations=24)
    q=wp.to_torch(helper.ik_joint_q)[0];q[-2:]=opening
    wp.to_torch(example.control.joint_target_q).copy_(q)
    return phase
