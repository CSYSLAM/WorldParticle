"""Dynamic FR3 pick/place: a free rigid box is released onto a pinned cloth.

IK plans arm drive targets only. The payload has its own FREE joint and
never receives a pose, velocity, grasp attachment, or scripted force.
"""
from types import SimpleNamespace
import numpy as np

DURATION = 6.4
PICK_X = -.12
PLACE_X = .14
CLOTH_HEIGHT = .235


def command(time, *, grip_open=.04, grip_closed=.0225, grasp_height=.21, place_height=.29, **unused):
    """Continuous finite-speed drive targets, including an explicit release."""
    def ramp(begin, end):
        u = float(np.clip((time-begin)/(end-begin), 0., 1.))
        return u*u*(3.-2.*u)
    x = PICK_X+(PLACE_X-PICK_X)*ramp(2.3, 3.5)
    z = .32+(grasp_height-.32)*ramp(.2, .95)
    z += (.335-grasp_height)*ramp(1.6, 2.3)
    z += (place_height-.335)*ramp(3.5, 4.3)
    z += (.36-place_height)*ramp(5., 5.8)
    opening = grip_open+(grip_closed-grip_open)*ramp(.95, 1.6)
    opening += (grip_open-grip_closed)*ramp(4.3, 5.)
    phase = next((name for end, name in ((.2,'approach'),(.95,'descend'),(1.6,'grasp'),
                 (2.3,'lift'),(3.5,'transfer'),(4.3,'lower'),(5.,'release'),(5.8,'retreat'))
                  if time < end), 'settle')
    return np.array([x, -.5, z]), opening, phase


def build(wp, newton, *, device='cuda:0', substeps=10, initialize_controls=True,
          start_time=0., arm_drive_ke=20000., grip_open=.04, grip_closed=.0225,
          grasp_height=.21, place_height=.29, friction=None, **unused):
    from newton.examples.softbody.example_softbody_franka import Example
    b = newton.ModelBuilder(gravity=(0., 0., -9.81))
    helper = Example.__new__(Example)
    helper.create_articulation(b)
    b.joint_target_ke[:] = [arm_drive_ke]*7+[2000.]*2
    b.joint_target_kd[:] = [100.]*7+[20.]*2
    b.joint_target_mode[:] = [int(newton.JointTargetMode.POSITION)]*9
    b.joint_q[7:9] = [grip_open]*2
    b.default_shape_cfg.ke = 1e4
    b.default_shape_cfg.kd = 10.
    b.default_shape_cfg.mu = 1.5 if friction is None else friction
    b.add_shape_box(-1, xform=wp.transform((0., -.5, .08), wp.quat_identity()),
                    hx=.4, hy=.30, hz=.08, color=(.29,.35,.43), label='workbench')
    b.add_shape_box(-1, xform=wp.transform((PICK_X, -.5, .18), wp.quat_identity()),
                    hx=.065, hy=.065, hz=.02, color=(.48,.53,.60), label='pickup_platform')
    # A visible frame supports a cloth over a 7.5 cm recess. Boundary pins
    # represent clamping to the frame; the interior remains fully dynamic.
    for x, y, hx, hy in ((PLACE_X-.127,-.5,.007,.127),(PLACE_X+.127,-.5,.007,.127),
                         (PLACE_X,-.627,.12,.007),(PLACE_X,-.373,.12,.007)):
        b.add_shape_box(-1, xform=wp.transform((x,y,.194),wp.quat_identity()),
                        hx=hx,hy=hy,hz=.039,color=(.20,.27,.34),label='cloth_frame')
    b.add_cloth_grid(pos=wp.vec3(PLACE_X-.12,-.62,CLOTH_HEIGHT),rot=wp.quat_identity(),
                     vel=wp.vec3(0.), dim_x=8,dim_y=8,cell_x=.03,cell_y=.03,mass=.0008,
                     fix_left=True,fix_right=True,fix_top=True,fix_bottom=True,
                     tri_ke=300.,tri_ka=300.,tri_kd=.1,edge_ke=.003,edge_kd=.001,
                     particle_radius=.002,color=(.12,.55,.86),label='receiving_cloth')
    payload = b.add_body(xform=wp.transform((PICK_X,-.5,.2255),wp.quat_identity()),label='free_payload')
    cfg = b.default_shape_cfg.copy()
    cfg.density = 400.
    b.add_shape_box(payload,hx=.025,hy=.025,hz=.025,cfg=cfg,color=(1.,.60,.10),label='rigid_payload')
    b.add_ground_plane(height=-.12)
    b.color()
    model = b.finalize(device=device,requires_grad=False)
    model.soft_contact_ke=1e4;model.soft_contact_kd=10.
    model.soft_contact_mu=1.5 if friction is None else friction
    model.shape_material_ke.fill_(1e4);model.shape_material_kd.fill_(10.)
    model.shape_material_mu.fill_(model.soft_contact_mu)
    common=dict(model=model,payload_id=payload,soft_asset='rigid_cloth',
                grip_open=grip_open,grip_closed=grip_closed,grasp_height=grasp_height,place_height=place_height,
                sim_substeps=substeps,sim_dt=1/(60*substeps),frame_dt=1/60,sim_time=0.)
    if not initialize_controls:
        return SimpleNamespace(**common,state_0=model.state(),state_1=model.state())
    helper.model=model
    target,opening,_=command(start_time,grip_open=grip_open,grip_closed=grip_closed,grasp_height=grasp_height,place_height=place_height)
    helper.targets=np.array([[*target,1.,0.,0.,0.,1.]],dtype=np.float32)
    helper.set_up_ik()
    helper.ik_joint_q=wp.clone(helper.ik_joint_q)
    helper.ik_solver.step(helper.ik_joint_q,helper.ik_joint_q,iterations=96)
    q=wp.to_torch(helper.ik_joint_q)[0]
    q[7:9]=opening
    # Independent initial FK, once. FREE payload coordinates are untouched.
    wp.to_torch(model.joint_q)[:9].copy_(q[:9])
    state=model.state();newton.eval_fk(model,model.joint_q,model.joint_qd,state)
    wp.copy(model.body_q,state.body_q)
    wp.to_torch(model.joint_target_q)[:9].copy_(q[:9])
    return SimpleNamespace(**common,state_0=state,state_1=model.state(),control=model.control(),helper=helper)


def controller(example,time,wp):
    target,opening,phase=command(time,grip_open=example.grip_open,grip_closed=example.grip_closed,
                                 grasp_height=example.grasp_height,place_height=example.place_height)
    e=example.helper
    e.pos_obj.set_target_position(0,wp.vec3(*target))
    e.ik_solver.step(e.ik_joint_q,e.ik_joint_q,iterations=24)
    q=wp.to_torch(e.ik_joint_q)[0];q[7:9]=opening
    wp.to_torch(example.control.joint_target_q)[:9].copy_(q[:9])
    return phase
