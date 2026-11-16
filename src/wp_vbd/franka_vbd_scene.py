"""A single-solver dynamic Franka/elastic grasp, with IK used only for targets.

This authored test uses the official FR3 URDF, its inertia and mimic relation.
The low resolution block is intentionally a new asset, not the official duck.
"""
from types import SimpleNamespace

import numpy as np


def command(time, *, grip_open=.04, grip_closed=.02, grasp_height=.21):
    descent = float(np.clip((time-.3)/1., 0., 1.))
    closure = float(np.clip((time-1.3)/.8, 0., 1.))
    lift = float(np.clip((time-2.1)/1.2, 0., 1.))
    phase = 'approach' if time<.3 else 'descend' if time<1.3 else 'pinch' if time<2.1 else 'lift' if time<3.3 else 'hold'
    return .30+(grasp_height-.30)*descent+.12*lift, grip_open+(grip_closed-grip_open)*closure, phase


def build(wp, newton, *, device='cuda:0', soft_asset='block', substeps=10, initialize_controls=True, start_time=0.,
          material_mu=None, material_lambda=None, arm_drive_ke=2000.,
          grasp_x=-.005, grasp_yaw=0., grip_open=.04, grip_closed=.02, friction=None, grasp_height=.21):
    from newton.examples.softbody.example_softbody_franka import Example
    b = newton.ModelBuilder(gravity=(0., 0., -9.81))
    helper = Example.__new__(Example)
    helper.create_articulation(b)
    # The source example is kinematically driven. This scene authors finite
    # dynamic drives, retaining every imported body inertia and mimic joint.
    b.joint_target_ke[:] = [arm_drive_ke]*7+[2000.]*2
    b.joint_target_kd[:] = [100.]*7+[20.]*2
    b.joint_target_mode[:] = [int(newton.JointTargetMode.POSITION)]*9
    b.joint_q[-2:] = [grip_open, grip_open]
    b.default_shape_cfg.ke = 1e4; b.default_shape_cfg.kd = 10.; b.default_shape_cfg.mu = 1.5
    b.add_shape_box(-1, xform=wp.transform((0., -.5, .1), wp.quat_identity()), hx=.4, hy=.4, hz=.1,
                    color=(.35, .42, .50))
    if soft_asset == 'block':
        b.add_soft_grid(pos=wp.vec3(-.025, -.525, .203), rot=wp.quat_identity(), vel=wp.vec3(0.),
                        dim_x=4, dim_y=4, dim_z=4, cell_x=.0125, cell_y=.0125, cell_z=.0125,
                        density=400., k_mu=5000. if material_mu is None else material_mu,
                        k_lambda=5000. if material_lambda is None else material_lambda,
                        k_damp=1., particle_radius=.003,
                        color=(.95, .65, .12), label='elastic_block')
    elif soft_asset == 'duck':
        from pxr import Usd
        stage = Usd.Stage.Open(str(newton.utils.download_asset('manipulation_objects/rubber_duck')/'model.usda'))
        mesh = newton.TetMesh.create_from_usd(stage.GetPrimAtPath('/root/Model/TetMesh'), compat_namespaces=())
        b.add_soft_mesh(pos=wp.vec3(0., -.5, .23), rot=wp.quat_identity(), scale=1., vel=wp.vec3(0.),
                        mesh=mesh, density=100., k_mu=1e6 if material_mu is None else material_mu,
                        k_lambda=1e6 if material_lambda is None else material_lambda, k_damp=1., particle_radius=.005)
    else: raise ValueError(soft_asset)
    b.add_ground_plane(height=-.12)
    b.color()
    model = b.finalize(device=device, requires_grad=False)
    model.soft_contact_ke = 1e4 if soft_asset == 'block' else 2e6
    model.soft_contact_kd = 10. if soft_asset == 'block' else .2
    model.soft_contact_mu = 1.5 if soft_asset == 'block' else .5
    model.shape_material_ke.fill_(model.soft_contact_ke)
    model.shape_material_kd.fill_(model.soft_contact_kd)
    model.shape_material_mu.fill_(1.5)
    if friction is not None:
        model.soft_contact_mu=friction;model.shape_material_mu.fill_(friction)
    helper.model = model
    if not initialize_controls:
        return SimpleNamespace(model=model, state_0=model.state(), state_1=model.state(), soft_asset=soft_asset)
    # Same tool convention as the official scene. Only target coordinates
    # from this scratch IK state enter controls after initialization.
    z, opening, _ = command(start_time,grip_open=grip_open,grip_closed=grip_closed,grasp_height=grasp_height)
    # Rz(yaw) * Rx(pi), using Newton/Warp's XYZW convention.
    rotation=wp.quat(float(np.cos(.5*grasp_yaw)),float(np.sin(.5*grasp_yaw)),0.,0.)
    helper.targets = np.array([[grasp_x, -.5, z, *list(rotation), 1.]], dtype=np.float32)
    helper.set_up_ik()
    # wp.array(existing_array, shape=...) in the official IK helper is a
    # view. Dynamics requires model.joint_q to remain the immutable rest
    # reference; target planning must own a separate coordinate buffer.
    helper.ik_joint_q = wp.clone(helper.ik_joint_q)
    helper.ik_solver.step(helper.ik_joint_q, helper.ik_joint_q, iterations=96)
    target = wp.to_torch(helper.ik_joint_q)[0]
    target[-2:] = opening
    wp.to_torch(model.joint_q).copy_(target)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    wp.copy(model.body_q, state.body_q)
    model.joint_target_q.assign(model.joint_q)
    # Initial FK is complete. Actual states are advanced by the chosen
    # physics solver; controller() never writes body_q or particle_q.
    return SimpleNamespace(model=model, state_0=state, state_1=model.state(), control=model.control(),
                           helper=helper, sim_substeps=substeps, sim_dt=1/(60*substeps), frame_dt=1/60,
                           grasp_x=grasp_x,grip_open=grip_open,grip_closed=grip_closed,grasp_height=grasp_height,
                           sim_time=0., soft_asset=soft_asset)


def controller(example, time, wp):
    # 0-.3 hold; .3-1.3 descend; 1.3-2.1 pinch;
    # 2.1-3.3 lift; 3.3-4 hold. Commands do not prescribe physical poses.
    z, opening, phase = command(time,grip_open=example.grip_open,grip_closed=example.grip_closed,grasp_height=example.grasp_height)
    e = example.helper
    e.pos_obj.set_target_position(0, wp.vec3(example.grasp_x, -.5, z))
    e.ik_solver.step(e.ik_joint_q, e.ik_joint_q, iterations=24)
    q = wp.to_torch(e.ik_joint_q)[0]
    q[-2:] = opening
    wp.to_torch(example.control.joint_target_q).copy_(q)
    return phase
