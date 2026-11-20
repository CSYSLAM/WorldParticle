"""Dynamic Piper picks a tetrahedral block and releases it onto a free rigid cube.

URDF kinematics plan motor targets only. Imported inertias, collision shapes,
visuals and the negative-ratio finger mimic are retained; no payload pins.
"""
from pathlib import Path
from types import SimpleNamespace
import xml.etree.ElementTree as ET

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .w1_dual_grasp_scene import transform

TABLE_TOP = .15
TABLE_BOUNDS = (.23, .67, -.36, .36)
PICK = np.array([.40, -.10, TABLE_TOP + .028])
STACK = np.array([.40, .10, TABLE_TOP + .031])
CUBE_HALF = .03
TOOL_OFFSET = np.array([0., 0., .108])
DURATION = 5.2
OPENING = .034
CLOSED = .022


def schedule():
    high = .32
    place = STACK + np.array([0., 0., CUBE_HALF + .028 + .004])
    return [(0., PICK + [0., 0., .075], OPENING),
            (.25, PICK + [0., 0., .075], OPENING),
            (.9, PICK, OPENING), (1.4, PICK, CLOSED),
            (2.1, np.array([PICK[0], PICK[1], high]), CLOSED),
            (2.9, np.array([STACK[0], STACK[1], high]), CLOSED),
            (3.6, place, CLOSED), (4., place, OPENING),
            (4.5, np.array([STACK[0], STACK[1], high]), OPENING),
            (DURATION, np.array([STACK[0], STACK[1], high]), OPENING)]


def phase(time):
    return next((name for end, name in [(.25, 'approach'), (.9, 'lower to soft block'),
        (1.4, 'close fingers'), (2.1, 'lift soft block'), (2.9, 'transfer over rigid cube'),
        (3.6, 'lower to rigid cube'), (4., 'release soft block'),
        (4.5, 'withdraw gripper')] if time < end), 'settle on rigid cube')


def prepare(asset, output):
    source = ET.parse(asset).getroot()
    for mesh in source.findall('.//mesh'):
        path = (asset.parent / mesh.get('filename')).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        mesh.set('filename', str(path))
    output.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(source).write(output, encoding='unicode')
    return source


def plan(source):
    joints = [source.find(f"joint[@name='joint{i}']") for i in range(1, 7)]
    lower = np.array([float(j.find('limit').get('lower')) for j in joints]) + 1e-6
    upper = np.array([float(j.find('limit').get('upper')) for j in joints]) - 1e-6
    origins = [transform(j.find('origin')) for j in joints]
    axes = [np.fromstring(j.find('axis').get('xyz'), sep=' ') for j in joints]

    def fk(q):
        pose = np.eye(4)
        for origin, axis, angle in zip(origins, axes, q):
            move = np.eye(4)
            move[:3, :3] = Rotation.from_rotvec(axis * angle).as_matrix()
            pose = pose @ origin @ move
        return pose

    rotation = Rotation.from_euler('y', 2*np.pi/3).as_matrix()
    initial = np.array([0., 1.5, -1.5, 0., .5, 0.])
    points = schedule()
    # Cartesian intermediate samples keep the carried block's path level;
    # these are motor targets, never body or particle state assignments.
    dense = [points[0]]
    for a, b in zip(points[:-1], points[1:]):
        for u in np.linspace(0., 1., 5)[1:]:
            dense.append((a[0] + (b[0] - a[0]) * u,
                          a[1] + (b[1] - a[1]) * u,
                          a[2] + (b[2] - a[2]) * u))
    poses = []
    errors = []
    for index, (_, target, _) in enumerate(dense):
        def residual(q):
            pose = fk(q)
            point = pose[:3, 3] + pose[:3, :3] @ TOOL_OFFSET
            angle = Rotation.from_matrix(rotation @ pose[:3, :3].T).as_rotvec()
            return np.r_[point - target, .2 * angle, .00003 * (q - initial)]

        seeds = [initial] if index else [initial, [0., 2., -1.8, 1.5, .7, 1.5],
                                       [0., 1., -2., -1.5, -.7, -1.5]]
        fits = [least_squares(residual, np.clip(seed, lower, upper), bounds=(lower, upper),
                              max_nfev=250, ftol=1e-10, xtol=1e-10, gtol=1e-10) for seed in seeds]
        fit = min(fits, key=lambda result: np.linalg.norm(result.fun[:6]))
        pose = fk(fit.x)
        error = np.linalg.norm(pose[:3, 3] + pose[:3, :3] @ TOOL_OFFSET - target)
        angular = np.linalg.norm(Rotation.from_matrix(rotation @ pose[:3, :3].T).as_rotvec())
        if error > .001 or angular > .01:
            raise ValueError(f'Piper target {index} is unreachable: position {error:.5f} m, rotation {angular:.5f} rad')
        poses.append(fit.x)
        errors.append((error, angular))
        initial = fit.x
    return dense, np.asarray(poses), np.asarray(errors)


def controller(example, time, wp):
    points = example.waypoint_schedule
    index = min(max(np.searchsorted(example.waypoint_times, time, side='right') - 1, 0), len(points) - 2)
    a, b = points[index:index + 2]
    u = float(np.clip((time - a[0]) / (b[0] - a[0]), 0., 1.))
    # Dense samples are linear in Cartesian time; smooth whole intervals at
    # planning time instead of stopping at every intermediate target.
    q = example.angles[index] + u * (example.angles[index + 1] - example.angles[index])
    opening = a[2] + u * (b[2] - a[2])
    targets = wp.to_torch(example.control.joint_target_q)
    import torch
    targets[example.arm_targets] = torch.as_tensor(q, device=targets.device, dtype=targets.dtype)
    targets[example.finger_targets] = targets.new_tensor([opening, -opening])
    return phase(time)


def build(wp, newton, *, asset, output, device='cuda:0', substeps=3,
          initialize_controls=True, friction=2., material_mu=5000., material_lambda=5000.):
    asset = Path(asset).expanduser().resolve()
    if asset.is_dir():
        asset = asset / 'piper.urdf'
    prepared = Path(output) / 'piper_prepared.urdf'
    source = prepare(asset, prepared)
    b = newton.ModelBuilder(gravity=(0., 0., -9.81))
    b.add_urdf(str(prepared), floating=False, collapse_fixed_joints=True,
               enable_self_collisions=False, joint_damping_mode='passive')
    robot_bodies = b.body_count
    b.joint_target_ke[:] = [20000.] * 6 + [2000.] * 2
    b.joint_target_kd[:] = [100.] * 6 + [20.] * 2
    b.joint_target_mode[:] = [int(newton.JointTargetMode.POSITION)] * 8
    b.default_shape_cfg.ke = 1e4
    b.default_shape_cfg.kd = 10.
    b.default_shape_cfg.mu = friction
    b.add_shape_box(-1, xform=wp.transform((.45, 0., TABLE_TOP - .025), wp.quat_identity()),
                    hx=.22, hy=.36, hz=.025, color=(.42, .48, .55), label='table')
    for x in (.26, .64):
        for y in (-.31, .31):
            b.add_shape_box(-1, xform=wp.transform((x, y, .04), wp.quat_identity()),
                            hx=.012, hy=.012, hz=.04, color=(.23, .28, .33),
                            cfg=b.ShapeConfig(has_particle_collision=False))
    b.add_soft_grid(pos=wp.vec3(*(PICK - [.025, .025, .025])), rot=wp.quat_identity(),
                    vel=wp.vec3(0.), dim_x=4, dim_y=4, dim_z=4,
                    cell_x=.0125, cell_y=.0125, cell_z=.0125, density=400.,
                    k_mu=material_mu, k_lambda=material_lambda, k_damp=1., particle_radius=.003,
                    color=(.15, .65, .94), label='soft_block')
    payload = b.add_body(xform=wp.transform(tuple(STACK), wp.quat_identity()), label='free_rigid_support_cube')
    cfg = b.default_shape_cfg.copy()
    cfg.density = 1800.
    b.add_shape_box(payload, hx=CUBE_HALF, hy=CUBE_HALF, hz=CUBE_HALF,
                    cfg=cfg, color=(1., .55, .12), label='rigid_cube')
    b.add_ground_plane(height=-.003)
    b.color()
    model = b.finalize(device=device, requires_grad=False)
    # URDF inertial-frame rotations in the native Float32 importer leave
    # tiny antisymmetric roundoff (about 2e-8 relative for this asset).
    # Restore mathematical tensor symmetry at this asset boundary; reject
    # a meaningful asymmetry instead of weakening the solver's checks.
    inertia_roundoff = {}
    for name in ('body_inertia', 'body_inv_inertia'):
        tensor = getattr(model, name)
        value = tensor.numpy()
        transpose = np.swapaxes(value, -1, -2)
        scale = np.maximum(np.abs(value).max(axis=(-1, -2)), 1e-30)
        asymmetry = np.abs(value - transpose).max(axis=(-1, -2)) / scale
        if np.max(asymmetry) > 1e-6:
            raise ValueError('Imported Piper inertia has non-roundoff asymmetry: ' + name)
        symmetric = (value + transpose) * .5
        if np.any(np.linalg.eigvalsh(symmetric.astype(np.float64)) <= 0):
            raise ValueError('Imported Piper inertia must remain positive definite')
        tensor.assign(symmetric)
        inertia_roundoff[name] = float(asymmetry.max())
    model.soft_contact_ke = 1e4
    model.soft_contact_kd = 10.
    model.soft_contact_mu = friction
    model.shape_material_ke.fill_(1e4)
    model.shape_material_kd.fill_(10.)
    model.shape_material_mu.fill_(friction)
    common = dict(model=model, payload_id=payload, robot_bodies=robot_bodies,
                  sim_substeps=substeps, sim_dt=1/(60*substeps), frame_dt=1/60,
                  asset=str(asset), inertia_roundoff=inertia_roundoff)
    if not initialize_controls:
        return SimpleNamespace(**common, state_0=model.state(), state_1=model.state())
    labels = {label.rsplit('/', 1)[-1]: index for index, label in enumerate(model.joint_label)}
    starts = model.joint_q_start.numpy()
    targets = model.joint_target_q_start.numpy()
    points, angles, errors = plan(source)
    q = model.joint_q.numpy()
    arm = np.array([starts[labels[f'joint{i}']] for i in range(1, 7)])
    fingers = np.array([starts[labels[f'joint{i}']] for i in (7, 8)])
    q[arm] = angles[0]
    q[fingers] = [OPENING, -OPENING]
    model.joint_q.assign(q)
    state = model.state()
    newton.eval_fk(model, model.joint_q, model.joint_qd, state)
    wp.copy(model.body_q, state.body_q)
    model.joint_target_q.assign(q)
    tool_body = next(i for i, label in enumerate(model.body_label) if label.rsplit('/', 1)[-1] == 'link6')
    return SimpleNamespace(**common, state_0=state, state_1=model.state(), control=model.control(),
        waypoint_schedule=points, waypoint_times=np.array([p[0] for p in points]), angles=angles,
        ik_errors=errors, arm_coordinates=arm, finger_coordinates=fingers, tool_body=tool_body,
        arm_targets=[int(targets[labels[f'joint{i}']]) for i in range(1, 7)],
        finger_targets=[int(targets[labels[f'joint{i}']]) for i in (7, 8)])
