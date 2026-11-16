"""Generate actual pinned Newton reference trajectories; never imported by inference."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .adapters import export_static


ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()






def make_scene(family, index, config, newton, wp):
    rng = np.random.default_rng(config["seed"] + 1000*config["families"].index(family) + index)
    gravity = np.array([0.0, 0.0, -rng.uniform(3.5, 6.0)], dtype=np.float32)
    builder = newton.ModelBuilder(gravity=tuple(gravity))
    params = {}
    if family == "softbody":
        mu = float(rng.uniform(800, 1800)*config.get("softbody_stiffness_scale", 1))
        params = dict(mu=mu, lmbd=1.5*mu, density=float(rng.uniform(600, 1000)))
        builder.add_soft_grid(pos=wp.vec3(0, 0, 2), rot=wp.quat_identity(), vel=wp.vec3(0), dim_x=2, dim_y=1, dim_z=1, cell_x=0.12, cell_y=0.12, cell_z=0.12, density=params["density"], k_mu=mu, k_lambda=1.5*mu, k_damp=0, fix_left=True, particle_radius=0.005)
    elif family == "cloth":
        mu = float(rng.uniform(60, 160))
        params = dict(mu=mu, lmbd=1.5*mu, mass=float(rng.uniform(0.02, 0.04)), bending=0.04)
        builder.add_cloth_grid(pos=wp.vec3(0, 0, 2), rot=wp.quat_identity(), vel=wp.vec3(0), dim_x=3, dim_y=3, cell_x=0.1, cell_y=0.1, mass=params["mass"], fix_left=True, tri_ke=mu, tri_ka=1.5*mu, tri_kd=0, edge_ke=params["bending"], edge_kd=0, particle_radius=0.005)
    elif family == "cloth_slide":
        from .assets import load_asset_spec, build_asset
        spec = load_asset_spec(ROOT/"assets/live/cloth_slide.json")
        spec["material"].update(mu=float(rng.uniform(60, 160)), lmbd=float(rng.uniform(90, 240)), mass=float(rng.uniform(.02, .04)), bending=float(rng.uniform(.02, .07)))
        friction = [0, .2, .6, .1, .4, .8, .35, .5][index]
        spec["plane"].update(particle_mu=friction, ground_mu=friction)
        spec.update(gravity=gravity.tolist(), velocity=[float(rng.uniform(.35, .7)),0,0])
        model, _, _ = build_asset(spec, config["device"], newton, wp)
        params = {**spec["material"], "plane": spec["plane"], "asset_spec": spec}
        pipeline = newton.CollisionPipeline(model, deterministic=True, soft_contact_gap=.02)
        solver = newton.solvers.SolverVBD(model, iterations=config["teacher_iterations"], friction_epsilon=spec["plane"]["friction_epsilon"], particle_enable_self_contact=False, rigid_soft_enable_dat=False, collision_pipeline=pipeline, deterministic=wp.DeterministicMode.RUN_TO_RUN)
        return model, solver, params, gravity
    elif family == "joint":
        params = dict(linear_ke=2e4, angular_ke=2e3, mass=float(rng.uniform(0.3, 0.6)))
        root = builder.add_link(xform=wp.transform((0, 0, 2), wp.quat_identity()))
        builder.add_shape_box(root, hx=0.02, hy=0.02, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0))
        child = builder.add_link(xform=wp.transform((0.12, 0, 2), wp.quat_identity()), mass=params["mass"], inertia=wp.mat33(0.0002, 0, 0, 0, 0.003, 0, 0, 0, 0.003), lock_inertia=True)
        builder.add_shape_box(child, hx=0.12, hy=0.025, hz=0.025, cfg=newton.ModelBuilder.ShapeConfig(density=0))
        fixed = builder.add_joint_fixed(parent=-1, child=root, parent_xform=wp.transform((0, 0, 2), wp.quat_identity()))
        hinge = builder.add_joint_revolute(parent=root, child=child, axis=(0, 1, 0), child_xform=wp.transform((-0.12, 0, 0), wp.quat_identity()))
        builder.add_articulation([fixed, hinge])
    elif family == "rod":
        bend = float(rng.uniform(0.4, 1.2))
        params = dict(stretch=2e3, shear=1.4e3, bending=bend, twist=0.7*bend)
        rod = newton.Rod.create_straight(start=wp.vec3(0, 0, 2), direction=wp.vec3(1, 0, 0), length=0.4, segment_count=4, radius=0.015)
        bodies, _ = builder.add_rod(rod=rod, stretch_stiffness=params["stretch"], shear_stiffness=params["shear"], bend_stiffness=bend, twist_stiffness=0.7*bend)
        root = bodies[0]
        builder.body_mass[root] = builder.body_inv_mass[root] = 0
        builder.body_inertia[root] = builder.body_inv_inertia[root] = wp.mat33(0.0)
    else:
        raise ValueError(family)
    builder.color()
    model = builder.finalize(device=config["device"])
    solver = newton.solvers.SolverVBD(model, iterations=config["teacher_iterations"], particle_enable_self_contact=False, deterministic=wp.DeterministicMode.RUN_TO_RUN, rigid_joint_linear_ke=params.get("linear_ke", 2e4), rigid_joint_angular_ke=params.get("angular_ke", 2e3))
    return model, solver, params, gravity




def simulate(model, solver, family, static, config, wp, steps=None):
    node = family.split("_")[0] in ("softbody", "cloth")
    a, b = model.state(), model.state()
    control = model.control()
    def reset_numeric():
        for key in ("joint_lambda_lin", "joint_lambda_ang", "joint_drive_lambda", "joint_limit_lambda"):
            value = getattr(solver, key, None)
            if value is not None:
                value.zero_()
    def advance():
        a.clear_forces()
        reset_numeric()
        solver.step(a, b, control, None, config["dt"])
        if node:
            wp.copy(a.particle_q, b.particle_q); wp.copy(a.particle_qd, b.particle_qd)
        else:
            wp.copy(a.body_q, b.body_q); wp.copy(a.body_qd, b.body_qd)
    # Capture a fixed physical substep, including state and cold numerical-state updates.
    with wp.ScopedCapture(device=config["device"]) as capture:
        advance()
    def state():
        if node:
            x, v = a.particle_q.numpy(), a.particle_qd.numpy()
            q = static["rest_q"].copy(); omega = np.zeros_like(v)
        else:
            poses, twist = a.body_q.numpy(), a.body_qd.numpy()
            x, q, v, omega = poses[:, :3], poses[:, 3:], twist[:, :3], twist[:, 3:]
            if len(static["rest_x"]) > len(x):
                x=np.r_[x, [[0,0,0]]]; q=np.r_[q, [[0,0,0,1]]]; v=np.r_[v, [[0,0,0]]]; omega=np.r_[omega, [[0,0,0]]]
        return [np.asarray(item, dtype=np.float32).copy() for item in (x, q, v, omega)]
    states = [state()]
    angle_history = []
    for _ in range(steps or config["steps"]):
        wp.capture_launch(capture.graph)
        values = state()
        if not all(np.isfinite(value).all() for value in values):
            raise RuntimeError("Teacher produced non-finite state")
        states.append(values)
        angle = getattr(solver, "joint_angle_prev", None)
        angle_history.append(angle.numpy().copy() if angle is not None else np.empty(0))
    result = {key: np.stack([row[i] for row in states]) for i, key in enumerate(("x", "q", "v", "omega"))}
    result["teacher_joint_angle_history"] = np.asarray(angle_history, dtype=np.float32)
    return result


def generate(config, *, only=None):
    start = time.perf_counter()
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config["newton_root"], text=True).strip()
    if actual != config["newton_sha"]:
        raise RuntimeError("Reference source pin differs")
    sys.path.insert(0, config["newton_root"])
    import warp as wp
    import newton
    wp.config.kernel_cache_dir = str(ROOT / ".local/p0/warp-cache")
    wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
    wp.init(); wp.set_device(config["device"])
    directory = ROOT / config["data_dir"]; directory.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT / config["report_dir"]; report_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"status": "RUNNING", "config": config, "source_sha": actual, "generator_sha256": digest(__file__), "adapter_sha256": digest(Path(__file__).with_name("adapters.py")), "teacher": "SolverVBD offline only; contacts only in explicitly marked plane scenes; zero material damping; per-step numerical multipliers cold-started", "scenes": [], "convergence": []}
    for family in ([only] if only else config["families"]):
        for index in range(config["scenes_per_family"]):
            scene_start = time.perf_counter()
            model, solver, params, gravity = make_scene(family, index, config, newton, wp)
            static = export_static(family.split("_")[0], model, params, gravity, newton)
            if family == "cloth_slide":
                plane = params["plane"]
                static.update(plane_normal=np.asarray([0,0,1], np.float32), plane_offset=np.asarray(plane["height"], np.float32), contact_radius=np.asarray(plane["activation_radius"], np.float32), contact_material=np.asarray([plane["ke"], plane["kd"], np.sqrt(plane["particle_mu"]*plane["ground_mu"]), plane["friction_epsilon"]], np.float32))
            trajectory = simulate(model, solver, family, static, config, wp)
            quality = {}
            if len(static.get("tets", [])):
                points = trajectory["x"][:, static["tets"]]
                deformation = (points[:, :, 1:]-points[:, :, :1]).swapaxes(-1,-2) @ static["tet_Dm"]
                quality["minimum_detF"] = float(np.linalg.det(deformation).min())
                if quality["minimum_detF"] < .05:
                    manifest.update(status="FAIL", failure={"family":family, "index":index, **quality})
                    (report_dir/"data_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                    raise RuntimeError(f"Teacher trajectory violates tetrahedron contract: {quality}")
            split = next(key for key, indices in config["splits"].items() if index in indices)
            metadata = {"family": family, "index": index, "split": split, "params": params, "dt": config["dt"], "source_sha": actual, "contacts": "plane" if family == "cloth_slide" else False}
            if family == "cloth_slide" and float(trajectory["x"][...,2].min()) < float(static["plane_offset"]):
                raise RuntimeError("Teacher crossed physical plane; retain failure and fix data contract")
            path = directory / f"{family}_{index:02d}.npz"
            np.savez_compressed(path, **static, **trajectory, metadata=np.array(json.dumps(metadata)))
            entry = {**metadata, **quality, "path": str(path.relative_to(ROOT)), "sha256": digest(path), "bytes": path.stat().st_size, "nodes": len(static["rest_x"]), "seconds": time.perf_counter()-scene_start}
            manifest["scenes"].append(entry)
            if index == 0:
                cfg = {**config, "teacher_iterations": config["convergence_iterations"]}
                m2, s2, _, _ = make_scene(family, index, cfg, newton, wp)
                refined = simulate(m2, s2, family, static, cfg, wp)
                error = float(np.max(np.linalg.norm(trajectory["x"]-refined["x"], axis=-1)) / float(static["reference_length"]))
                record = {"family": family, "iterations": [config["teacher_iterations"], config["convergence_iterations"]], "max_position_error_over_L": error, "status": "PASS" if error <= config["convergence_max_position_error_over_L"] else "FAIL"}
                manifest["convergence"].append(record)
                if record["status"] != "PASS":
                    manifest["status"] = "FAIL"
                    (report_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
                    raise RuntimeError(f"Pilot teacher iteration check failed: {record}")
            (report_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
            print(f"{family}/{index} [{split}] N={entry['nodes']} saved ({entry['seconds']:.2f}s)", flush=True)
    manifest["status"] = "PASS"
    manifest["wall_seconds"] = time.perf_counter()-start
    (report_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n")
    return manifest
