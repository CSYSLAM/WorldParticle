"""Build static assets and initial states without a teacher or trajectory file."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import torch

from .adapters import export_static
from .data import Scene, State


def read_obj(path):
    vertices, triangles = [], []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "v":
            vertices.append([float(value) for value in parts[1:4]])
        elif parts[0] == "f":
            ids = [int(value.split("/")[0]) for value in parts[1:]]
            ids = [value-1 if value > 0 else len(vertices)+value for value in ids]
            for i in range(1, len(ids)-1):
                triangles.append([ids[0], ids[i], ids[i+1]])
    points = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(triangles, dtype=np.int32)
    if points.ndim != 2 or points.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("OBJ requires vertices and polygon faces")
    if not np.isfinite(points).all() or faces.min() < 0 or faces.max() >= len(points):
        raise ValueError("Invalid OBJ geometry or indices")
    if np.min(np.linalg.norm(np.cross(points[faces[:, 1]]-points[faces[:, 0]], points[faces[:, 2]]-points[faces[:, 0]]), axis=1)) < 1e-10:
        raise ValueError("OBJ contains degenerate triangles")
    return points, faces


def load_asset_spec(path):
    path = Path(path).resolve()
    spec = json.loads(path.read_text())
    spec["_directory"] = str(path.parent)
    spec["_asset_file"] = str(path)
    if spec.get("contacts", False) not in (False, "plane"):
        raise ValueError("Supported contacts: false or plane")
    if spec.get("material", {}).get("damping", 0) != 0:
        raise ValueError("This checkpoint requires zero material damping")
    return spec


def build_asset(spec, device, newton, wp, *, builder_only=False):
    spec = deepcopy(spec)
    family, geometry, material = spec["family"], spec["geometry"], spec["material"]
    gravity = np.asarray(spec.get("gravity", [0, 0, -4.8]), dtype=np.float32)
    if gravity.shape != (3,) or not np.isfinite(gravity).all():
        raise ValueError("gravity must have three finite components")
    if material.get("damping", 0) != 0:
        raise ValueError("Nonzero material damping is not implemented")
    if "young_modulus" in material or "poisson_ratio" in material:
        if family != "softbody" or "mu" in material or "lmbd" in material:
            raise ValueError("E/nu conversion is for 3D softbody only; choose E/nu or mu/lmbd")
        young, poisson = material.pop("young_modulus"), material.pop("poisson_ratio")
        if not np.isfinite(young) or young <= 0 or not -1 < poisson < 0.5:
            raise ValueError("Require E > 0 and -1 < nu < 0.5")
        material.update(mu=young/(2*(1+poisson)), lmbd=young*poisson/((1+poisson)*(1-2*poisson)))
        if material["lmbd"] < 0:
            raise ValueError("This pilot supports nonnegative lambda (nu >= 0)")
    for key, value in material.items():
        if not np.isfinite(value) or value < 0:
            raise ValueError(f"Invalid material parameter {key}")
    builder = newton.ModelBuilder(gravity=tuple(gravity))
    position = wp.vec3(*spec.get("position", [0, 0, 2]))
    rotation = wp.quat(*spec.get("rotation_xyzw", [0, 0, 0, 1]))
    if abs(float(wp.length(rotation))-1) > 1e-5:
        raise ValueError("rotation_xyzw must be a unit quaternion")
    velocity = wp.vec3(*spec.get("velocity", [0, 0, 0]))
    params = dict(material)
    plane = spec.get("plane", {})
    radius = float(plane.get("activation_radius", 0.01)) if spec.get("contacts") == "plane" else 0.005
    if family in ("softbody", "cloth"):
        allowed = {"mu", "lmbd", "density", "mass", "bending", "damping", "areal_density"}
        if set(material)-allowed:
            raise ValueError(f"Unknown material keys: {set(material)-allowed}")
        if family == "cloth":
            kwargs = dict(pos=position, rot=rotation, vel=velocity, tri_ke=material["mu"], tri_ka=material["lmbd"], tri_kd=0, edge_ke=material.get("bending", 0.04), edge_kd=0, particle_radius=radius)
            if geometry["type"] == "obj":
                path = Path(spec["_directory"])/geometry["path"]
                vertices, indices = read_obj(path)
                if "mass" in material and "areal_density" in material:
                    raise ValueError("Specify mass per node or areal_density, not both")
                builder.add_cloth_mesh(**kwargs, vertices=[wp.vec3(*point) for point in vertices], indices=indices.flatten().tolist(), scale=float(geometry.get("scale", 1)), density=material.get("areal_density", 1.0))
                if "mass" in material:
                    builder.particle_mass[:] = [material["mass"]]*len(builder.particle_mass)
            elif geometry["type"] == "cloth_grid":
                nx, ny = geometry.get("resolution", [3, 3])
                sx, sy = geometry.get("size", [0.3, 0.3])
                builder.add_cloth_grid(**kwargs, dim_x=nx, dim_y=ny, cell_x=sx/nx, cell_y=sy/ny, mass=material["mass"])
            else:
                raise ValueError("Cloth geometry must be obj or cloth_grid")
        else:
            kwargs = dict(pos=position, rot=rotation, vel=velocity, density=material["density"], k_mu=material["mu"], k_lambda=material["lmbd"], k_damp=0, particle_radius=radius)
            if geometry["type"] == "tet_npz":
                with np.load(Path(spec["_directory"])/geometry["path"], allow_pickle=False) as mesh:
                    points, ids = mesh["vertices"], mesh["tets"]
                builder.add_soft_mesh(**kwargs, vertices=[wp.vec3(*point) for point in points], indices=ids.flatten().tolist(), scale=float(geometry.get("scale", 1)))
            elif geometry["type"] == "tet_grid":
                nx, ny, nz = geometry.get("resolution", [2, 1, 1])
                sx, sy, sz = geometry.get("size", [0.24, 0.12, 0.12])
                builder.add_soft_grid(**kwargs, dim_x=nx, dim_y=ny, dim_z=nz, cell_x=sx/nx, cell_y=sy/ny, cell_z=sz/nz)
            else:
                raise ValueError("Softbody geometry must be tet_npz or tet_grid")
        pins = spec.get("pins", {"axis": "x", "side": "min"})
        points = np.asarray([list(point) for point in builder.particle_q])
        if "indices" in pins:
            fixed = np.asarray(pins["indices"], dtype=int)
            if np.any(fixed < 0) or np.any(fixed >= len(points)):
                raise ValueError("Pin index outside asset")
        else:
            axis = "xyz".index(pins.get("axis", "x"))
            edge = points[:, axis].min() if pins.get("side", "min") == "min" else points[:, axis].max()
            fixed = np.where(np.abs(points[:, axis]-edge) < 1e-6)[0]
        for i in fixed:
            builder.particle_mass[int(i)] = 0.0
            builder.particle_qd[int(i)] = wp.vec3(0)
    elif family == "rod":
        if set(material)-{"stretch", "shear", "bending", "twist", "density", "damping"}:
            raise ValueError("Unknown rod material parameter")
        points = [position+wp.quat_rotate(rotation, wp.vec3(*point)) for point in geometry["points"]]
        rod = newton.Rod(points, radius=float(geometry.get("radius", 0.015)))
        rod.compute_frames()
        builder.default_shape_cfg.density = material.get("density", 1000)
        bodies, _ = builder.add_rod(rod=rod, stretch_stiffness=material["stretch"], shear_stiffness=material["shear"], bend_stiffness=material["bending"], twist_stiffness=material["twist"])
        for root in spec.get("fixed_segments", [0]):
            root = bodies[root]
            builder.body_mass[root] = builder.body_inv_mass[root] = 0
            builder.body_inertia[root] = builder.body_inv_inertia[root] = wp.mat33(0)
    elif family == "rigid":
        if set(material)-{"mass", "damping"}:
            raise ValueError("Unknown free rigid body material parameter")
        half = geometry.get("half_extents", [.08,.06,.05])
        inertia = np.diag(material["mass"]/3*np.array([half[1]**2+half[2]**2,half[0]**2+half[2]**2,half[0]**2+half[1]**2]))
        body = builder.add_body(xform=wp.transform(position,rotation),mass=material["mass"],inertia=wp.mat33(*inertia.flatten()),lock_inertia=True)
        builder.body_qd[body] = wp.spatial_vector(*list(velocity),0,0,0)
        builder.add_shape_box(body,hx=half[0],hy=half[1],hz=half[2],cfg=newton.ModelBuilder.ShapeConfig(density=0))
    elif family == "joint":
        if set(material)-{"linear_ke", "angular_ke", "mass", "damping"}:
            raise ValueError("Unknown joint material parameter")
        half = geometry.get("half_extents", [0.12, 0.025, 0.025])
        root = builder.add_link(xform=wp.transform(position, rotation))
        builder.add_shape_box(root, hx=0.02, hy=0.02, hz=0.02, cfg=newton.ModelBuilder.ShapeConfig(density=0))
        center = position+wp.quat_rotate(rotation, wp.vec3(half[0], 0, 0))
        inertia = np.diag(material["mass"]/3*np.array([half[1]**2+half[2]**2, half[0]**2+half[2]**2, half[0]**2+half[1]**2]))
        child = builder.add_link(xform=wp.transform(center, rotation), mass=material["mass"], inertia=wp.mat33(*inertia.flatten()), lock_inertia=True)
        builder.add_shape_box(child, hx=half[0], hy=half[1], hz=half[2], cfg=newton.ModelBuilder.ShapeConfig(density=0))
        fixed = builder.add_joint_fixed(parent=-1, child=root, parent_xform=wp.transform(position, rotation))
        hinge = builder.add_joint_revolute(parent=root, child=child, axis=wp.vec3(*geometry.get("axis", [0, 1, 0])), child_xform=wp.transform(wp.vec3(-half[0], 0, 0), wp.quat_identity()))
        builder.add_articulation([fixed, hinge])
    else:
        raise ValueError(f"Unsupported family {family}")
    if spec.get("contacts") == "plane":
        if family not in ("cloth", "softbody"):
            raise ValueError("Plane contact currently supports cloth and softbody particles")
        allowed = {"height", "activation_radius", "particle_mu", "ground_mu", "ke", "kd", "friction_epsilon"}
        if set(plane)-allowed or not all(np.isfinite(value) for value in plane.values()):
            raise ValueError("Invalid plane fields")
        if any(value < 0 for key,value in plane.items() if key != "height") or radius <= 0 or plane.get("friction_epsilon", .01) <= 0:
            raise ValueError("Invalid plane material or radius")
        builder.add_ground_plane(height=plane.get("height", 0), cfg=newton.ModelBuilder.ShapeConfig(ke=plane.get("ke", 5000), kd=plane.get("kd", 0), mu=plane.get("ground_mu", .4), margin=0))
    if builder_only:
        return builder, params
    builder.color()
    model = builder.finalize(device=device)
    static = export_static(family, model, params, gravity, newton)
    if spec.get("contacts") == "plane":
        model.soft_contact_ke = float(plane.get("ke", 5000))
        model.soft_contact_kd = float(plane.get("kd", 0))
        model.soft_contact_mu = float(plane.get("particle_mu", .4))
        static.update(plane_normal=np.asarray([0,0,1], np.float32), plane_offset=np.asarray(plane.get("height", 0), np.float32), contact_radius=np.asarray(radius, np.float32), contact_material=np.asarray([model.soft_contact_ke, model.soft_contact_kd, np.sqrt(model.soft_contact_mu*plane.get("ground_mu", .4)), plane.get("friction_epsilon", .01)], np.float32))
        if np.min(static["rest_x"][:, 2]-static["plane_offset"]) < 0:
            raise ValueError("Initial particles must be above the physical plane")
    static["reference_length"] = np.asarray(spec.get("reference_length", 0.4), dtype=np.float32)
    values = {key: torch.as_tensor(value.copy(), device=device, dtype=torch.bool if value.dtype.kind == "b" else torch.long if value.dtype.kind in "iu" else torch.float32) for key,value in static.items()}
    metadata = {"family": family, "contacts": spec.get("contacts", False), "params": params, "asset_file": spec.get("_asset_file"), "dt": spec.get("dt", 1/120)}
    scene = Scene(values, metadata)
    state = State(scene.rest_x[None].clone(), scene.rest_q[None].clone(), torch.zeros(1, scene.n, 3, device=device), torch.zeros(1, scene.n, 3, device=device))
    if family in ("softbody", "cloth"):
        state.v[0, :model.particle_count] = torch.as_tensor(model.particle_qd.numpy(), device=device)
    else:
        state.v[0, :model.body_count] = torch.as_tensor(model.body_qd.numpy()[:, :3], device=device)
        state.omega[0, :model.body_count] = torch.as_tensor(model.body_qd.numpy()[:, 3:], device=device)
    return model, scene, state
