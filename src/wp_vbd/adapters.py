"""Static Newton model conversion. This module never constructs or steps a solver."""

import numpy as np


def np_qmul(a, b):
    return np.concatenate((a[3]*b[:3] + b[3]*a[:3] + np.cross(a[:3], b[:3]), [a[3]*b[3]-np.dot(a[:3], b[:3])]))


def np_rotate(q, x):
    return x + 2*np.cross(q[:3], np.cross(q[:3], x) + q[3]*x)


def export_static(family, model, params, gravity, newton, *, include_joint_relations=True):
    kind = {"softbody": 0, "cloth": 1, "joint": 2, "rigid": 2, "rod": 3}[family]
    nodes = family in ("softbody", "cloth")
    def get(name, shape, dtype=np.float32):
        value = getattr(model, name, None)
        return np.asarray(value.numpy(), dtype=dtype).reshape(shape) if value is not None else np.zeros(tuple(max(0, dim) for dim in shape), dtype=dtype)
    if nodes:
        n = model.particle_count
        rest_x = get('particle_q',(-1,3))
        rest_q = np.tile([0, 0, 0, 1], (n, 1)).astype(np.float32)
        mass = get('particle_mass',(-1,))
        movable = get('particle_inv_mass',(-1,)) > 0
        if n and getattr(model,'particle_flags',None) is not None:
            movable=movable & ((model.particle_flags.numpy() & int(newton.ParticleFlags.ACTIVE))!=0)
        inertia = np.tile(np.eye(3, dtype=np.float32), (n, 1, 1))
    else:
        n = model.body_count
        poses = get('body_q',(-1,7))
        com = get('body_com',(-1,3))
        rest_q = poses[:, 3:7]
        rest_x = np.asarray([p[:3]+np_rotate(p[3:7], c) for p, c in zip(poses, com)],np.float32).reshape(-1,3)
        mass = get('body_mass',(-1,))
        movable = get('body_inv_mass',(-1,)) > 0
        inertia = get('body_inertia',(-1,3,3))
    out = dict(rest_x=rest_x, rest_q=rest_q, mass=mass, movable=movable, inertia=inertia, kind=np.full(n, kind, np.int64), object_id=np.zeros(n, np.int64), gravity=gravity, reference_length=np.array(0.4, np.float32))
    if not nodes:
        out["body_com"] = com.copy()
    if family == "joint":
        out["object_id"] = np.arange(n, dtype=np.int64)
    if nodes:
        out.update(tets=get("tet_indices", (-1, 4), np.int64), tet_Dm=get("tet_poses", (-1, 3, 3)), tet_material=get("tet_materials", (-1, 3)), triangles=get("tri_indices", (-1, 3), np.int64), tri_Dm=get("tri_poses", (-1, 2, 2)), tri_area=get("tri_areas", (-1,)), tri_material=get("tri_materials", (-1, 5)), bend_edges=get("edge_indices", (-1, 4), np.int64), bend_angle=get("edge_rest_angle", (-1,)), bend_length=get("edge_rest_length", (-1,)), bend_material=get("edge_bending_properties", (-1, 2)))
        # Newton releases have also used plural rest-angle/length names.
        for target, source in (("bend_angle", "edge_rest_angles"), ("bend_length", "edge_rest_lengths")):
            value = getattr(model, source, None)
            if value is not None:
                out[target] = value.numpy()
    else:
        types=get('joint_type',(-1,),np.int64)
        # New finite-joint assemblers own their complete rows separately.
        # Export only real body slots there, without a second legacy joint
        # energy or an additional world ghost from this older codec.
        ids = [j for j, jt in enumerate(types) if jt != int(newton.JointType.FREE)] if include_joint_relations else []
        parents, children = get('joint_parent',(-1,),np.int64),get('joint_child',(-1,),np.int64)
        xp, xc = get('joint_X_p',(-1,7)),get('joint_X_c',(-1,7))
        pairs, tags, kp, kc, qp, qc, axes, materials, rest_rel = [], [], [], [], [], [], [], [], []
        ghost = len(out["rest_x"])
        if any(parents[j] < 0 for j in ids):
            out["rest_x"] = np.concatenate((out["rest_x"], [[0, 0, 0]]))
            out["rest_q"] = np.concatenate((out["rest_q"], [[0, 0, 0, 1]]))
            out["mass"] = np.r_[out["mass"], 0]
            out["movable"] = np.r_[out["movable"], False]
            out["inertia"] = np.concatenate((out["inertia"], np.eye(3)[None]))
            out["body_com"] = np.concatenate((out["body_com"], [[0, 0, 0]]))
            out["kind"] = np.r_[out["kind"], kind]
            out["object_id"] = np.r_[out["object_id"], int(out["object_id"].max())+1]
        for j in ids:
            p, c = int(parents[j]), int(children[j])
            p = ghost if p < 0 else p
            tag = {int(newton.JointType.REVOLUTE): 0, int(newton.JointType.ROD): 1, int(newton.JointType.FIXED): 2}[int(types[j])]
            pairs.append([p, c]); tags.append(tag)
            kp.append(xp[j, :3]-out["body_com"][p]); kc.append(xc[j, :3]-out["body_com"][c]); qp.append(xp[j, 3:]); qc.append(xc[j, 3:])
            axis = model.joint_axis.numpy()[int(model.joint_qd_start.numpy()[j])] if tag == 0 else [0, 0, 1]
            axes.append(axis)
            materials.append([params["stretch"], params["shear"], params["bending"], params["twist"]] if tag == 1 else [params["linear_ke"], params["linear_ke"], params["angular_ke"], params["angular_ke"]])
            qa = np_qmul(out["rest_q"][p], xp[j, 3:]); qb = np_qmul(out["rest_q"][c], xc[j, 3:])
            rest_rel.append(np_qmul(qa*np.array([-1, -1, -1, 1]), qb))
        out.update(joints=np.array(pairs, np.int64).reshape(-1, 2), joint_tag=np.array(tags, np.int64), anchor_p=np.array(kp).reshape(-1, 3), anchor_c=np.array(kc).reshape(-1, 3), anchor_qp=np.array(qp).reshape(-1, 4), anchor_qc=np.array(qc).reshape(-1, 4), joint_axis=np.array(axes).reshape(-1, 3), joint_material=np.array(materials).reshape(-1, 4), joint_rest_q=np.array(rest_rel).reshape(-1, 4))
    for key, value in out.items():
        if np.asarray(value).dtype.kind == "f":
            out[key] = np.asarray(value, dtype=np.float32)
    return out
