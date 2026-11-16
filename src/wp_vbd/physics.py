"""Bounded local proposals; no Newton import or full iterative reference solve.

Tet/membrane elastic force and projected per-node blocks follow the equations
audited at Newton 25298aea. Pilot joint energies are small-angle proposals, not
an exact reproduction of its compliant ALM rotational equation or rod law.
Their remaining errors are supervised by state labels, not a zero-residual loss.
"""

import torch

from . import math3d as so3


def scatter(values, indices, count):
    shape = (values.shape[0], count, *values.shape[3:])
    result = values.new_zeros(shape)
    return result.index_add(1, indices.flatten(), values.flatten(1, 2))


def elastic_nodes(scene, x):
    b, n, _ = x.shape
    gradient = torch.zeros_like(x)
    blocks = x.new_zeros(b, n, 3, 3)
    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    if "tets" in scene.values and len(scene.tets):
        ids, dm = scene.tets, scene.tet_Dm
        points = x[:, ids]
        f = (points[:, :, 1:]-points[:, :, :1]).transpose(-1, -2) @ dm
        cof = so3.cofactor(f)
        jac = (f[..., :, 0]*cof[..., :, 0]).sum(-1)
        mu, lam = scene.tet_material[:, 0], scene.tet_material[:, :2].sum(-1)
        volume = scene.tet_volume if 'tet_volume' in scene.values else (1/(6*torch.linalg.det(dm))).abs()
        p = mu[None, :, None, None]*f + (lam[None]*(jac-1)-mu[None])[..., None, None]*cof
        m = torch.cat((-dm.sum(-2, keepdim=True), dm), -2)
        d = (p[:, :, None] @ m[None, ..., None]).squeeze(-1)*volume[None, :, None, None]
        cj = (cof[:, :, None] @ m[None, ..., None]).squeeze(-1)
        h = volume[None, :, None, None, None]*(mu[None, :, None, None, None]*m.square().sum(-1)[None, :, :, None, None]*eye + lam[None, :, None, None, None]*cj[..., :, None]*cj[..., None, :])
        gradient = gradient + scatter(d, ids, n)
        blocks = blocks + scatter(h, ids, n)
    if "triangles" in scene.values and len(scene.triangles):
        ids, dm = scene.triangles, scene.tri_Dm
        points = x[:, ids]
        f = (points[:, :, 1:]-points[:, :, :1]).transpose(-1, -2) @ dm
        f0, f1 = f[..., 0], f[..., 1]
        normal_raw = torch.linalg.cross(f0, f1)
        js = normal_raw.square().sum(-1).clamp_min(1e-20).sqrt()
        normal = normal_raw/js[..., None]
        j0, j1 = torch.linalg.cross(f1, normal), torch.linalg.cross(normal, f0)
        mu, lam = scene.tri_material[:, 0], scene.tri_material[:, :2].sum(-1)
        s = lam[None]*(js-1)-mu[None]
        p0 = mu[None, :, None]*f0+s[..., None]*j0
        p1 = mu[None, :, None]*f1+s[..., None]*j1
        m = torch.cat((-dm.sum(-2, keepdim=True), dm), -2)
        a, c = m[..., 0], m[..., 1]
        area = scene.tri_area
        grad = (p0[:, :, None]*a[None, ..., None]+p1[:, :, None]*c[None, ..., None])*area[None, :, None, None]
        dj = j0[:, :, None]*a[None, ..., None]+j1[:, :, None]*c[None, ..., None]
        w = f1[:, :, None]*a[None, ..., None]-f0[:, :, None]*c[None, ..., None]
        r = s.clamp_min(0)/js
        icoeff = mu[None, :, None]*(a*a+c*c)[None]+r[..., None]*w.square().sum(-1)
        h = icoeff[..., None, None]*eye + (lam[None]-r)[..., None, None, None]*dj[..., :, None]*dj[..., None, :] - r[..., None, None, None]*w[..., :, None]*w[..., None, :]
        gradient = gradient+scatter(grad, ids, n)
        blocks = blocks+scatter(h*area[None, :, None, None, None], ids, n)
    # Bending uses a differentiable scalar energy. Its metric is a deliberately
    # conservative diagonal approximation; it is not a colored VBD sweep.
    if "bend_edges" in scene.values and len(scene.bend_edges):
        valid = scene.bending_indices
        ids = scene.bend_edges[valid]
        if len(ids):
            with torch.enable_grad():
                xx = x if x.requires_grad else x.detach().requires_grad_(True)
                pts = xx[:, ids]
                n1 = so3.normalize(torch.linalg.cross(pts[:, :, 2]-pts[:, :, 0], pts[:, :, 3]-pts[:, :, 0]))
                n2 = so3.normalize(torch.linalg.cross(pts[:, :, 3]-pts[:, :, 1], pts[:, :, 2]-pts[:, :, 1]))
                edge = so3.normalize(pts[:, :, 3]-pts[:, :, 2])
                theta = torch.atan2((torch.linalg.cross(n1, n2)*edge).sum(-1), (n1*n2).sum(-1))
                k = scene.bend_material[valid, 0]*scene.bend_length[valid]
                energy = 0.5*(k*(theta-scene.bend_angle[valid]).square()).sum()
                d = torch.autograd.grad(energy, xx, create_graph=x.requires_grad)[0]
            gradient = gradient+d
            metric = (scene.bend_material[valid, 0]/scene.bend_length[valid].clamp_min(1e-6)).repeat_interleave(4)
            diagonal = x.new_zeros(n).index_add(0, ids.flatten(), metric)
            blocks = blocks + diagonal[None, :, None, None]*eye
    return gradient, blocks


def joint_proposal(scene, x, q):
    if "joints" not in scene.values or not len(scene.joints):
        return torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
    enabled_grad = torch.is_grad_enabled()
    with torch.enable_grad():
        xx = x if x.requires_grad else x.detach().requires_grad_(True)
        delta = torch.zeros_like(x, requires_grad=True)
        qq = so3.multiply(so3.exp(delta), q)
        p, c = scene.joints.unbind(-1)
        qa = so3.multiply(qq[:, p], scene.anchor_qp)
        qb = so3.multiply(qq[:, c], scene.anchor_qc)
        pa = xx[:, p]+so3.rotate(qq[:, p], scene.anchor_p)
        pb = xx[:, c]+so3.rotate(qq[:, c], scene.anchor_c)
        gap = so3.rotate(so3.conjugate(qa), pb-pa)
        rel = so3.multiply(so3.conjugate(qa), qb)
        angular = so3.log(so3.multiply(rel, so3.conjugate(scene.joint_rest_q)))
        axis = scene.joint_axis
        hinge = scene.joint_tag == 0
        angular = torch.where(hinge[None, :, None], angular-(angular*axis).sum(-1, keepdim=True)*axis, angular)
        mat = scene.joint_material
        rod = scene.joint_tag == 1
        linear_k = torch.stack((mat[:, 1], mat[:, 1], mat[:, 0]), -1)
        angular_k = torch.stack((mat[:, 2], mat[:, 2], mat[:, 3]), -1)
        linear_k = torch.where(rod[:, None], linear_k, mat[:, :1].expand(-1, 3))
        energy = 0.5*(linear_k*gap.square()+angular_k*angular.square()).sum()
        gx, gq = torch.autograd.grad(energy, (xx, delta), create_graph=enabled_grad)
    if not enabled_grad:
        gx, gq = gx.detach(), gq.detach()
    kx = linear_k.amax(-1)
    lever = scene.anchor_p.square().sum(-1)+scene.anchor_c.square().sum(-1)
    kr = angular_k.amax(-1)+kx*lever
    linear_metric = x.new_zeros(scene.n).index_add(0, scene.joints.flatten(), kx.repeat_interleave(2))
    angular_metric = x.new_zeros(scene.n).index_add(0, scene.joints.flatten(), kr.repeat_interleave(2))
    return gx, gq, linear_metric[None, :, None].expand_as(x), angular_metric[None, :, None].expand_as(x)


def plane_contact(scene, x, start_x, dt):
    """Newton fixed-penalty normal law and regularized Coulomb metric.

    Normal load used by friction excludes damping, matching the reference.
    The plane is static; start_x is the start of the physical step, never an
    iteration-dependent friction anchor. No persistent contact memory is used.
    """
    n = scene.plane_normal
    ke, kd, mu, eps = scene.contact_material.unbind()
    penetration = (scene.contact_radius-(x*n).sum(-1)+scene.plane_offset).clamp_min(0)
    active = penetration > 0
    displacement = x-start_x
    dn = (displacement*n).sum(-1)
    normal_load = ke*penetration
    damp = (kd/dt)*active*(dn < 0)
    normal_force = normal_load-damp*dn
    tangent = displacement-dn[..., None]*n
    norm = (tangent.square().sum(-1)+1e-30).sqrt()
    epsilon = eps*dt
    factor = torch.where(norm > epsilon, 1/norm, (2-norm/epsilon)/epsilon)
    scale = mu*normal_load*factor*(norm > 1e-15)
    force = normal_force[..., None]*n-scale[..., None]*tangent
    nn = n[:, None]*n[None, :]
    tangent_projector = torch.eye(3, device=x.device, dtype=x.dtype)-nn
    blocks = (ke*active+damp)[..., None, None]*nn+scale[..., None, None]*tangent_projector
    return -force, blocks


def inertial_target(scene, state, dt, force=None, torque=None):
    movable = scene.movable[None, :, None]
    target_x = state.x+dt*state.v+dt*dt*scene.gravity
    target_q = so3.multiply(so3.exp(dt*state.omega), state.q)
    if force is not None:
        target_x = target_x+dt*dt*force/scene.mass.clamp_min(1e-8)[None, :, None]
    if torque is not None:
        rotation = so3.matrix(state.q)
        inertia = rotation @ scene.inertia @ rotation.transpose(-1, -2)
        acceleration = torch.linalg.solve_ex(inertia+1e-8*torch.eye(3, device=state.x.device), torque[..., None], check_errors=False).result.squeeze(-1)
        target_q = so3.multiply(so3.exp(dt*dt*acceleration), target_q)
    return torch.where(movable, target_x, state.x), torch.where(movable, target_q, state.q)


def local_update(scene, x, q, target_x, target_q, dt, start_x=None):
    mass_h2 = scene.mass.clamp_min(1e-8)/dt**2
    eye = torch.eye(3, device=x.device, dtype=x.dtype)
    gx = mass_h2[None, :, None]*(x-target_x)
    blocks = mass_h2[None, :, None, None]*eye
    blocks = blocks.expand(x.shape[0], scene.n, 3, 3)
    g_elastic, h_elastic = elastic_nodes(scene, x)
    gx = gx+g_elastic
    blocks = blocks+h_elastic
    if scene.has_plane:
        contact_gradient, contact_blocks = plane_contact(scene, x, start_x, dt)
        gx = gx+contact_gradient
        blocks = blocks+contact_blocks
    jx, jq, hx, hq = joint_proposal(scene, x, q)
    gx = gx+jx
    blocks = blocks+torch.diag_embed(hx)
    dx = torch.linalg.solve_ex(blocks.detach()+1e-7*eye, -gx[..., None], check_errors=False).result.squeeze(-1)
    rotation = so3.matrix(q)
    inertia_world = rotation @ scene.inertia @ rotation.transpose(-1, -2)
    gq = (inertia_world @ so3.log(so3.multiply(q, so3.conjugate(target_q)))[..., None]).squeeze(-1)/dt**2+jq
    angular_blocks = inertia_world/dt**2+torch.diag_embed(hq)+1e-7*eye
    dr = torch.linalg.solve_ex(angular_blocks.detach(), -gq[..., None], check_errors=False).result.squeeze(-1)
    move = scene.movable[None, :, None]
    body = (scene.kind >= 2)[None, :, None]
    return torch.where(move, dx, torch.zeros_like(dx)), torch.where(move & body, dr, torch.zeros_like(dr))
