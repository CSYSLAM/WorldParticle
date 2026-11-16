"""Contact-free acceptance guard; no general DAT contact claim is made."""

import torch


def tetrahedron_prefix_coefficients(scene, start, end):
    ids, dm = scene.tets, scene.tet_Dm
    a = start[:, ids]
    b = end[:, ids]
    f = (a[:, :, 1:]-a[:, :, :1]).transpose(-1, -2) @ dm
    df = ((b[:, :, 1:]-b[:, :, :1]).transpose(-1, -2) @ dm)-f
    x, y, z = f.unbind(-1)
    u, v, w = df.unbind(-1)
    triple = lambda p, q, r: (p*torch.linalg.cross(q, r)).sum(-1)
    c0 = triple(x, y, z)
    c1 = triple(u, y, z)+triple(x, v, z)+triple(x, y, w)
    c2 = triple(u, v, z)+triple(u, y, w)+triple(x, v, w)
    c3 = triple(u, v, w)
    return c0,c1,c2,c3


def tetrahedron_polynomial_minimum(coefficients,alpha):
    c0,c1,c2,c3=coefficients
    aa, bb, cc = 3*c3, 2*c2, c1
    discriminant = bb.square()-4*aa*cc
    denominator = torch.where(aa.abs() > 1e-12, 2*aa, torch.ones_like(aa))
    root = discriminant.clamp_min(0).sqrt()
    r1, r2 = (-bb+root)/denominator, (-bb-root)/denominator
    linear = -cc/torch.where(bb.abs() > 1e-12, bb, torch.ones_like(bb))
    candidates = [torch.zeros_like(c0), alpha[:, None].expand_as(c0)]
    for r, valid in ((r1, (aa.abs() > 1e-12) & (discriminant >= 0)), (r2, (aa.abs() > 1e-12) & (discriminant >= 0)), (linear, (aa.abs() <= 1e-12) & (bb.abs() > 1e-12))):
        candidates.append(torch.where(valid & (r > 0) & (r < alpha[:, None]), r, torch.zeros_like(r)))
    values = [((c3*t+c2)*t+c1)*t+c0 for t in candidates]
    return torch.stack(values).amin(0).amin(-1)


def tetrahedron_prefix_minimum(scene, start, end, alpha):
    return tetrahedron_polynomial_minimum(tetrahedron_prefix_coefficients(scene,start,end),alpha)


def accepted_scale(scene, start, end):
    with torch.no_grad():
        alpha = start.new_ones(start.shape[0])
        valid = torch.ones_like(alpha, dtype=torch.bool)
        if scene.has_plane:
            # Along a linear proposal, signed plane distance is affine. This
            # guards the physical plane, not the numerical activation offset.
            a = (start*scene.plane_normal).sum(-1)-scene.plane_offset
            b = (end*scene.plane_normal).sum(-1)-scene.plane_offset
            valid = (a >= -1e-7*scene.length).all(-1)
            bound = (a.clamp_min(0)/(a-b).clamp_min(1e-20)).clamp(0, 1)
            bound = torch.where(b < 0, 0.99*bound, torch.ones_like(bound))
            alpha = bound.amin(-1)
        if "tets" not in scene.values or not len(scene.tets):
            return torch.where(valid, alpha, torch.zeros_like(alpha)), valid
        for _ in range(6):
            minimum = tetrahedron_prefix_minimum(scene, start, end, alpha)
            alpha = torch.where(minimum >= 0.05, alpha, alpha*0.5)
        valid = valid & (tetrahedron_prefix_minimum(scene, start, end, alpha) >= 0.05)
        return torch.where(valid, alpha, torch.zeros_like(alpha)), valid
