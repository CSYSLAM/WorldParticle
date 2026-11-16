"""Six material-strain fields from batched local affine forest prefixes."""
import torch


def fields(scene, jacobian, profile=None):
    """Solve each child 6x6 block, then compose its affine parent map.

    A child tangent satisfies Jc*z_child = strain-Jp*z_parent. Parallel
    doubling integrates these local equations from the component's prescribed
    body, or its original root when every body is movable. Authored material
    row orientation remains unchanged when integration reverses an edge.
    The result has six material columns per interaction component, independent
    of rod length. It is a basis construction, not a global dynamics solve.
    """
    ids = scene.rod.joints; batch = len(jacobian)
    parent, child = scene.rod_integration_parent,scene.rod_integration_child
    if scene.rod_original_direction:
        jp,jc=jacobian[...,0,:],jacobian[...,1,:]
    else:
        role=scene.rod_integration_child_role[None,:,None,None]==1
        jp=torch.where(role,jacobian[...,0,:],jacobian[...,1,:])
        jc=torch.where(role,jacobian[...,1,:],jacobian[...,0,:])
    # Per-row material frames own the six coefficients. Segment scaling keeps
    # long chains numerically comparable before H normalization.
    length = (scene.rest_x[child]-scene.rest_x[parent]).norm(dim=-1).clamp_min(1e-12)
    size = length/scene.length
    if profile is None:
        desired = torch.diag_embed(torch.cat((length[:, None].expand(-1, 3), size[:, None].expand(-1, 3)), -1))[None].expand(batch,-1,-1,-1)
    else:
        if profile.shape != jacobian.shape[:3]:raise ValueError('Six per-row material strain profiles required')
        desired = torch.diag_embed(profile.detach())
    rhs = torch.cat((-jp, desired), -1)
    solved = torch.linalg.solve_ex(jc, rhs, check_errors=False)
    a = torch.eye(6, device=scene.device, dtype=jacobian.dtype)[None, None].expand(batch, scene.n, -1, -1).clone()
    b = jacobian.new_zeros(batch, scene.n, 6, 6)
    a = a.index_copy(1, child, solved.result[..., :6]); b = b.index_copy(1, child, solved.result[..., 6:])
    for ancestors in scene.rod_prefix_ancestors:
        b, a = b+a@b[:, ancestors], a@a[:, ancestors]
    valid = (solved.info == 0).all(-1) & torch.isfinite(a).flatten(1).all(-1) & torch.isfinite(b).flatten(1).all(-1)
    for plan in getattr(scene,'rod_boundary_plans',()):
        from .rod_boundary_ports import compatible
        corrected,boundary_valid=compatible(plan,a,b)
        b=b.index_copy(1,plan.nodes,corrected)
        valid=valid&boundary_valid
    return b*scene.dof[None, :, :, None], valid
