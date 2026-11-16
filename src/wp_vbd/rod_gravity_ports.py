"""Six gravity-moment material fields, without solving global dynamics."""
import torch


def subtree_sum(scene, values):
    """Sum descendant world wrenches with fixed logarithmic pointer doubling."""
    depth=getattr(scene,'_rod_gravity_depth',None)
    if depth is None:
        depth=(scene.rod_parent!=torch.arange(scene.n,device=scene.device)).long()
        for ancestor in scene.rod_prefix_ancestors:depth=depth+depth[ancestor]
        scene._rod_gravity_depth=depth
    result=values
    for level,ancestor in enumerate(scene.rod_prefix_ancestors):
        active=depth>=(1<<level)
        result=result.index_add(1,ancestor[active],result[:,active])
    return result


def profile(scene,state,jacobian,weight):
    """Transport known gravity moments, then express them in material axes.

    This constructs only six trial columns. It does not invert a global
    inertia/contact/material operator or consume the nonlinear residual.
    The existing positive H projection and acceptance choose their response.
    """
    force=scene.mass[None,:,None]*scene.gravity[None,None]
    force=force.expand(len(state.x),-1,-1)
    world_moment=torch.linalg.cross(state.x,force,dim=-1)
    total=subtree_sum(scene,torch.cat((force,world_moment),-1))
    child=scene.rod_integration_child
    wrench=total[:,child]
    torque=wrench[...,3:]-torch.linalg.cross(state.x[:,child],wrench[...,:3],dim=-1)
    wrench=torch.cat((wrench[...,:3],torque),-1)
    if scene.rod_original_direction:jc=jacobian[...,1,:]
    else:
        role=scene.rod_integration_child_role[None,:,None,None]==1
        jc=torch.where(role,jacobian[...,1,:],jacobian[...,0,:])
    solved=torch.linalg.solve_ex(jc.transpose(-1,-2),wrench[...,None],check_errors=False)
    value=torch.where(weight>0,solved.result[...,0]/weight.clamp_min(1e-30),0.)
    valid=(solved.info==0).all(-1)&torch.isfinite(value).flatten(1).all(-1)
    return value.detach(),valid
