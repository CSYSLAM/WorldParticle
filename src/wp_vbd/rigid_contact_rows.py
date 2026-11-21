"""Local rigid contact wrenches and positive solve metrics.

This is a row operator, not a collision pipeline or dynamics solver. A caller
must supply current collision geometry, effective depth and frozen friction
C0. Contact identity, buffer capacity, history lifetime and feasible motion
are deliberately outside this module; no demo is admitted by its existence.
"""
from dataclasses import dataclass
import torch
from .joint_history import compliant_coefficients,compliant_dual_step
from .rod_rows import rotate_pinned
from .response_operator import skew


def project_disk(value,radius):
    length=value.norm(dim=-1)
    return value*(radius/length.clamp_min(1e-30)).clamp(0,1)[...,None]


def compliant_contact_dual(dual,normal,depth,tangent,material,rho_n,rho_t,mu):
    """New numerical multiplier; the accepted input is never modified."""
    old_n=(dual*normal).sum(-1)
    new_n=compliant_dual_step(old_n,depth,material,rho_n).clamp_min(0)
    old_t=dual-old_n[...,None]*normal
    new_t=project_disk(old_t+rho_t[...,None]*tangent,mu*new_n)
    return new_n[...,None]*normal+new_t


def stabilized_contact(depth,c0,normal,alpha):
    c0_n=(c0*normal).sum(-1)
    return depth-alpha*c0_n,(1-alpha)*(c0-c0_n[...,None]*normal)


def numerical_weight_policy(normal_support,tangent_support,structural_support,material):
    """Pinned contact policy; unlike bilateral rods, there is no 9K floor.

    Supports are inverse-Delassus quantities supplied by the assembler. These
    are numerical weights and never replace the physical contact material.
    """
    rho_n=torch.where((normal_support>0)&(material>0),normal_support,0.)
    return rho_n,torch.maximum(tangent_support,torch.minimum(rho_n,structural_support))


def reciprocal_slip(length,epsilon):
    safe=epsilon.clamp_min(1e-30)
    ramp=(2-length/safe)/safe
    return torch.where((length>epsilon)&(length>0),1/length.clamp_min(1e-30),torch.where(epsilon>0,ramp,0.))


@dataclass(frozen=True)
class ContactContext:
    material:torch.Tensor
    rho_n:torch.Tensor
    rho_t:torch.Tensor
    damping:torch.Tensor
    mu:torch.Tensor
    dual:torch.Tensor
    friction_c0:torch.Tensor
    friction_epsilon:torch.Tensor
    dt:float
    mode:str='compliant'
    compiled_rows:bool=False


def evaluate(position,rotation,previous_position,previous_rotation,body_com,point_local,offset_local,normal,depth,surface_velocity,context):
    if getattr(context,'compiled_rows',False) and not torch.is_grad_enabled() and position.is_cuda and depth.numel():
        from .compiled_contact import evaluate as compiled
        return compiled(_evaluate_reference,position,rotation,previous_position,previous_rotation,body_com,
                        point_local,offset_local,normal,depth,surface_velocity,context)
    return _evaluate_reference(position,rotation,previous_position,previous_rotation,body_com,
                               point_local,offset_local,normal,depth,surface_velocity,context)


def _evaluate_reference(position,rotation,previous_position,previous_rotation,body_com,point_local,offset_local,normal,depth,surface_velocity,context):
    """Return physical pair wrenches and 6x6 COM solve blocks.

    Leading dimensions are independent contacts; the penultimate body axis
    has length two. A world shape is represented by an identity pose and zero
    COM; its fixed DOFs must be masked by the assembler. Normal damping uses
    skeleton points; friction uses rotating surface anchors. The returned
    metric is PSD and is not advertised as an exact friction Hessian.
    """
    if context.mode not in ('compliant','legacy_hard','legacy_soft'):raise ValueError(context.mode)
    if context.dt<=0:raise ValueError('Positive physical timestep required')
    ctx=context;n=normal;dt=ctx.dt
    skeleton=position+rotate_pinned(rotation,point_local)
    previous_skeleton=previous_position+rotate_pinned(previous_rotation,point_local)
    surface=position+rotate_pinned(rotation,point_local+offset_local)
    previous_surface=previous_position+rotate_pinned(previous_rotation,point_local+offset_local)
    center=position+rotate_pinned(rotation,body_com)
    # Preserve the source's subtraction order before damping amplifies it.
    relative=(skeleton[...,1,:]-previous_skeleton[...,1,:]-skeleton[...,0,:]+previous_skeleton[...,0,:])/dt
    approach=(relative*n).sum(-1)
    relative_surface=(surface[...,1,:]-previous_surface[...,1,:]-surface[...,0,:]+previous_surface[...,0,:])/dt+surface_velocity
    tangent_velocity=relative_surface-(relative_surface*n).sum(-1,keepdim=True)*n
    slip=tangent_velocity*dt
    lam_n=(ctx.dual*n).sum(-1);lam_t=ctx.dual-lam_n[...,None]*n
    k=ctx.rho_n
    if ctx.mode=='compliant':
        scale,k,_=compliant_coefficients(ctx.material,ctx.rho_n)
        effective=normal*(scale*lam_n)[...,None]+lam_t
        lam_n=(effective*normal).sum(-1)
        lam_t=effective-lam_n[...,None]*normal
    load=(k*depth+lam_n).clamp_min(0)
    active=(k>0)&((depth>1e-9)|(lam_n>0))
    if ctx.mode!='legacy_hard':active&=load>0
    outer=n[..., :,None]*n[...,None,:]
    identity=torch.eye(3,device=n.device,dtype=n.dtype)
    tangent_projector=identity-outer
    fn=load[...,None]*n;kn=k[...,None,None]*outer
    if ctx.mode in ('compliant','legacy_hard'):
        rho_t=ctx.rho_t if ctx.mode=='compliant' else ctx.rho_n
        trial=rho_t[...,None]*(-slip+ctx.friction_c0)+lam_t
        length=trial.norm(dim=-1);limit=ctx.mu*load
        ft=project_disk(trial,limit)
        kt=rho_t[...,None,None]*tangent_projector
        if ctx.mode=='compliant':
            direction=trial/length.clamp_min(1e-30)[...,None]
            along=direction[..., :,None]*direction[...,None,:]
            scale=limit/length.clamp_min(1e-30)
            slide=limit*reciprocal_slip(slip.norm(dim=-1),ctx.friction_epsilon*dt)
            sliding=(scale*rho_t)[...,None,None]*(tangent_projector-along)+slide[...,None,None]*along
            kt=torch.where((length>limit)[...,None,None],sliding,kt)
    else:
        length=slip.norm(dim=-1)
        scale=ctx.mu*load*reciprocal_slip(length,ctx.friction_epsilon*dt)
        scale=torch.where(length>0,scale,0.)
        ft=-scale[...,None]*slip;kt=scale[...,None,None]*tangent_projector
    friction=(ctx.mu>0)&(load>0)
    ft=torch.where(friction[...,None],ft,0.);kt=torch.where(friction[...,None,None],kt,0.)
    kd=torch.where((ctx.damping>0)&(approach<0)&(load>0),ctx.damping,0.)
    fn=fn-(kd*approach)[...,None]*n
    kn=kn+(kd/dt)[...,None,None]*outer
    sign=n.new_tensor([-1.,1.])
    force=(fn+ft)[...,None,:]*sign[:,None]
    rs,rc=skeleton-center,surface-center
    torque=(torch.linalg.cross(rs,fn[...,None,:])+torch.linalg.cross(rc,ft[...,None,:]))*sign[:,None]
    jac_s=torch.cat((identity.expand(*rs.shape[:-1],3,3),-skew(rs)),-1)
    jac_c=torch.cat((identity.expand(*rc.shape[:-1],3,3),-skew(rc)),-1)
    blocks=jac_s.transpose(-1,-2)@kn[...,None,:,:]@jac_s+jac_c.transpose(-1,-2)@kt[...,None,:,:]@jac_c
    force=torch.where(active[...,None,None],force,0.)
    torque=torch.where(active[...,None,None],torque,0.)
    blocks=torch.where(active[...,None,None,None],blocks,0.)
    finite=torch.stack([torch.isfinite(value).reshape(*depth.shape,-1).all(-1) for value in (position,rotation,previous_position,previous_rotation,body_com,point_local,offset_local,n,depth,surface_velocity,ctx.material,ctx.rho_n,ctx.rho_t,ctx.damping,ctx.mu,ctx.dual,ctx.friction_c0,ctx.friction_epsilon,force,torque,blocks)]).all(0)
    valid=finite&((n.square().sum(-1)-1).abs()<1e-5)
    valid&=((rotation.square().sum(-1)-1).abs()<1e-5).all(-1)&((previous_rotation.square().sum(-1)-1).abs()<1e-5).all(-1)
    valid&=(ctx.friction_c0*n).sum(-1).abs()<=1e-5*ctx.friction_c0.norm(dim=-1).clamp_min(1e-8)
    for value in (ctx.material,ctx.rho_n,ctx.rho_t,ctx.damping,ctx.mu,ctx.friction_epsilon):valid&=value>=0
    pair_sign=sign[:,None,None]
    return {'force':force,'torque':torque,'self_metric':blocks,
        'relative_skeleton_jacobian':jac_s*pair_sign,
        'relative_surface_jacobian':jac_c*pair_sign,
        'normal_solve_metric':torch.where(active[...,None,None],kn,0.),
        'tangent_solve_metric':torch.where(active[...,None,None],kt,0.),
        'normal_load_before_damping':load,'tangent_force':torch.where(active[...,None],ft,0.),'valid':valid}
