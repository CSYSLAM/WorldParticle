"""Metric-aware learned subspace replacement, with exact physical initialization.

The v17 factor has a useful safety bound, but its response is at most 1.96
times the local inverse. That cannot resolve a missing slow global mode.
Here the learned six-column space obtains its scale from U^T H U, rather
than asking a small local multiplier to represent the global inverse.
"""
import torch
from torch import nn
from .protected_response import InvariantPortResponse,centers,metric_action,metric_terms
from .response_operator import skew


def energy_gram(scene,metric,columns):
    """Positive sparse terms; the returned matrix has at most six rows."""
    value=columns.double()
    gram=torch.einsum('bnip,bnij,bnjq->bpq',value,metric.local.double(),value)
    for term in metric_terms(scene,metric):
        image=torch.einsum('brcsi,brsip->brcp',term.jacobian.double(),value[:,term.ids])
        gram=gram+torch.einsum('brcp,brc,brcq->bpq',image,term.weight.double(),image)
    return gram


def inverse_replacement(residual,modes,factor,gram_h,logit,*,ridge=1e-8):
    """Return (B_theta-B0) r using two <=6x6 positive solves.

    G_B=U^T B0^-1 U and G_H=U^T H U. Replace the local inverse on this
    subspace by [(1-g)G_B+g G_H+ridge I]^-1. The subtraction is evaluated
    with an inverse-difference identity, so g=0 returns exactly zero.

    A trace bound sets a safe negative gate range; positive gates approach
    the projected physical inverse. No assumed global Jacobi spectral bound
    or eigenvalue decomposition is needed.
    """
    # Whitening times the local Cholesky factor is generally not triangular.
    z,info_factor=torch.linalg.solve_ex(factor,modes,check_errors=False)
    gb=torch.einsum('bnip,bniq->bpq',z,z)
    scale=gb.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-24).rsqrt()
    u=modes*scale[:,None,None,None]
    gb=gb*scale[:,None,None].square()
    gh=gram_h*scale[:,None,None].square()
    eye=torch.eye(gb.shape[-1],device=gb.device,dtype=gb.dtype)
    b=gb+ridge*eye;h=gh+ridge*eye
    lb,info_b=torch.linalg.cholesky_ex(b,check_errors=False)
    trace=torch.cholesky_solve(h,lb).diagonal(dim1=-2,dim2=-1).sum(-1)
    finite=torch.isfinite(trace)&(trace>0)
    safe=torch.where(finite,trace,1.).clamp_min(1.)
    negative=.5/safe
    # Both sides have unit slope at zero. A scaled tanh would saturate the
    # positive gate at tiny logits whenever the safe negative interval is
    # small, suppressing the useful positive-side training gradient.
    positive=-torch.expm1(-logit.clamp_min(0))
    below=negative*torch.expm1(logit.clamp_max(0)/negative)
    gate=torch.where(logit>=0,positive,below)
    mix=b+gate[:,None,None]*(gh-gb)
    lm,info_m=torch.linalg.cholesky_ex(mix,check_errors=False)
    restricted=torch.einsum('bnip,bni->bp',u,residual)
    first=torch.cholesky_solve(restricted[...,None],lb)
    second=torch.cholesky_solve((gh-gb)@first,lm)[...,0]
    delta=-gate[:,None,None]*torch.einsum('bnip,bp->bni',u,second)
    valid=finite&(info_b==0)&(info_m==0)&(info_factor==0).all(-1)&torch.isfinite(delta).flatten(1).all(-1)
    return delta,valid,gate


class InversePortResponse(InvariantPortResponse):
    """Shared shapes plus boundary/drive context and a scalar inverse gate."""
    def __init__(self,width=32,*,ridge=1e-8):
        super().__init__(width)
        self.factor=nn.Linear(width,1)
        nn.init.zeros_(self.factor.weight);nn.init.zeros_(self.factor.bias)
        self.context=nn.Sequential(nn.Linear(4,width),nn.SiLU(),nn.Linear(width,width))
        nn.init.zeros_(self.context[-1].weight);nn.init.zeros_(self.context[-1].bias)
        self.ridge=ridge

    def encode(self,scene,state,physical,baseline,relation,dt):
        h,features=super().encode(scene,state,physical,baseline,relation,dt)
        position=centers(scene,state)
        extra=position.new_zeros(len(position),scene.n,4)
        batched=bool(getattr(scene,'component_batch',False))
        if batched:
            from .component_batch import context_features
            extra=context_features(scene,state,physical,position,dt)
        for ids in (() if batched else scene.interaction_entities):
            mobile=scene.movable[ids]
            mass=scene.mass[ids]*mobile
            center=(position[:,ids]*mass[None,:,None]).sum(1,keepdim=True)/mass.sum().clamp_min(1e-30)
            fixed=(~mobile).to(position.dtype)
            support=(position[:,ids]*fixed[None,:,None]).sum(1,keepdim=True)/fixed.sum().clamp_min(1)
            support=torch.where((fixed.sum()>0),support,center)
            activity=(physical[:,ids,:3].norm(dim=-1)/scene.length+physical[:,ids,3:].norm(dim=-1))*mobile
            drive=(position[:,ids]*activity[...,None]).sum(1,keepdim=True)/activity.sum(-1,keepdim=True).clamp_min(1e-30)[...,None]
            drive=torch.where((activity.sum(-1)>1e-20)[:,None,None],drive,center)
            axis=drive-support;length=axis.norm(dim=-1,keepdim=True)
            signed=((position[:,ids]-support)*axis).sum(-1,keepdim=True)/(length.clamp_min(1e-12)*scene.length)
            signed=torch.where(length>scene.length*1e-8,signed,0.)
            work=(physical[:,ids,:3]*state.v[:,ids]).sum(-1,keepdim=True)*dt/scene.length**2
            work=work+(physical[:,ids,3:]*state.omega[:,ids]).sum(-1,keepdim=True)*dt
            extra[:,ids]=torch.cat(((position[:,ids]-support).norm(dim=-1,keepdim=True)/scene.length*(fixed.sum()>0),
                (position[:,ids]-drive).norm(dim=-1,keepdim=True)/scene.length,signed,work),-1).to(extra.dtype)
        added=self.context(extra.to(h.dtype))
        return h+torch.where(scene.movable[None,:,None],added,0.),features

    def forward(self,scene,state,residual,whitening,baseline,relation,physical,coarse,metric,dt):
        h,_=self.encode(scene,state,physical,baseline,relation,dt)
        position=centers(scene,state).double();w=whitening.double()
        safe_w=w+torch.diag_embed((~scene.dof).double())[None]
        root,info=torch.linalg.cholesky_ex(baseline.double(),check_errors=False)
        factor=safe_w@root
        valid=(info==0).all(-1)&torch.isfinite(root).flatten(1).all(-1)
        result=-(w@baseline.double()@(w@residual.double()[...,None]))[...,0]
        for index,ids in enumerate(scene.interaction_entities):
            mobile=scene.movable[ids];mass=scene.mass[ids].double()*mobile
            center=(position[:,ids]*mass[None,:,None]).sum(1,keepdim=True)/mass.sum().clamp_min(1e-30)
            shape=1+.5*self.shape(h[:,ids]).tanh().double()
            eye=torch.eye(3,device=position.device,dtype=torch.float64)
            raw=position.new_zeros(len(position),len(ids),6,6)
            raw[...,:3,:3]=scene.length*shape[...,:1,None]*eye
            raw[...,:3,3:]=-shape[...,1:,None]*skew(position[:,ids]-center)
            raw[...,3:,3:]=shape[...,1:,None]*eye
            raw=raw*scene.dof[ids][None,:,:,None]
            full=raw.new_zeros(len(position),scene.n,6,6).index_copy(1,ids,raw)
            full=full-coarse(metric_action(metric,full))
            complete=coarse.ranks[index]>=scene.dof[ids].sum()
            full=torch.where(complete[:,None,None,None],0.,full)
            gh=energy_gram(scene,metric,full)
            pooled=(h[:,ids]*mobile[None,:,None]).sum(1)/mobile.sum().clamp_min(1)
            delta,ok,_=inverse_replacement(residual[:,ids].double(),full[:,ids],factor[:,ids],gh,
                self.factor(pooled)[...,0].double(),ridge=self.ridge)
            result=result.index_add(1,ids,-delta);valid&=ok
        return result*scene.dof,valid
