"""Unintegrated prototype: learn around the computed physical inverse.

Zero modulation retains the existing physical-vector control exactly. The
learned factor acts relative to its projected physical metric, avoiding a
gate initialized at the weaker Jacobi response. Only <=6x6 solves are used.
"""
import torch
from torch import nn
from .inverse_response import inverse_replacement

class AnchoredSubspaceFactor(nn.Module):
    def __init__(self,width=32,bound=.4):
        super().__init__()
        if not 0<bound<1:raise ValueError('Strictly positive factor bound required')
        self.head=nn.Linear(width,21);self.bound=bound
        nn.init.zeros_(self.head.weight);nn.init.zeros_(self.head.bias)

    def forward(self,pooled):
        values=self.head(pooled).tanh();ids=torch.tril_indices(6,6,device=values.device)
        lower=values.new_zeros(len(values),6,6);lower[:,ids[0],ids[1]]=values
        symmetric=lower+lower.transpose(-1,-2)-torch.diag_embed(lower.diagonal(dim1=-2,dim2=-1))
        return symmetric*(self.bound/6)

def anchored_delta(residual,modes,factor,gram_h,modulation,*,ridge=1e-8):
    """Physical-vector delta plus a cancellation-free relative inverse factor."""
    physical,valid,_=inverse_replacement(residual,modes,factor,gram_h,residual.new_full((len(residual),),30.),ridge=ridge)
    z,info=torch.linalg.solve_ex(factor,modes,check_errors=False)
    gb=torch.einsum('bnip,bniq->bpq',z,z);scale=gb.diagonal(dim1=-2,dim2=-1).sum(-1).clamp_min(1e-24).rsqrt()
    u=modes*scale[:,None,None,None];gb=gb*scale[:,None,None].square();gh=gram_h*scale[:,None,None].square()
    eye=torch.eye(6,device=gb.device,dtype=gb.dtype);gate=-torch.expm1(gb.new_tensor(-30.))
    metric=gb+ridge*eye+gate*(gh-gb);root,status=torch.linalg.cholesky_ex(metric,check_errors=False)
    rhs=torch.einsum('bnip,bni->bp',u,residual)[...,None]
    y=torch.linalg.solve_triangular(root,rhs,upper=False)
    t=modulation.to(gb.dtype);change=2*t+t@t
    coeff=torch.linalg.solve_triangular(root.transpose(-1,-2),change@y,upper=True)[...,0]
    correction=torch.einsum('bnip,bp->bni',u,coeff)
    return physical+correction,valid&(info==0).all(-1)&(status==0)&torch.isfinite(correction).flatten(1).all(-1)
