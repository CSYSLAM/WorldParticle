"""Positive geometric stiffness for linear particle paths under prestress.

For the existing stable Neo-Hookean energy, the second Piola stress is
S=mu I + ((mu+lambda)(J-1)-mu) J (F^T F)^-1. Its positive spectral part
adds sum_i dF_i S_+ dF_i^T to the frozen positive material metric. It vanishes
at stress-free rest; it need not annihilate rotations along a LINEAR path
when the material is prestressed. No force or constitutive energy changes.
"""
import torch

from . import math3d as so3


@torch.no_grad()
def positive_prestress_factor(f,dm,material,measure):
    batch,rows,spatial,dimension=f.shape
    if spatial!=3 or dimension not in (2,3) or dm.shape!=(rows,dimension,dimension):
        raise ValueError('A physical3x2 or3x3 deformation and matched rest inverse required')
    if material.shape[0]!=rows or measure.shape!=(rows,):raise ValueError('Matched element material and measure required')
    mu=material[:,0];lam=material[:,:2].sum(-1)
    if not bool(torch.isfinite(f).all() & torch.isfinite(material).all() & torch.isfinite(measure).all()):
        raise ValueError('Finite deformation and material required')
    if bool(((mu<0)|(material[:,1]<0)|(measure<=0)).any()):raise ValueError('Nonnegative Lamé data and positive measure required')
    eigen=f.new_zeros(batch,rows,dimension)
    frame=torch.eye(dimension,device=f.device,dtype=f.dtype).expand(batch,rows,dimension,dimension)
    active=torch.where(lam>0)[0]
    if len(active):
        current=f[:,active];gram=current.transpose(-1,-2)@current
        inverse=torch.linalg.inv_ex(gram,check_errors=False)
        jacobian=torch.linalg.det(current) if dimension==3 else torch.linalg.cross(current[...,0],current[...,1]).norm(dim=-1)
        if bool((inverse.info!=0).any() | (jacobian<=0).any()) or not bool(torch.isfinite(inverse.inverse).all()):
            raise ValueError('Prestress needs nonsingular, positively oriented material geometry')
        pressure=lam[active]*(jacobian-1)-mu[active]
        stress=mu[None,active,None,None]*torch.eye(dimension,device=f.device,dtype=f.dtype)+(
            pressure*jacobian)[...,None,None]*inverse.inverse
        if not bool(torch.isfinite(stress).all()):raise ValueError('Nonfinite second Piola stress')
        values,vectors=so3.eigh_zero_safe((stress+stress.transpose(-1,-2))/2)
        eigen=eigen.index_copy(1,active,values.clamp_min(0));frame=frame.index_copy(1,active,vectors)
    m=torch.cat((-dm.sum(-2,keepdim=True),dm),-2)
    shape=torch.einsum('rsa,brap->brsp',m,frame)
    eye=torch.eye(3,device=f.device,dtype=f.dtype)
    jacobian=torch.einsum('brsp,ij->bripsj',shape,eye).flatten(2,3)
    weight=(eigen[:,:,None].expand(-1,-1,3,-1)*measure[None,:,None,None]).flatten(2,3)
    return torch.nn.functional.pad(jacobian,(0,3)),weight
