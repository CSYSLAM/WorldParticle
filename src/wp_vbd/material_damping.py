"""Newton's objective C=F^T F and dihedral-rate material damping.

The previous configuration is the accepted PHYSICAL step start. It never
changes between neural microiterations. These are material forces, distinct
from contact damping, numerical relaxation, and ALM penalties.
"""

from dataclasses import dataclass

import torch

from . import math3d as so3


@dataclass
class DampingTerm:
    ids: torch.Tensor
    jacobian: torch.Tensor
    weight: torch.Tensor


def metric_coordinate(scene,x,key):
    ids=getattr(scene,key)
    dm=scene.tet_Dm if key=='tets' else scene.tri_Dm
    p=x[:,ids];f=(p[:,:,1:]-p[:,:,:1]).transpose(-1,-2)@dm
    return (f.transpose(-1,-2)@f).flatten(-2),f


def metric_jacobian(scene,f,key):
    dm=scene.tet_Dm if key=='tets' else scene.tri_Dm
    m=torch.cat((-dm.sum(-2,keepdim=True),dm),-2)
    derivative=torch.einsum('bria,rsc->bracsi',f,m)
    derivative=derivative+derivative.transpose(2,3)
    return derivative.flatten(2,3)


def dihedral(points):
    n1=so3.normalize(torch.linalg.cross(points[:,:,2]-points[:,:,0],points[:,:,3]-points[:,:,0]))
    n2=so3.normalize(torch.linalg.cross(points[:,:,3]-points[:,:,1],points[:,:,2]-points[:,:,1]))
    edge=so3.normalize(points[:,:,3]-points[:,:,2])
    return torch.atan2((torch.linalg.cross(n1,n2)*edge).sum(-1),(n1*n2).sum(-1))


def damping_system(scene,x,previous,dt):
    residual=torch.zeros_like(x);blocks=x.new_zeros(*x.shape,3);terms=[]
    for enabled,key in zip(scene.material_damping_flags[:2],('tets','triangles')):
        if not enabled:continue
        ids=getattr(scene,key);value,f=metric_coordinate(scene,x,key)
        old,_=metric_coordinate(scene,previous,key)
        mat=scene.tet_material if key=='tets' else scene.tri_material
        measure=scene.tet_volume if key=='tets' else scene.tri_area
        coefficient=mat[:,2]*measure/dt
        jacobian=metric_jacobian(scene,f,key)
        force=torch.einsum('brcsi,brc,r->brsi',jacobian,value-old,coefficient)
        hessian=torch.einsum('brcsi,brcsj,r->brsij',jacobian,jacobian,coefficient)
        residual=residual.index_add(1,ids.flatten(),force.flatten(1,2))
        blocks=blocks.index_add(1,ids.flatten(),hessian.flatten(1,2))
        terms.append(DampingTerm(ids,torch.nn.functional.pad(jacobian.detach(),(0,3)),coefficient[None,:,None].expand(x.shape[0],-1,value.shape[-1])))
    if scene.material_damping_flags[2]:
        ids=scene.bend_edges[scene.bending_indices]
        enabled=torch.is_grad_enabled()
        with torch.enable_grad():
            points=x[:,ids]
            if not points.requires_grad:points=points.detach().requires_grad_(True)
            theta=dihedral(points)
            jacobian=torch.autograd.grad(theta.sum(),points,create_graph=enabled)[0]
        if not enabled:theta,jacobian=theta.detach(),jacobian.detach()
        old=dihedral(previous[:,ids])
        selected=scene.bending_indices
        coefficient=scene.bend_material[selected,1]*scene.bend_length[selected]/dt
        force=jacobian*(theta-old)[...,None,None]*coefficient[None,:,None,None]
        hessian=jacobian[...,None]*jacobian[...,None,:]*coefficient[None,:,None,None,None]
        residual=residual.index_add(1,ids.flatten(),force.flatten(1,2))
        blocks=blocks.index_add(1,ids.flatten(),hessian.flatten(1,2))
        terms.append(DampingTerm(ids,torch.nn.functional.pad(jacobian[:,:,None].detach(),(0,3)),coefficient[None,:,None].expand(x.shape[0],-1,-1)))
    return residual,blocks,terms


def damping_potential(scene,x,previous,dt):
    from .mixed_physics import energy_owner
    energy=x.new_zeros(x.shape[:2])
    for enabled,key in zip(scene.material_damping_flags[:2],('tets','triangles')):
        if not enabled:continue
        value,_=metric_coordinate(scene,x,key);old,_=metric_coordinate(scene,previous,key)
        mat=scene.tet_material if key=='tets' else scene.tri_material
        measure=scene.tet_volume if key=='tets' else scene.tri_area
        term=.5*mat[None,:,2]*measure[None]/dt*(value-old).square().sum(-1)
        energy=energy.index_add(1,energy_owner(scene,getattr(scene,key)),term)
    if scene.material_damping_flags[2]:
        selected=scene.bending_indices;ids=scene.bend_edges[selected]
        delta=dihedral(x[:,ids])-dihedral(previous[:,ids])
        term=.5*scene.bend_material[selected,1]*scene.bend_length[selected]/dt*delta.square()
        energy=energy.index_add(1,energy_owner(scene,ids),term)
    return energy
