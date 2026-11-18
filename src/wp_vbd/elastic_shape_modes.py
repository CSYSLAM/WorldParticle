"""Residual-adaptive smooth material modes for a bounded shared response.

The existing rigid/tree space cannot represent compression or panel bending.
This constructs a small polynomial material space, removes rigid motion in
the mass metric, and compresses its physical quadratic response to one column
per material object. The shared neural coarse head still supplies its gain.
No forces, state constraints, neural parameters or acceptance rules change.
"""
import torch

from .coarse_response import rigid_component_modes
from .protected_response import metric_action


def _pseudoinverse_solve(matrix,rhs):
    eigen,frame=torch.linalg.eigh((matrix+matrix.transpose(-1,-2))*.5)
    threshold=eigen.abs().amax(-1,keepdim=True).clamp_min(1e-30)*1e-10
    if not bool(torch.isfinite(eigen).all()) or bool((eigen < -100*threshold).any()):
        raise ValueError('Elastic shape response requires finite positive semidefinite curvature')
    good=eigen>threshold
    coefficients=frame.transpose(-1,-2)@rhs[...,None]
    return (frame@(coefficients*torch.where(good,eigen.clamp_min(1e-300).reciprocal(),0.)[...,None]))[...,0]


def polynomial_basis(points,mass):
    """Six affine strains plus eighteen quadratic displacements, rigid-free."""
    center=(points*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
    radius=((points-center).square().sum(-1)*mass).sum(-1).div(mass.sum()).sqrt().clamp_min(1e-8)
    coordinate=(points-center)/radius[:,None,None]
    identity=torch.eye(3,dtype=points.dtype,device=points.device)
    columns=[]
    for a,b in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
        value=coordinate[...,a,None]*identity[b]
        if a!=b:value=value+coordinate[...,b,None]*identity[a]
        columns.append(value)
    for a,b in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
        scalar=coordinate[...,a]*coordinate[...,b]
        scalar=scalar-(scalar*mass).sum(-1,keepdim=True)/mass.sum()
        for axis in range(3):columns.append(scalar[...,None]*identity[axis])
    basis=torch.stack(columns,-1)
    dof=torch.ones(len(mass),6,dtype=torch.bool,device=mass.device)
    rigid=rigid_component_modes(points,center,dof)[...,:3,:]
    gram=torch.einsum('bnip,n,bniq->bpq',rigid,mass,rigid)
    cross=torch.einsum('bnip,n,bniq->bpq',rigid,mass,basis)
    solution=torch.stack([_pseudoinverse_solve(gram,cross[...,i]) for i in range(basis.shape[-1])],-1)
    basis=basis-torch.einsum('bnip,bpq->bniq',rigid,solution)
    norm=torch.einsum('bnip,n,bnip->bp',basis,mass,basis).div(mass.sum()).sqrt()
    return basis/norm[:,None,None].clamp_min(1e-12)


@torch.no_grad()
def response_columns(scene,state,metric,residual,ids):
    """One nonrigid physical response column per actual material object."""
    material=ids[(scene.kind[ids]<2)&scene.movable[ids]]
    columns=[];details=[]
    for object_id in torch.unique(scene.object_id[material]).tolist():
        selected=material[scene.object_id[material]==object_id]
        if len(selected)<8:continue
        # Bound this prototype's preprocessing; large objects need a sparse
        # basis implementation rather than an implicit unbounded allocation.
        if len(selected)>4096:raise ValueError('Elastic shape mode object exceeds 4096-node budget')
        basis=polynomial_basis(state.x[:,selected],scene.mass[selected])
        local=residual.new_zeros(len(residual),len(selected),6,basis.shape[-1]);local[...,:3,:]=basis
        full=residual.new_zeros(len(residual),scene.n,6,basis.shape[-1]).index_copy(1,selected,local)
        image=metric_action(metric,full)
        gram=torch.einsum('bnip,bniq->bpq',full,image)
        rhs=torch.einsum('bnip,bni->bp',full,residual)
        coefficients=_pseudoinverse_solve(gram,rhs)
        value=torch.einsum('bnip,bp->bni',local,coefficients)
        rms=((value[...,:3].square().sum(-1)*scene.mass[selected]).sum(-1)/scene.mass[selected].sum()).sqrt()
        active=torch.isfinite(value).flatten(1).all(-1)&(rms>1e-14)
        if not bool(active.any()):continue
        value=torch.where(active[:,None,None],value/rms[:,None,None].clamp_min(1e-30),0.)
        columns.append(residual.new_zeros(len(residual),scene.n,6).index_copy(1,selected,value))
        details.append({'object_id':object_id,'nodes':len(selected),'candidate_basis_columns':basis.shape[-1],
                        'quadratic_response_rms_m':rms.cpu().tolist()})
    previous=getattr(scene,'elastic_shape_mode_diagnostic',{})
    previous.update({row['object_id']:row for row in details});scene.elastic_shape_mode_diagnostic=previous
    return torch.stack(columns,-1) if columns else residual.new_zeros(len(residual),scene.n,6,0)
