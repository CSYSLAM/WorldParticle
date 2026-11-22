"""Bounded physical material correction after the shared response.

Solve the remaining quadratic in a rigid-free material basis, retaining
cross-object terms. No dynamics solver, learned direction, or dense global
inverse is used. Outer geometry/merit checks still admit the resulting step.
"""
import torch

from .elastic_shape_modes import polynomial_basis
from .protected_response import metric_action


def reduced_correction(columns,images,residual,*,rank_relative=1e-10):
    """Solve B^T H B c = -B^T r with scale-aware rank selection."""
    with torch.no_grad():
        gram=torch.einsum('bnic,bnid->bcd',columns,images)
        gram=(gram+gram.transpose(-1,-2))*.5
        eigen,frame=torch.linalg.eigh(gram)
        threshold=eigen.abs().amax(-1,keepdim=True).clamp_min(1e-30)*rank_relative
        valid=torch.isfinite(eigen).all(-1)&(eigen>=-100*threshold).all(-1)
        inverse=torch.where(eigen>threshold,eigen.clamp_min(1e-300).reciprocal(),0.)
        inverse=torch.where(valid[:,None],inverse,0.)
    rhs=torch.einsum('bnic,bni->bc',columns,residual)
    coefficient=-frame@(inverse[...,None]*(frame.transpose(-1,-2)@rhs[...,None]))
    value=torch.einsum('bnic,bc->bni',columns,coefficient[...,0])
    return value,valid


def correct(scene,state,metric,coarse,residual,update,*,maximum_columns=96,
            maximum_nodes=4096,blocked=False,observer=None):
    """One reduced solve per interacting component, after neural proposal."""
    if maximum_columns<24 or maximum_columns>192:
        raise ValueError('Material correction needs a bounded 24..192 column budget')
    if not 8<=maximum_nodes<=16384:raise ValueError('A bounded material node budget required')
    valid=torch.ones(len(update),device=update.device,dtype=torch.bool)
    scene._analytic_material_last_solves=0
    corrected=update.double()
    for component in scene.interaction_entities:
        material=component[(scene.kind[component]<2)&scene.movable[component]]
        if not len(material):continue
        objects=[material[scene.object_id[material]==obj] for obj in torch.unique(scene.object_id[material]).tolist()]
        objects=[ids for ids in objects if len(ids)>=8]
        if any(len(ids)>maximum_nodes for ids in objects):raise ValueError('Material correction exceeds the current node budget')
        per_block=maximum_columns//24
        if len(objects)>per_block and not blocked:
            raise ValueError('Interacting material correction exceeds its declared column budget')
        for offset in range(0,len(objects),per_block):
            basis_columns=[]
            with torch.no_grad():
                for ids in objects[offset:offset+per_block]:
                    basis=polynomial_basis(state.x[:,ids].detach().double(),scene.mass[ids].double())
                    columns=torch.zeros(len(update),scene.n,6,24,device=update.device,dtype=torch.float64)
                    local=torch.nn.functional.pad(basis,(0,0,0,3))
                    columns=columns.index_copy(1,ids,local)*scene.dof[None,:,:,None]
                    basis_columns.append(columns)
                columns=torch.cat(basis_columns,-1)
                columns=columns-coarse(metric_action(metric,columns))
                images=metric_action(metric,columns)
            # Refresh the residual after the preceding block, retaining its
            # coupling in the same quadratic. No extra object is discarded.
            effective=residual.double()+metric_action(metric,corrected)
            value,good=reduced_correction(columns[:,component],images[:,component],effective[:,component])
            if observer is not None:
                observer(component,objects[offset:offset+per_block],effective,value,good)
            scene._analytic_material_last_solves+=1
            corrected=corrected.index_add(1,component,value);valid=valid&good
    return corrected.to(update.dtype),valid
