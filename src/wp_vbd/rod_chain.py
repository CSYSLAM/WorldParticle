"""Current shear-compatible fields for an open rod chain.

This constructs kinematic coarse directions, not a dynamics solution. Edge
systems and the endpoint closure map are 3x3; translations use a prefix sum.
"""

import numpy as np
import torch


def curvature_fields(arc,rotations,mobile):
    """Reference scalar ports with curvature-supported extra directions.

    Straight assets retain exactly the original three fields. Cumulative
    turning angle provides a discontinuity-aware field at sharp turns, and a
    smooth higher-order field on smoothly curved rods. Rank selection and
    two-pass orthogonalization prevent nearly duplicate reference columns.
    """
    arc=np.asarray(arc,dtype=np.float64)
    base=np.stack((1-arc,arc,np.minimum(2*arc,2*(1-arc))),-1)*mobile[:,None]
    q=np.asarray(rotations,dtype=np.float64);x,y,z,w=q.T
    tangent=np.stack((2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)),-1)
    tangent/=np.linalg.norm(tangent,axis=-1,keepdims=True)
    turn=np.arctan2(np.linalg.norm(np.cross(tangent[:-1],tangent[1:]),axis=-1),(tangent[:-1]*tangent[1:]).sum(-1))
    if turn.sum()<1e-6:return base
    phase=np.r_[0.,np.cumsum(turn)]/turn.sum()
    output=[base[:,i] for i in range(3)];orthogonal=[]
    for column in output:
        residual=column.copy()
        for _ in range(2):
            for basis in orthogonal:residual-=np.dot(basis,residual)*basis
        if np.linalg.norm(residual)>1e-12:orthogonal.append(residual/np.linalg.norm(residual))
    for column in ((1-arc)*phase*mobile,arc*phase*mobile):
        residual=column.copy();original=np.linalg.norm(column)
        for _ in range(2):
            for basis in orthogonal:residual-=np.dot(basis,residual)*basis
        length=np.linalg.norm(residual)
        if length>1e-5*max(original,1e-12):
            orthogonal.append(residual/length);output.append(residual*(original/length))
    return np.stack(output,-1)


def curvature_flexure(arc,rotations,mobile):
    """A material-normal port where turning is nonuniform along the asset.

    The cumulative-turn defect vanishes on straight/uniformly curved assets
    and at the endpoints. On an asset with concentrated curvature it admits
    a local bend response alongside material spin, using two columns only.
    """
    arc=np.asarray(arc,dtype=np.float64)
    q=np.asarray(rotations,dtype=np.float64);x,y,z,w=q.T
    tangent=np.stack((2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)),-1)
    tangent/=np.linalg.norm(tangent,axis=-1,keepdims=True)
    turn=np.arctan2(np.linalg.norm(np.cross(tangent[:-1],tangent[1:]),axis=-1),(tangent[:-1]*tangent[1:]).sum(-1))
    if turn.sum()<1e-6:return None
    defect=np.r_[0.,np.cumsum(turn)]/turn.sum()-arc
    if np.max(np.abs(defect))<1e-4:return None
    return defect*np.asarray(mobile)


def description(component, order, pairs, fixed, arc, device):
    ids=component.cpu().tolist();order=list(map(int,order))
    lookup={tuple(sorted(pair)):index for index,pair in enumerate(pairs)}
    edges=[index for index,pair in enumerate(pairs) if pair[0] in ids and pair[1] in ids]
    if len(edges)!=len(order)-1 or not set(map(int,fixed)).issubset({order[0],order[-1]}):
        return None
    rows=[lookup[tuple(sorted((a,b)))] for a,b in zip(order[:-1],order[1:])]
    source={entity:i for i,entity in enumerate(order)}
    fields=np.concatenate((np.sin(np.pi*arc)[:,None,None]*np.eye(3)[None],np.sin(2*np.pi*arc)[:,None,None]*np.eye(3)[None]),-1)
    fields[0]=0.;fields[-1]=0.
    return {
        'rows':torch.tensor(rows,device=device),
        'forward':torch.tensor([pairs[row][0]==order[i] for i,row in enumerate(rows)],device=device),
        'in_order':torch.tensor([ids.index(entity) for entity in order],device=device),
        'from_order':torch.tensor([source[entity] for entity in ids],device=device),
        'arc':torch.as_tensor(arc,dtype=torch.float64,device=device),
        'corrections':torch.as_tensor(fields,dtype=torch.float64,device=device),
        'both':len(fixed)>1,
    }


def current_modes(plan, modes, metric):
    """Reconstruct free translations from current linear rod rows.

    Doubly pinned chains remove controllable endpoint mismatch through six
    angular correction fields. A regularized 3x3 closure solve handles the
    straight-chain axial null direction without a synchronizing eigenvalue
    check. Its geometry map is frozen in the backward pass, like the physical
    metric. Unresolved remainder is distributed along
    the chain and remains visible to the physical residual and energy guard.
    The separate axial field is preserved, since axial strain is physical.
    """
    chain=plan.chain;j=metric.jacobian[:,chain['rows'],:3].double()
    left=torch.where(chain['forward'][None,:,None,None],j[...,0,:],j[...,1,:])
    right=torch.where(chain['forward'][None,:,None,None],j[...,1,:],j[...,0,:])
    solved=torch.linalg.solve_ex(right[...,:3],-torch.cat((left[...,3:],right[...,3:]),-1),check_errors=False)
    valid=(solved.info==0).all(-1)&torch.isfinite(solved.result).flatten(1).all(-1)
    a,b=solved.result[...,:3],solved.result[...,3:]
    angular=modes[:,chain['in_order'],3:].double()
    if chain['both']:
        fields=chain['corrections'][None].expand(len(modes),-1,-1,-1)
        with torch.no_grad():
            closure_map=(a@fields[:,:-1]+b@fields[:,1:]).sum(1)
            gram=closure_map@closure_map.transpose(-1,-2)
            finite=torch.isfinite(gram).flatten(1).all(-1)
            safe=torch.where(finite[:,None,None],gram,torch.eye(3,device=modes.device,dtype=torch.float64))
            regularization=safe.diagonal(dim1=-2,dim2=-1).amax(-1).clamp_min(1e-30)*1e-10
            safe=safe+regularization[:,None,None]*torch.eye(3,device=modes.device,dtype=torch.float64)
            rhs=torch.where(finite[:,None,None],closure_map,0.)
            closure=torch.linalg.solve_ex(safe,rhs,check_errors=False)
            correction_map=closure.result.transpose(-1,-2)
        valid&=finite&(closure.info==0)&torch.isfinite(correction_map).flatten(1).all(-1)
        mismatch=(a@angular[:,:-1]+b@angular[:,1:]).sum(1)
        coefficient=correction_map@mismatch
        angular=angular-fields@coefficient[:,None]
    increments=a@angular[:,:-1]+b@angular[:,1:]
    translation=torch.cat((torch.zeros_like(increments[:,:1]),increments.cumsum(1)),1)
    if chain['both']:
        translation=translation-chain['arc'][None,:,None,None]*translation[:,-1:]
    reconstructed=torch.cat((translation,angular),-2)[:,chain['from_order']]
    axial=((plan.columns==4)&(plan.kind=='static'))[None,None,None,:]
    reconstructed=torch.where(axial,modes.double(),reconstructed)
    return reconstructed*metric.scene.dof[plan.ids][None,:,:,None],valid
