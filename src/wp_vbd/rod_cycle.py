"""Six closed-rod bending/twist modes in the linear shear nullspace.

This is reference-geometry construction. Each edge uses a 3x3 translation
block and closure uses one 3x6 least-squares system. No dynamics or teacher
state enters the construction; runtime coarse systems remain at most 6x6.
"""

import numpy as np


def cyclic_modes(points, order, joints, linear_jacobian):
    order=np.asarray(order,dtype=np.int64);points=np.asarray(points,dtype=np.float64)
    count=len(order)
    if count<8:raise ValueError('Cyclic modes require at least eight segments')
    lookup={tuple(sorted(pair)):index for index,pair in enumerate(joints)}
    edges=[]
    for i in range(count):
        left,right=int(order[i]),int(order[(i+1)%count])
        row=lookup[tuple(sorted((left,right)))];a=0 if joints[row][0]==left else 1;b=1-a
        jac=linear_jacobian[row]
        solved=np.linalg.solve(jac[:,b,:3],-np.concatenate((jac[:,a,:3],jac[:,a,3:],jac[:,b,3:]),axis=-1))
        edges.append((solved[:,:3],solved[:,3:6],solved[:,6:9]))
    def integrate(angular):
        translation=np.zeros_like(angular)
        for i,(transport,left,right) in enumerate(edges[:-1]):
            translation[i+1]=transport@translation[i]+left@angular[i]+right@angular[i+1]
        transport,left,right=edges[-1]
        closure=transport@translation[-1]+left@angular[-1]+right@angular[0]
        return translation,closure
    centered=points[order]-points[order].mean(0)
    _,_,axes=np.linalg.svd(centered,full_matrices=False);normal=axes[-1]
    winding=np.cross(centered,np.roll(centered,-1,axis=0)).sum(0)
    if np.dot(winding,normal)<0:normal=-normal
    tangent=np.roll(centered,-1,axis=0)-np.roll(centered,1,axis=0)
    tangent/=np.linalg.norm(tangent,axis=-1,keepdims=True)
    radial=np.cross(tangent,normal);radial/=np.linalg.norm(radial,axis=-1,keepdims=True)
    lengths=np.linalg.norm(np.roll(centered,-1,axis=0)-centered,axis=-1)
    phase=2*np.pi*np.r_[0.,np.cumsum(lengths[:-1])]/lengths.sum()
    cosine,sine=np.cos(2*phase),np.sin(2*phase)
    angular=np.stack((-2*sine[:,None]*radial,2*cosine[:,None]*radial,1.5*sine[:,None]*normal,-1.5*cosine[:,None]*normal,cosine[:,None]*tangent,sine[:,None]*tangent),axis=-1)
    # First-harmonic correction fields span the three closure conditions on a
    # ring. They vanish from an already closed seed mode up to roundoff.
    corrections=np.concatenate((np.cos(phase)[:,None,None]*np.eye(3)[None],np.sin(phase)[:,None,None]*np.eye(3)[None]),axis=-1)
    _,closure_map=integrate(corrections);_,mismatch=integrate(angular)
    coefficient=np.linalg.lstsq(closure_map,-mismatch,rcond=1e-12)[0]
    angular+=corrections@coefficient
    translation,remaining=integrate(angular)
    if np.max(np.abs(remaining))>1e-8*max(1.,lengths.sum()):raise ValueError('Cyclic mode closure is unresolved')
    translation-=translation.mean(0)
    basis=np.concatenate((translation,angular),axis=1)
    # The supplied source Jacobian independently checks every closed edge.
    residual=[]
    by_id={int(entity):i for i,entity in enumerate(order)}
    for i in range(count):
        row=lookup[tuple(sorted((int(order[i]),int(order[(i+1)%count]))))]
        pair=np.stack([basis[by_id[int(entity)]] for entity in joints[row]])
        residual.append(np.einsum('csi,sik->ck',linear_jacobian[row],pair))
    error=float(np.max(np.abs(residual)))
    if error>1e-8:raise ValueError(f'Cyclic modes violate their linear shear contract: {error}')
    frame_ids=order[[0,count//4,count//2,3*count//4]]
    frame=frame_from_points(points[frame_ids])
    return basis,frame_ids,frame,error,corrections


def frame_from_points(points):
    u=points[0]-points[2];u/=np.linalg.norm(u)
    v=points[1]-points[3];v-=np.dot(u,v)*u;v/=np.linalg.norm(v)
    return np.stack((u,v,np.cross(u,v)),axis=-1)
