"""Certify F64 rest-volume roundoff against an independent determinant."""
import numpy as np
import torch


def certify_tet_volume_roundoff(current, recorded, inverse_rest):
    if current.dtype!=torch.float64 or recorded.dtype!=torch.float64 or inverse_rest.dtype!=torch.float64:
        raise ValueError('Only F64 derived geometry is eligible for restoration')
    if current.shape!=recorded.shape or inverse_rest.shape!=(*current.shape,3,3) or current.ndim!=1:
        raise ValueError('Matched tetrahedron rest geometry required')
    if not all(bool(torch.isfinite(x).all()) for x in (current,recorded,inverse_rest)):
        raise ValueError('Nonfinite derived geometry')
    if bool((current<=0).any()|(recorded<=0).any()):raise ValueError('Positive rest volumes required')
    if not current.numel():return {'rows':0,'maximum_absolute_difference':0.,'maximum_relative_difference':0.}
    dm=inverse_rest.detach().cpu().numpy().astype(np.longdouble)
    positive=np.stack((dm[:,0,0]*dm[:,1,1]*dm[:,2,2],dm[:,0,1]*dm[:,1,2]*dm[:,2,0],dm[:,0,2]*dm[:,1,0]*dm[:,2,1]),-1)
    negative=np.stack((dm[:,0,2]*dm[:,1,1]*dm[:,2,0],dm[:,0,0]*dm[:,1,2]*dm[:,2,1],dm[:,0,1]*dm[:,1,0]*dm[:,2,2]),-1)
    determinant=np.abs(positive.sum(-1)-negative.sum(-1))
    permanent=np.abs(positive).sum(-1)+np.abs(negative).sum(-1)
    epsilon=np.longdouble(np.finfo(np.float64).eps)
    bound=64*epsilon*permanent
    if np.any(~np.isfinite(determinant)) or np.any(determinant<=bound):
        raise ValueError('Ill-conditioned rest determinants cannot be restored as F64 roundoff')
    lower=1/(6*(determinant+bound));upper=1/(6*(determinant-bound))
    for tensor in (current,recorded):
        value=tensor.detach().cpu().numpy().astype(np.longdouble)
        slack=8*epsilon*upper
        if np.any(value<lower-slack) or np.any(value>upper+slack):
            raise ValueError('Rest volume differs beyond certified F64 arithmetic roundoff')
    difference=(current-recorded).abs()
    return {'rows':current.numel(),'maximum_absolute_difference':float(difference.max()),
            'maximum_relative_difference':float((difference/recorded).max()),
            'determinant_permanent_epsilon_multiplier':64,
            'scope':'Independent long-double six-product determinant interval; geometry/material changes remain forbidden.'}
