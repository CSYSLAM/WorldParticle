"""Prototype VF/EE constitutive rows on four real particle owners.

Only local laws and a frozen geometry snapshot are defined here. Candidate
completeness, a collision path guard and solver/training admission are separate.
"""
from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch

from .contact_ports import PointPorts


def normal_potential(distance, margin, stiffness):
    """Pinned C2 log barrier with its finite quadratic continuation."""
    if not bool(torch.isfinite(distance).all() & torch.isfinite(margin).all() & torch.isfinite(stiffness).all()):
        raise ValueError('Finite self-contact geometry and material required')
    if bool((margin<0).any() | (stiffness<0).any()):raise ValueError('Nonnegative self-contact margin and stiffness required')
    tau=torch.where(margin>0,.5*margin,torch.ones_like(margin))
    minimum=torch.minimum(margin.new_full(margin.shape,1e-5),.5*tau)
    k2=tau.square()*stiffness
    safe=distance.clamp_min(minimum)
    logarithmic=.5*k2-k2*torch.log(safe/tau)
    at_min=.5*k2-k2*torch.log(minimum/tau)
    low=distance-minimum
    extension=at_min-k2/minimum*low+.5*k2/minimum.square()*low.square()
    energy=torch.where(distance>=tau,.5*stiffness*(margin-distance).square(),torch.where(distance>minimum,logarithmic,extension))
    gradient=torch.where(distance>=tau,stiffness*(distance-margin),torch.where(distance>minimum,-k2/safe,k2*(distance-2*minimum)/minimum.square()))
    hessian=torch.where(distance>=tau,stiffness,torch.where(distance>minimum,k2/safe.square(),k2/minimum.square()))
    active=(margin>0)&(distance<margin)
    return tuple(torch.where(active,value,0.) for value in (energy,gradient,hessian))


def safe_norm(vector):
    square=vector.square().sum(-1)
    return torch.where(square>0,torch.sqrt(torch.where(square>0,square,torch.ones_like(square))),0.)


def friction_factor(tangent,epsilon):
    distance=safe_norm(tangent)
    positive=torch.where(epsilon>0,epsilon,torch.ones_like(epsilon))
    coefficient=torch.where(distance>epsilon,1/distance.clamp_min(1e-30),(2-distance/positive)/positive)
    return torch.where(distance>0,coefficient,0.)


def friction_potential(tangent,epsilon):
    distance=safe_norm(tangent)
    positive=torch.where(epsilon>0,epsilon,torch.ones_like(epsilon))
    return torch.where(distance>epsilon,distance-epsilon/3,distance.square()/positive-distance.pow(3)/(3*positive.square()))


@dataclass
class SelfPointContacts:
    ports: PointPorts
    normal: torch.Tensor
    margin: torch.Tensor
    kind: torch.Tensor

    def __post_init__(self):
        count=len(self.ports.owners)
        if self.normal.shape!=(count,3) or self.margin.shape!=(count,) or self.kind.shape!=(count,):
            raise ValueError('Complete self-contact row geometry required')
        rigid_owner=self.ports.body_mask[self.ports.ids]&(self.ports.owners>=0)
        if self.kind.dtype!=torch.long or bool(self.ports.rigid.any() | rigid_owner.any()) or not bool(torch.isin(self.kind,self.kind.new_tensor([9,10])).all()):
            raise ValueError('Self-contact requires real particle VF=9 or EE=10 ports')
        if not bool(torch.isfinite(self.normal).all() & torch.isfinite(self.margin).all()) or bool((self.margin<0).any()):
            raise ValueError('Finite nonnegative self-contact margin required')
        if bool(((self.normal.square().sum(-1)-1).abs()>1e-8).any()):raise ValueError('Unit self-contact normal required')

    def geometry(self,state,body_com,*,jacobian=True):
        point,jacobian=self.ports.geometry(state,body_com,jacobian=jacobian)
        distance=((point[:,:,1]-point[:,:,0])*self.normal).sum(-1)
        return point,jacobian,distance


def begin(snapshot,start,body_com,dt,stiffness,damping,friction,friction_epsilon,*,material_start=None):
    if not math.isfinite(dt) or dt<=0 or not math.isfinite(friction_epsilon) or friction_epsilon<0:
        raise ValueError('Positive physical dt and finite friction smoothing required')
    for value in (stiffness,damping,friction):
        if value.shape!=snapshot.margin.shape or not bool(torch.isfinite(value).all()) or bool((value<0).any()):
            raise ValueError('Finite nonnegative per-row self-contact material required')
    previous=snapshot.geometry(start if material_start is None else material_start,body_com,jacobian=False)[0]
    return SimpleNamespace(start_point=previous,dt=dt,stiffness=stiffness,damping=damping,friction=friction,epsilon=friction_epsilon*dt)


def _row_values(snapshot,x,q,body_com,start_point,dt,stiffness,damping_coefficient,friction,epsilon,factorized,allowed_overlap):
    """Pure continuous rows: no query, history advance or metric-cache writes."""
    state=SimpleNamespace(x=x,q=q)
    point,jacobian,distance=snapshot.geometry(state,body_com,jacobian=not factorized);normal=snapshot.normal
    displacement=point[:,:,1]-start_point[:,:,1]-point[:,:,0]+start_point[:,:,0]
    normal_displacement=(displacement*normal).sum(-1)
    tangent=displacement-normal_displacement[...,None]*normal
    energy,gradient,hessian=normal_potential(distance,snapshot.margin,stiffness)
    physical_active=(snapshot.margin>0)&(distance<snapshot.margin)
    load=-gradient
    damping=damping_coefficient*physical_active*(normal_displacement/dt < -1e-6)
    coefficient=friction_factor(tangent,distance.new_tensor(epsilon))
    friction_weight=friction*load*coefficient
    endpoint_force=(load-damping*normal_displacement/dt)[...,None]*normal-friction_weight[...,None]*tangent
    force=torch.stack((-endpoint_force,endpoint_force),-2)
    residual=-snapshot.ports.pullback(jacobian,force)
    nn=normal[:,:,None]*normal[:,None,:]
    projector=torch.eye(3,device=distance.device,dtype=distance.dtype)-nn
    metric=(hessian+damping/dt)[...,None,None]*nn+friction_weight[...,None,None]*projector
    active=physical_active&((stiffness+damping_coefficient)>0)
    return {'point':point,'jacobian':jacobian,'distance':distance,'raw_depth':snapshot.margin-distance,
            'force':force,'residual':residual,'_metric':metric.detach(),'normal_load':load,'damping_active':damping,
            'tangent_displacement':tangent,'active':active,'normal_energy':energy,
            'valid':torch.isfinite(distance)&torch.where(allowed_overlap[None]>0,distance>=-allowed_overlap[None],distance>0),
            # Native self friction is direct regularized Coulomb, not an ALM
            # state. No numerical tangent dual is advanced for these rows.
            'tangent_force':torch.zeros_like(endpoint_force)}


def rows(snapshot,state,body_com,context):
    compact=getattr(context,'compact_frozen_contact_metric',False)
    factorized=compact and not snapshot.ports.has_rigid and not snapshot.ports.has_repeated_owners and state.q.dtype==state.x.dtype==body_com.dtype
    arguments=(snapshot,state.x,state.q,body_com,context.start_point,context.dt,
               context.stiffness,context.damping,context.friction,context.epsilon,factorized,
               getattr(context,'allowed_overlap',state.x.new_zeros(len(snapshot.kind))))
    if getattr(context,'contact_row_checkpointing',False) and torch.is_grad_enabled() and len(snapshot.kind):
        from torch.utils.checkpoint import checkpoint
        # Bind this round's tensors/scalars explicitly. The live context and
        # geometry oracle may have advanced by the time backward recomputes.
        result=checkpoint(_row_values,*arguments,use_reentrant=False,preserve_rng_state=False)
    else:result=_row_values(*arguments)
    jacobian=result['jacobian']
    metric_ports=getattr(snapshot,'metric_ports',snapshot.ports)
    result['metric_term']=metric_ports.metric_term(None if jacobian is None else jacobian.detach(),
                                                    result.pop('_metric'),compact=compact)
    return result


def merit(snapshot,state,body_com,context,frozen):
    point,_,distance=snapshot.geometry(state,body_com,jacobian=False);normal=snapshot.normal
    displacement=point[:,:,1]-context.start_point[:,:,1]-point[:,:,0]+context.start_point[:,:,0]
    dn=(displacement*normal).sum(-1);tangent=displacement-dn[...,None]*normal
    normal_energy=normal_potential(distance,snapshot.margin,context.stiffness)[0]
    damping=.5*frozen['damping_active']/context.dt*dn.clamp_max(0).square()
    friction=context.friction*frozen['normal_load']*friction_potential(tangent,distance.new_tensor(context.epsilon))
    return normal_energy+damping+friction
