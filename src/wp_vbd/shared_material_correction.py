"""Experimental shared-latent material amplitude, replacing the LAST solve.

Analytic predictor rounds retain their old material response. This module
uses one residual-vector H action and one proposal-vector projection; it
does not build a 24-column H image/energy Gram or solve a material system.
The material basis and the existing outer physical checks remain explicit.
"""
import math
import torch
from torch import nn

from .elastic_shape_modes import material_basis
from .protected_response import metric_action


MODE_COUNT=24
EXTRA_FEATURES=50


def apply_override(config,value,*,omit=False):
    if omit and value is not None:
        raise ValueError('Choose either omission or a learned/analytic material override')
    if omit:
        config['omit_final_material_correction']=True
    if value is not None:
        config['omit_final_material_correction']=False
        if value and not config.get('shared_material_correction',False):
            raise ValueError('Learned material deployment requires trained v31 head weights')
        config['shared_material_use_learned']=bool(value)


def statistics(model):
    return dict(enabled=bool(model is not None and getattr(model,'shared_material_head',None) is not None
            and model.config.get('shared_material_use_learned',True)
            and not model.config.get('omit_final_material_correction',False)),
        omitted_final_stage_calls=getattr(model,'_omitted_material_calls',0),
        scoped_analytic_final_stage_calls=getattr(model,'_scoped_analytic_material_calls',0),
        training_teacher_solves=getattr(model,'_material_training_teacher_solves',0),
        learned_final_stage_calls=getattr(model,'_shared_material_calls',0),
        analytic_reduced_solves=getattr(model,'_analytic_material_total_solves',0))


def teacher_loss(model,scene,state,metric,coarse,residual,proposal,predicted):
    """Training-only on-policy analytic label; never queried by eval/no-grad."""
    if not model.training or not torch.is_grad_enabled():
        raise RuntimeError('Material teacher labels are training-only')
    from .analytic_material_response import correct
    previous=getattr(scene,'_analytic_material_last_solves',0)
    with torch.no_grad():
        target,valid=correct(scene,state,metric,coarse,residual.detach(),proposal.detach(),
            maximum_columns=int(model.config.get('analytic_material_maximum_columns',96)),
            maximum_nodes=int(model.config.get('analytic_material_maximum_nodes',4096)),
            blocked=bool(model.config.get('analytic_material_block_solve',False)))
        calls=scene._analytic_material_last_solves
        scene._analytic_material_last_solves=previous
        if not bool(valid.all()):raise ValueError('Invalid on-policy material teacher')
        target_delta=target-proposal.detach()
        weights=scene.mass.double()[:,None]*proposal.new_tensor([1.,1.,1.,*([float(scene.length)**2]*3)])
        floor=scene.mass.double().sum()*(float(scene.length)*1e-6)**2
        denominator=(target_delta.square()*weights).sum((1,2))+floor
    model._material_training_teacher_solves=getattr(model,'_material_training_teacher_solves',0)+calls
    # Subtract the SAME differentiable proposal: supervision concerns the
    # learned addition, not the physical prefix that generated the proposal.
    error=predicted-proposal-target_delta
    return ((error.square()*weights).sum((1,2))/denominator).mean()


def object_inputs(scene,state,ids,latent,effective,update,whitening,baseline,dt):
    if not math.isfinite(dt) or dt<=0:raise ValueError('Positive material correction dt required')
    if latent is None or latent.shape[:2]!=state.x.shape[:2]:
        raise ValueError('Current shared backbone latent required for material correction')
    mass=scene.mass[ids].double();total=mass.sum()
    if not bool(torch.isfinite(mass).all() & (mass>0).all()):raise ValueError('Positive finite material masses required')
    # Freeze the geometric linearization, as in the analytic material solve.
    # The rigid Gram has repeated translation eigenvalues: differentiating
    # its arbitrary eigenvectors makes multi-step gradients nonfinite even
    # though the represented projector and forward result are well defined.
    with torch.no_grad():
        basis=material_basis(state.x[:,ids].double(),scene.rest_x[ids].double(),mass)
    inverse=whitening[:,ids].double()@baseline[:,ids].double()@whitening[:,ids].double()
    hint=-(inverse@effective[:,ids,:,None].double())[...,:3,0]
    proposal=update[:,ids,:3].double()
    scale=((mass[None]*(hint.square().sum(-1)+proposal.square().sum(-1))).sum(-1)/total).sqrt()
    scale=scale.clamp_min(float(scene.length)*1e-10)
    def coordinates(value):
        return torch.einsum('bnip,n,bni->bp',basis,mass,value)/(total*scale[:,None])
    pooled=(latent[:,ids]*mass.to(latent.dtype)[None,:,None]).sum(1)/total.to(latent.dtype)
    extras=torch.cat((coordinates(hint),coordinates(proposal),
        torch.log(scale/float(scene.length)).clamp(-24,8)[:,None],
        scale.new_full((len(scale),1),math.log(dt*180))),-1)
    features=torch.cat((pooled,extras.to(pooled.dtype)),-1)
    if not bool(torch.isfinite(features).all() & torch.isfinite(basis).all()):
        raise ValueError('Finite shared material features required')
    return features,basis,scale


class SharedMaterialCorrection(nn.Module):
    def __init__(self,width=648,*,hidden=256):
        super().__init__()
        if width<8 or hidden<8:raise ValueError('Positive shared material readout widths required')
        self.width=width
        self.latent_norm=nn.LayerNorm(width)
        self.register_buffer('feature_scale',torch.tensor([16.]*48+[1/12,.5],dtype=torch.float32))
        self.local_readout=nn.Linear(EXTRA_FEATURES,MODE_COUNT,bias=False)
        nn.init.zeros_(self.local_readout.weight)
        self.readout=nn.Sequential(nn.Linear(width+EXTRA_FEATURES,hidden),nn.SiLU(),
            nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,MODE_COUNT))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)

    def forward(self,features):
        value=features.to(self.readout[0].weight.dtype)
        extra=value[...,self.width:]*self.feature_scale
        value=torch.cat((self.latent_norm(value[...,:self.width]),extra),-1)
        return self.readout(value)+self.local_readout(extra)

    def correct(self,scene,state,metric,coarse,residual,update,whitening,baseline,latent,dt):
        if scene.metadata.get('contract')!='elastic_surface_joint_contact_v38':
            raise ValueError('v31 shared material correction currently validated only for surface/joint contracts')
        material=(scene.kind<2)&scene.movable
        delta=torch.zeros_like(update,dtype=torch.float64)
        valid=torch.ones(len(update),device=update.device,dtype=torch.bool)
        # Fixed-budget first-order response feature. The old postsolve also
        # evaluated this vector; no projected 24-column operator is formed.
        effective=residual.double()+metric_action(metric,update.double())
        for obj in torch.unique(scene.object_id[material]).tolist():
            ids=torch.where(material&(scene.object_id==obj))[0]
            if len(ids)<8:continue
            features,basis,scale=object_inputs(scene,state,ids,latent,effective,update,whitening,baseline,dt)
            coefficient=self(features).double()*scale[:,None]
            local=torch.einsum('bnip,bp->bni',basis,coefficient)
            local=torch.nn.functional.pad(local,(0,3))
            delta=delta.index_add(1,ids,local)
        # The existing protected component is preserved with ONE vector
        # action, rather than projecting every candidate material column.
        delta=delta-coarse(metric_action(metric,delta))
        delta=delta*scene.dof[None]
        valid&=torch.isfinite(delta).all((1,2))
        return update+torch.where(valid[:,None,None],delta,0.).to(update.dtype),valid
