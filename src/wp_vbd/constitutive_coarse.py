"""Positive shared coarse response from the same cross-type latent field."""
import torch
from torch import nn


class ConstitutiveCoarseGain(nn.Module):
    """One invariant positive scalar per actual mobile interaction component.

    With Q^T H Q=I this scales QQ^T by a positive scalar, preserving the
    symmetric positive inverse and exact zero-residual root. It gives a
    complete six-DOF physical coarse space a learnable finite-budget response.
    The same head is used for particles, bodies and rods.
    """
    def __init__(self,width=32,*,scope='active_contact',protect_complete_response=False):
        super().__init__();self.readout=nn.Linear(width,1)
        if scope not in ('all','active_contact','active_contact_or_finite_joint'):raise ValueError('Unknown constitutive coarse scope')
        self.scope=scope
        self.protect_complete_response=protect_complete_response
        nn.init.zeros_(self.readout.weight);nn.init.zeros_(self.readout.bias)

    def forward(self,scene,latent,relation,*,complete_mask=None):
        gain=latent.new_ones(*latent.shape[:2])
        active=latent.new_zeros(*latent.shape[:2])
        if self.scope in ('active_contact','active_contact_or_finite_joint'):
            kinds=scene.layout.relation_kind
            contacts=(kinds>=7)&(kinds<=10)
            if self.scope=='active_contact_or_finite_joint' and (scene.metadata.get('contract')=='rigid_joint_contact_finite_v26' or scene.metadata.get('mixed_finite_joints',False)):
                contacts=contacts|(kinds==3)|(kinds==11)|(kinds==12)|(kinds==13)
            if self.scope=='active_contact_or_finite_joint' and scene.metadata.get('contract')=='rod_shape_contact_com_v40':
                contacts=contacts|(kinds==4)
            for rows,ids in scene.layout.slots:
                active=active.index_add(1,ids,(relation[:,rows,27]*contacts[rows]).to(active.dtype))
        if getattr(scene,'component_batch',False):
            from .component_batch import layout
            groups=layout(scene);logit=self.readout(groups.pool(latent));scalar=1+.5*logit.tanh()
            if self.scope!='all':
                selected=groups.sum(active[...,None])>0
                scalar=torch.where(selected,scalar,1.)
            result=scalar[:,groups.labels,0]
            if complete_mask is None:return result
            safe=.99+.01*torch.nn.functional.softsign(logit)
            if self.scope!='all':safe=torch.where(selected,safe,1.)
            return torch.where(complete_mask,safe[:,groups.labels,0],result)
        for ids in scene.interaction_entities:
            mobile=scene.movable[ids]
            pooled=(latent[:,ids]*mobile[None,:,None]).sum(1)/mobile.sum().clamp_min(1)
            logit=self.readout(pooled);scalar=1+.5*logit.tanh()
            if self.scope!='all':scalar=torch.where(active[:,ids].sum(-1,keepdim=True)>0,scalar,1.)
            value=scalar.expand(-1,len(ids))
            if complete_mask is not None:
                safe=.99+.01*torch.nn.functional.softsign(logit)
                if self.scope!='all':safe=torch.where(active[:,ids].sum(-1,keepdim=True)>0,safe,1.)
                value=torch.where(complete_mask[:,ids],safe.expand_as(value),value)
            gain=gain.index_copy(1,ids,value)
        return gain
