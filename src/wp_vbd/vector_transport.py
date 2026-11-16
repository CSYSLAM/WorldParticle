"""Invariant attention transports polar/axial carriers to remote DOFs.

Scalar hierarchical conditioning alone cannot create a direction where all
local carriers vanish. This full-resolution bypass supplies such directions;
it never directly adds a displacement to the physical update. The caller
projects its columns and constructs the same bounded positive inverse.
"""
import math

import torch
from torch import nn

from .protected_response import centers


class InvariantVectorTransport(nn.Module):
    def __init__(self,width=32,key_width=8,tile=128):
        super().__init__()
        self.norm=nn.LayerNorm(width)
        self.query=nn.Linear(width,key_width,bias=False)
        self.key=nn.Linear(width,key_width,bias=False)
        self.gain=nn.Linear(width,2)
        nn.init.zeros_(self.gain.weight);nn.init.zeros_(self.gain.bias)
        self.log_bandwidth=nn.Parameter(torch.zeros(()))
        self.key_width,self.tile=key_width,tile

    def forward(self,scene,state,proposal,latent):
        generation=int(getattr(scene,'topology_generation',0))
        key=(id(scene),generation)
        if not hasattr(scene,'_transport_plans'):scene._transport_plans={}
        if key not in scene._transport_plans:
            scene._transport_plans[key]=tuple(ids[scene.movable[ids]] for ids in scene.interaction_entities if bool(scene.movable[ids].any()))
        position=centers(scene,state).to(latent.dtype)/scene.length
        hidden=self.norm(latent);q=self.query(hidden);k=self.key(hidden)
        values=proposal.to(latent.dtype)
        out=torch.zeros_like(values)
        bandwidth=.25+.75*self.log_bandwidth.sigmoid()
        batched=bool(getattr(scene,'component_batch',False))
        if batched:
            bucket_key=(*key,'component_buckets')
            if bucket_key not in scene._transport_plans:
                buckets={}
                for ids in scene._transport_plans[key]:buckets.setdefault(len(ids),[]).append(ids)
                scene._transport_plans[bucket_key]=tuple(torch.stack(ids) for ids in buckets.values())
            for ids in scene._transport_plans[bucket_key]:
                context=k[:,ids];source=values[:,ids];pos=position[:,ids]
                for start in range(0,ids.shape[1],self.tile):
                    selected=ids[:,start:start+self.tile]
                    distance=(position[:,selected][:,:,:,None]-pos[:,:,None]).square().sum(-1)
                    scores=q[:,selected]@context.transpose(-1,-2)/math.sqrt(self.key_width)-distance/(2*bandwidth.square())
                    transported=scores.softmax(-1)@source
                    out=out.index_copy(1,selected.flatten(),transported.flatten(1,2).to(out.dtype))
        for ids in (() if batched else scene._transport_plans[key]):
            context=k[:,ids];source=values[:,ids];pos=position[:,ids]
            for start in range(0,len(ids),self.tile):
                selected=ids[start:start+self.tile]
                distance=(position[:,selected,None]-pos[:,None]).square().sum(-1)
                scores=q[:,selected]@context.transpose(-1,-2)/math.sqrt(self.key_width)-distance/(2*bandwidth.square())
                transported=scores.softmax(-1)@source
                out=out.index_copy(1,selected,transported.to(out.dtype))
        gain=.5*self.gain(hidden).tanh()
        return out.to(proposal.dtype),gain.to(proposal.dtype)
