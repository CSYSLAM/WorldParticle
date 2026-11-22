"""Hierarchy augmentation of v19's audited direction-preserving response."""
from contextlib import nullcontext
import torch

from .heterogeneous_backbone import HeterogeneousBackbone,build_plan
from .vector_response import VectorInverseResponse,dimensionless_relations
from .vector_transport import InvariantVectorTransport


def _snapshot_scene(scene):
    # Scene exports replace topology rather than editing its index tensors.
    # Keep this call's physical components, but give neural caches a new owner.
    # MixedScene.__getattr__ makes copy.copy unsafe before values is installed.
    frozen=type(scene).__new__(type(scene))
    vars(frozen).update(vars(scene))
    frozen._hierarchy_plans={};frozen._hierarchy_material_masks={}
    frozen._component_batch_layout=None
    return frozen


def relation_features(scene,state,relation):
    value,rate=relation[...,:9],relation[...,18:27]
    layout=scene.layout;batch=len(value)
    count=value.new_zeros(len(layout.incidence));center=value.new_zeros(batch,len(count),3)
    for rows,ids in layout.slots:
        center=center.index_add(1,rows,state.x[:,ids]);count=count.index_add(0,rows,torch.ones_like(rows,dtype=value.dtype))
    center=center/count.clamp_min(1)[None,:,None]
    extent=value.new_zeros(batch,len(count))
    for rows,ids in layout.slots:
        extent=extent.index_add(1,rows,((state.x[:,ids]-center[:,rows])/scene.length).square().sum(-1))
    extent=extent/count.clamp_min(1)[None]
    features=torch.stack((value.square().sum(-1),rate.square().sum(-1),(value*rate).sum(-1),extent,
        relation[...,30],count[None].expand(batch,-1)/4,relation[...,27],torch.ones_like(extent)),-1)
    # Only the currently admitted physical backend uses this extractor.
    # New relation kinds require explicit masks and history-aware features.
    key=(id(scene),int(getattr(scene,'topology_generation',0)),value.dtype)
    if not hasattr(scene,'_hierarchy_material_masks'):scene._hierarchy_material_masks={}
    if key not in scene._hierarchy_material_masks:
        sizes=relation.new_tensor([3,3,2,6,6,1,5,5,5,5,5,6,6,6,6,1])
        scene._hierarchy_material_masks[key]=(torch.arange(9,device=value.device)[None]<sizes[layout.relation_kind,None]).to(value.dtype)
    mask=scene._hierarchy_material_masks[key]
    return features,mask[None].expand(batch,-1,-1)


class HierarchicalVectorResponse(VectorInverseResponse):
    def __init__(self,width=32,*,maximum_super_tokens=128,hierarchy_heads=4,
                 hierarchy_encoder_layers=3,hierarchy_decoder_layers=2,**kwargs):
        super().__init__(width,**kwargs)
        if hierarchy_encoder_layers<1 or hierarchy_decoder_layers<1:
            raise ValueError('Positive hierarchy encoder/decoder depths required')
        self.hierarchy=HeterogeneousBackbone(width,heads=hierarchy_heads,
            encoder_layers=hierarchy_encoder_layers,decoder_layers=hierarchy_decoder_layers,
            maximum_super_tokens=maximum_super_tokens)
        self.transport=InvariantVectorTransport(width)
        self.hierarchy_enabled=True

    def make_ports(self,scene,state,proposal,latent,relation,dt,*,unit_shapes=False):
        raw=super().make_ports(scene,state,proposal,latent,relation,dt,unit_shapes=unit_shapes)
        if unit_shapes or not self.hierarchy_enabled:return raw
        transported,gain=self.transport(scene,state,proposal,latent)
        addition=torch.zeros_like(raw)
        # Add to already active proposal columns. Introducing a brand-new zero
        # column would be killed by the rank mask and have no initial gradient.
        addition[...,:3,0]=transported[...,:3]*gain[...,:1]
        addition[...,3:,1]=transported[...,3:]*gain[...,1:]
        return raw+addition

    def encode(self,scene,state,physical,baseline,relation,dt):
        latent,features=self._encode_neural(scene,state,physical,baseline,relation,dt)
        if getattr(self,'shared_velocity_features',False) or getattr(self,'shared_material_features',False):self.last_latent=latent
        return latent,features

    def _encode_neural(self,scene,state,physical,baseline,relation,dt):
        self.relation_chunk_rows=self.hierarchy.relation_condition.checkpoint_rows
        if self.hierarchy.relation_condition.checkpoint_rows:
            self.hierarchy.relation_condition.checkpoint_inputs_on_cpu=bool(getattr(self,'checkpoint_inputs_on_cpu',False))
        for layer in (*self.hierarchy.encoder,*self.hierarchy.decoder):
            if layer.checkpoint_block:layer.checkpoint_inputs_on_cpu=bool(getattr(self,'checkpoint_inputs_on_cpu',False))
        prepacked=False
        if (getattr(self,'prepack_before_encode',True) and
            getattr(self,'compact_relations',False) and
            (not self.hierarchy_enabled or getattr(self.hierarchy,'compact_relations',False))):
            from .neural_relations import active_layout
            layout,selected=active_layout(scene.layout,relation[...,27])
            if len(selected)<relation.shape[1]:
                # Both encoders already discard exactly these inactive rows.
                # Do it before normalization and checkpoint input storage, so
                # a dense physical candidate matrix is not retained per round.
                # Never replace the caller's physical topology or components.
                scene=_snapshot_scene(scene)
                scene.layout=layout
                relation=relation[:,selected]
                prepacked=True
        if getattr(self,'encoder_checkpointing',False) and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            from .data import State
            # Physical steps replace contact topology and accepted history.
            # Backward must recompute against this invocation's snapshot.
            frozen=scene if prepacked else _snapshot_scene(scene)
            history=None if state.history is None else dict(state.history)
            def evaluate(x,q,v,omega,force,inverse,relations):
                current=State(x,q,v,omega,history)
                return self._encode(frozen,current,force,inverse,relations,dt)
            # Non-reentrant checkpointing saves its inputs too. Dense contact
            # relation inputs can dominate several unrolled steps even though
            # the encoder's intermediate activations are recomputed. Preserve
            # their exact dtype/values on host until backward needs them.
            storage=torch.autograd.graph.save_on_cpu(pin_memory=False) if getattr(self,'checkpoint_inputs_on_cpu',False) else nullcontext()
            with storage:
                return checkpoint(evaluate,state.x,state.q,state.v,state.omega,
                                  physical,baseline,relation,use_reentrant=False)
        return self._encode(scene,state,physical,baseline,relation,dt)

    def _encode(self,scene,state,physical,baseline,relation,dt):
        old,features=super().encode(scene,state,physical,baseline,relation,dt)
        if not self.hierarchy_enabled:return old,features
        generation=int(getattr(scene,'topology_generation',0))
        key=(id(scene),self.hierarchy.maximum_super_tokens,generation)
        if not hasattr(scene,'_hierarchy_plans'):scene._hierarchy_plans={}
        if key not in scene._hierarchy_plans:scene._hierarchy_plans[key]=build_plan(scene,self.hierarchy.maximum_super_tokens)
        normalized=dimensionless_relations(scene,relation,dt)
        rows,mask=relation_features(scene,state,normalized)
        dtype=old.dtype
        added=self.hierarchy(scene,features.to(dtype),rows.to(dtype),normalized[...,9:18].to(dtype),mask.to(dtype),normalized[...,27],plan=scene._hierarchy_plans[key])
        return old+added,features
