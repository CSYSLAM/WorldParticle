"""Typed, conditioned hierarchy with a full-resolution hyperedge bypass.

This is a small WorldParticle research variant, not an upstream checkpoint.
Only latent tokens merge; physical DOFs, geometry and history remain intact.
Topology plans belong to one potential-interaction graph generation. Dynamic
contact callers must rebuild the plan before capture when that graph changes.
"""
from dataclasses import dataclass
from enum import IntEnum
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F


class RelationKind(IntEnum):
    TET=0
    MEMBRANE=1
    BENDING=2
    JOINT=3
    ROD=4
    ATTACHMENT=5
    PLANE_CONTACT=6
    RIGID_CONTACT=7
    PARTICLE_SHAPE_CONTACT=8
    VERTEX_FACE_CONTACT=9
    EDGE_EDGE_CONTACT=10
    JOINT_DRIVE=11
    JOINT_LIMIT=12
    MIMIC=13
    EXTERNAL_EXCHANGE=14
    SPRING=15


def mlp(a, width, b):
    return nn.Sequential(nn.Linear(a, width), nn.SiLU(), nn.Linear(width, b))


class _Permutation(torch.autograd.Function):
    """Unique per-batch gather with inverse-gather backward.

    PyTorch 2.7's deterministic generic GatherBackward uses a scatter path
    containing a host-to-device copy, which cannot be CUDA-graph captured.
    No reductions are necessary for a permutation.
    """
    @staticmethod
    def forward(ctx,value,order):
        ctx.save_for_backward(order.argsort(dim=1))
        return value.gather(1,order[...,None].expand_as(value))

    @staticmethod
    def backward(ctx,gradient):
        inverse,=ctx.saved_tensors
        return gradient.gather(1,inverse[...,None].expand_as(gradient)),None


class _TieAverage(torch.autograd.Function):
    """Weighted tie-run average with a fixed-selection prefix backward."""
    @staticmethod
    def forward(ctx,value,sizes,lower,upper):
        def run_sum(source):
            prefix=torch.cat((torch.zeros_like(source[:,:1]),source.double().cumsum(1)),1)
            return prefix.gather(1,upper[...,None].expand_as(source))-prefix.gather(1,lower[...,None].expand_as(source))
        total_mass=run_sum(sizes);tied=(upper-lower)[...,None]>1
        averaged=run_sum(value.double()*sizes.double())/total_mass
        out=torch.where(tied,averaged.to(value.dtype),value)
        mass=torch.where(tied,(total_mass/(upper-lower)[...,None]).to(sizes.dtype),sizes)
        ctx.save_for_backward(sizes.double()/total_mass,lower,upper,tied)
        ctx.mark_non_differentiable(mass);ctx.set_materialize_grads(False)
        return out,mass

    @staticmethod
    def backward(ctx,gradient,unused_mass_gradient):
        if gradient is None:return None,None,None,None
        ratio,lower,upper,tied=ctx.saved_tensors
        prefix=torch.cat((torch.zeros_like(gradient[:,:1]),gradient.double().cumsum(1)),1)
        total=prefix.gather(1,upper[...,None].expand_as(gradient))-prefix.gather(1,lower[...,None].expand_as(gradient))
        return torch.where(tied,(ratio*total).to(gradient.dtype),gradient),None,None,None


class ConditionalFFN(nn.Module):
    def __init__(self, width, condition):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.film = mlp(condition, width, 2*width)
        self.ffn = mlp(width, 2*width, width)
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)
        self.checkpoint_rows=0
        self.checkpoint_inputs_on_cpu=False

    def forward(self, value, condition):
        rows=self.checkpoint_rows
        if rows and value.ndim==3 and value.shape[1]>rows:
            from torch.utils.checkpoint import checkpoint
            pieces=[]
            for first in range(0,value.shape[1],rows):
                # Host input staging restores contiguous CUDA tensors. Use
                # that layout in the initial call too, so Linear dispatch and
                # checkpoint saved-tensor ordering match during replay.
                v=value[:,first:first+rows].contiguous();c=condition[:,first:first+rows].contiguous()
                if self.training and torch.is_grad_enabled():
                    # Nested checkpoints keep outer encoder recomputation
                    # from retaining every row's wide FiLM/FFN intermediates.
                    storage=torch.autograd.graph.save_on_cpu(pin_memory=False) if self.checkpoint_inputs_on_cpu else nullcontext()
                    with storage:part=checkpoint(self._forward,v,c,use_reentrant=False)
                else:part=self._forward(v,c)
                pieces.append(part)
            return torch.cat(pieces,1)
        return self._forward(value,condition)

    def _forward(self,value,condition):
        gain, bias = self.film(condition).chunk(2, -1)
        return value + self.ffn((1+.1*gain.tanh())*self.norm(value)+.1*bias.tanh())


class AttentionBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        if width % heads:raise ValueError('Head width must divide latent width')
        self.width, self.heads = width, heads
        self.query_norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.query = nn.Linear(width, width)
        self.key_value = nn.Linear(width, 2*width)
        self.output = nn.Linear(width, width)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = mlp(width, 2*width, width)
        self.checkpoint_block=False
        self.checkpoint_inputs_on_cpu=False

    def forward(self, query, context, sizes):
        replay=getattr(self,'_attention_replay',None)
        if replay is not None:return replay(query,context,sizes)
        if self.checkpoint_block and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            # Host staging canonicalizes layout. Match it before the first
            # call as well, retaining checkpoint metadata verification.
            query=query.contiguous();context=context.contiguous();sizes=sizes.contiguous()
            storage=torch.autograd.graph.save_on_cpu(pin_memory=False) if self.checkpoint_inputs_on_cpu else nullcontext()
            with storage:return checkpoint(self._eager_attention,query,context,sizes,use_reentrant=False)
        return self._eager_attention(query,context,sizes)

    def _eager_attention(self, query, context, sizes):
        def heads(value):
            return value.reshape(len(value),-1,self.heads,self.width//self.heads).transpose(1,2)
        q = heads(self.query(self.query_norm(query)))
        k, v = self.key_value(self.context_norm(context)).chunk(2,-1)
        # Multiplicity is retained after merging; no scene-wide normalization.
        value = F.scaled_dot_product_attention(q, heads(k), heads(v), attn_mask=sizes.log().transpose(1,2)[:,None])
        value = value.transpose(1,2).reshape_as(query)
        updated = query + self.output(value)
        return updated + self.ffn(self.ffn_norm(updated))


@dataclass(frozen=True)
class HierarchyComponent:
    ids: torch.Tensor
    groups: tuple[torch.Tensor, ...]
    merge_rounds: int
    coarse_count: int


@dataclass(frozen=True)
class HierarchyPlan:
    components: tuple[HierarchyComponent, ...]
    type_ids: tuple[torch.Tensor, ...]
    generation: int
    maximum_super_tokens: int
    topology_owner: int
    buckets: tuple[tuple[int, ...], ...] = ()


def build_plan(graph, maximum_super_tokens=128):
    if maximum_super_tokens < 1:raise ValueError('Positive token budget required')
    components=[];buckets={}
    # Component labels are only partition identifiers, never numeric features.
    for label in torch.unique(graph.interaction_id, sorted=True).tolist():
        ids=torch.where((graph.interaction_id==label)&graph.movable)[0]
        if not len(ids):continue
        keys={}
        # Partition metadata is discrete. Transfer it together instead of
        # synchronizing twice per entity when the graph lives on CUDA. The
        # ascending ids, sorted keys and local member order remain identical.
        metadata=torch.stack((graph.object_id[ids],graph.kind[ids]),-1).cpu().tolist()
        for local, (object_id, kind) in enumerate(metadata):
            key=(object_id,kind)
            keys.setdefault(key,[]).append(local)
        groups=tuple(torch.tensor(value,device=ids.device) for _,value in sorted(keys.items()))
        counts=[len(group) for group in groups];rounds=0
        while sum(counts)>maximum_super_tokens and any(n>1 for n in counts):
            counts=[(n+1)//2 for n in counts];rounds+=1
        # Many single-DOF objects can exceed the soft budget. Retain every
        # object and report the actual count rather than silently dropping it.
        components.append(HierarchyComponent(ids,groups,rounds,sum(counts)))
        signature=(len(ids),tuple(tuple(value) for _,value in sorted(keys.items())),rounds)
        buckets.setdefault(signature,[]).append(len(components)-1)
    return HierarchyPlan(tuple(components),tuple(torch.where((graph.kind==kind)&graph.movable)[0] for kind in range(4)),int(getattr(graph,'topology_generation',0)),maximum_super_tokens,id(graph),tuple(tuple(indices) for indices in buckets.values()))


def merge_group(value, sizes, rounds):
    """Bounded-memory balanced pairing, selected by current invariant latent.

    Ordering/selection stop gradients. A fixed scalar projection avoids an
    N-by-N matching matrix; learned features change the assignment over time.
    Equal-score runs use their weighted mean and redistribute multiplicity
    evenly, so their internal ordering cannot encode arbitrary node indices.
    """
    width=value.shape[-1]
    projection=torch.sin(torch.arange(1,width+1,device=value.device,dtype=value.dtype))
    for _ in range(rounds):
        if value.shape[1]<=1:break
        with torch.no_grad():
            scores=(value*projection).sum(-1)
            order=scores.argsort(dim=1,stable=True)
            sorted_scores=scores.gather(1,order).contiguous()
            lower=torch.searchsorted(sorted_scores,sorted_scores,right=False)
            upper=torch.searchsorted(sorted_scores,sorted_scores,right=True)
        value=_Permutation.apply(value,order)
        sizes=_Permutation.apply(sizes,order)
        value,sizes=_TieAverage.apply(value,sizes,lower,upper)
        pairs=value.shape[1]//2
        a,b=value[:,:2*pairs:2],value[:,1:2*pairs:2]
        sa,sb=sizes[:,:2*pairs:2],sizes[:,1:2*pairs:2]
        total=sa+sb;merged=(a*sa+b*sb)/total
        value=torch.cat((merged,value[:,2*pairs:]),1)
        sizes=torch.cat((total,sizes[:,2*pairs:]),1)
    return value,sizes


def _relation_member_values(h,slots,row_count,chunk_rows):
    members=h.new_zeros(len(h),row_count,h.shape[-1])
    for rows,ids in slots:
        if chunk_rows and len(rows):
            for begin in range(0,len(rows),chunk_rows):
                selected=rows[begin:begin+chunk_rows]
                members.index_add_(1,selected,h[:,ids[begin:begin+chunk_rows]])
        else:members=members.index_add(1,rows,h[:,ids])
    return members


class _BoundedRelationMembers(torch.autograd.Function):
    """A sparse linear map needs its indices, not gathered feature values."""
    @staticmethod
    def forward(ctx,h,slots,row_count,chunk_rows,dense_forward):
        ctx.input_shape=tuple(h.shape);ctx.chunk_rows=chunk_rows
        ctx.save_for_backward(*(value for slot in slots for value in slot))
        return _relation_member_values(h,slots,row_count,0 if dense_forward else chunk_rows)

    @staticmethod
    def backward(ctx,gradient):
        result=gradient.new_zeros(ctx.input_shape)
        positions=ctx.saved_tensors
        slots=tuple(zip(positions[::2],positions[1::2]))
        # Reverse the original ordered accumulation. Repeated node IDs use
        # its sparse transpose; every feature/slot participates exactly once.
        for rows,ids in reversed(slots):
            starts=tuple(range(0,len(rows),ctx.chunk_rows)) or (0,)
            for begin in reversed(starts):
                selected=rows[begin:begin+ctx.chunk_rows]
                result.index_add_(1,ids[begin:begin+ctx.chunk_rows],gradient[:,selected])
        return result,None,None,None,None


def relation_members(h,slots,row_count,*,chunk_rows=0):
    """Keep original ordered slot sums and an exact sparse-transpose VJP.

    Positive bounds limit both forward and backward gather temporaries. Only
    incidence indices are saved; no full gathered feature copies survive.
    The default retains the original dense autograd execution path.
    """
    if chunk_rows:return _BoundedRelationMembers.apply(h,slots,row_count,chunk_rows,False)
    return _relation_member_values(h,slots,row_count,0)


def relation_messages(row,slots,node_count,active,*,chunk_rows):
    """Original per-slot message reductions; retain only incidence indices.

    Forward keeps whole-slot index_add order for repeated destination nodes.
    The sparse-transpose backward limits its gathers by the same row bound.
    """
    masked=torch.where(active[...,None].bool(),row,0.)
    positions=tuple((ids,rows) for rows,ids in slots)
    return _BoundedRelationMembers.apply(masked,positions,node_count,chunk_rows,True)


class HeterogeneousBackbone(nn.Module):
    """Four DOF tokenizers, shared FiLM/attention and typed latent readout.

    Relation kind IDs include reserved rows for rigid/particle contacts,
    vertex-face, edge-edge, joint drives and external exchange. Supplying
    their features does not implement their physical/history operators.
    """
    def __init__(self,width=32,heads=4,encoder_layers=3,decoder_layers=2,maximum_super_tokens=128,relation_kinds=16):
        super().__init__()
        self.maximum_super_tokens=maximum_super_tokens
        self.tokenizers=nn.ModuleList([mlp(28,width,width) for _ in range(4)])
        self.material=ConditionalFFN(width,18)
        self.relation_kind=nn.Embedding(relation_kinds,width)
        self.relation_state=mlp(8,width,width)
        self.relation_members=nn.Linear(width,width,bias=False)
        self.relation_condition=ConditionalFFN(width,18)
        self.local=mlp(2*width,width,width)
        self.encoder=nn.ModuleList([AttentionBlock(width,heads) for _ in range(encoder_layers)])
        self.decoder=nn.ModuleList([AttentionBlock(width,heads) for _ in range(decoder_layers)])
        self.readout=nn.ModuleList([nn.Linear(width,width) for _ in range(4)])
        for layer in self.readout:
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self.width=width

    def _relation_chunk(self,h,features,material,mask,active,kinds,slots,movable):
        count=features.shape[1]
        row=self.relation_state(features)+self.relation_kind(kinds)[None]
        members=relation_members(h,slots,count,chunk_rows=self.relation_condition.checkpoint_rows)
        denominator=h.new_zeros(count,1)
        for rows,ids in slots:
            denominator=denominator.index_add(0,rows,movable[ids,None].to(h.dtype))
        row=row+self.relation_members(members/denominator.clamp_min(1)[None])
        row=self.relation_condition(row,torch.cat((material,mask),-1))
        return relation_messages(row,slots,h.shape[1],active,chunk_rows=self.relation_condition.checkpoint_rows)

    def stream_relation_messages(self,h,features,material,mask,active,kinds,slots,movable):
        """Checkpoint complete independent relation rows before node reduction.

        No full R-by-width relation/member matrix is materialized. Each pure
        chunk returns a node message, then releases its wide row values.
        FP32 node reduction association can differ from the dense schedule.
        """
        from torch.utils.checkpoint import checkpoint
        bound=self.relation_condition.checkpoint_rows
        if bound<=0:raise ValueError('Positive complete relation chunk bound required')
        message=torch.zeros_like(h);count=features.shape[1];device=h.device
        node=h.to('cpu') if self.relation_condition.checkpoint_inputs_on_cpu else h
        for begin in tuple(range(0,count,bound)) or (0,):
            end=min(begin+bound,count)
            local=[]
            for rows,ids in slots:
                selected=(rows>=begin)&(rows<end)
                local.append((rows[selected]-begin,ids[selected]))
            def evaluate(node,rf,mat,mm,enabled,kind,*,positions=tuple(local)):
                return self._relation_chunk(node.to(device),rf,mat,mm,enabled,kind,positions,movable)
            storage=torch.autograd.graph.save_on_cpu(pin_memory=False) if self.relation_condition.checkpoint_inputs_on_cpu else nullcontext()
            with storage:
                partial=checkpoint(evaluate,node,features[:,begin:end].contiguous(),material[:,begin:end].contiguous(),
                    mask[:,begin:end].contiguous(),active[:,begin:end].contiguous(),kinds[begin:end].contiguous(),
                    use_reentrant=False,preserve_rng_state=False)
            message=message+partial
        return message

    def forward(self,graph,node_features,relation_features,material,material_mask,active,*,plan):
        # Shapes are fixed within one capture, while batch values/materials and
        # merge choices are refreshed on every call.
        h=node_features.new_zeros(*node_features.shape[:2],self.width)
        for ids,tokenizer in zip(plan.type_ids,self.tokenizers):
            if len(ids):h=h.index_copy(1,ids,tokenizer(node_features[:,ids]))
        if plan.topology_owner!=id(graph) or plan.generation!=int(getattr(graph,'topology_generation',0)):raise ValueError('Stale interaction graph plan')
        layout=graph.layout
        if getattr(self,'compact_relations',False):
            from .neural_relations import active_layout
            layout,selected=active_layout(layout,active)
            relation_features=relation_features[:,selected]
            material=material[:,selected];material_mask=material_mask[:,selected]
            active=active[:,selected]
        relation_features=torch.where(active[...,None].bool(),relation_features,0.)
        material=torch.where(active[...,None].bool()&material_mask.bool(),material,0.)
        material_mask=torch.where(active[...,None].bool(),material_mask,0.)
        degree=h.new_zeros(len(h),graph.n,1);message=torch.zeros_like(h)
        bounded_messages=bool(self.relation_condition.checkpoint_rows)
        if bounded_messages and self.training and torch.is_grad_enabled():
            message=self.stream_relation_messages(h,relation_features,material,material_mask,active,
                layout.relation_kind,layout.slots,graph.movable)
        else:
            row=self.relation_state(relation_features)+self.relation_kind(layout.relation_kind)[None]
            members=relation_members(h,layout.slots,len(layout.incidence),
                                     chunk_rows=self.relation_condition.checkpoint_rows)
            member_count=h.new_zeros(len(layout.incidence),1)
            for rows,ids in layout.slots:
                member_count=member_count.index_add(0,rows,graph.movable[ids,None].to(h.dtype))
            row=row+self.relation_members(members/member_count.clamp_min(1)[None])
            row=self.relation_condition(row,torch.cat((material,material_mask),-1))
            if bounded_messages:
                message=relation_messages(row,layout.slots,graph.n,active,
                                          chunk_rows=self.relation_condition.checkpoint_rows)
        condition=h.new_zeros(len(h),graph.n,18)
        for rows,ids in layout.slots:
            keep=active[:,rows,None].to(h.dtype)
            if not bounded_messages:message=message.index_add(1,ids,torch.where(keep.bool(),row[:,rows],0.))
            degree=degree.index_add(1,ids,keep)
            condition=condition.index_add(1,ids,torch.where(keep.bool(),torch.cat((material[:,rows],material_mask[:,rows]),-1),0.))
        bypass=message/degree.clamp_min(1)
        h=self.material(h,condition/degree.clamp_min(1))+self.local(torch.cat((h,bypass),-1))
        h=torch.where(graph.movable[None,:,None],h,0.)
        decoded=torch.zeros_like(h)
        bucketed=bool(getattr(graph,'component_batch',False)) and bool(plan.buckets)
        if bucketed:
            for bucket in plan.buckets:
                components=[plan.components[index] for index in bucket]
                ids=torch.stack([component.ids for component in components]);batch=len(h);count=len(ids);length=ids.shape[1]
                local=h[:,ids].reshape(batch*count,length,self.width);features=[];counts=[]
                for group in components[0].groups:
                    value=local[:,group];size=value.new_ones(len(value),len(group),1)
                    value,size=merge_group(value,size,components[0].merge_rounds)
                    features.append(value);counts.append(size)
                coarse=torch.cat(features,1);sizes=torch.cat(counts,1)
                for layer in self.encoder:coarse=layer(coarse,coarse,sizes)
                value=local;bucket_bypass=bypass[:,ids].reshape_as(local)
                for layer in self.decoder:value=layer(value+bucket_bypass,coarse,sizes)
                decoded=decoded.index_copy(1,ids.flatten(),value.reshape(batch,count*length,self.width))
        fast_singletons=not bucketed and bool(getattr(graph,'fast_rigid_contact',False))
        single_ids=[component.ids for component in plan.components if len(component.ids)==1] if fast_singletons else []
        if single_ids:
            ids=torch.cat(single_ids);batch=len(h);count=len(ids)
            local=h[:,ids].reshape(batch*count,1,self.width)
            coarse=local;sizes=local.new_ones(batch*count,1,1)
            for layer in self.encoder:coarse=layer(coarse,coarse,sizes)
            value=local;single_bypass=bypass[:,ids].reshape_as(local)
            for layer in self.decoder:value=layer(value+single_bypass,coarse,sizes)
            decoded=decoded.index_copy(1,ids,value.reshape(batch,count,self.width))
        for component in plan.components:
            if bucketed:continue
            if fast_singletons and len(component.ids)==1:continue
            local=h[:,component.ids];features=[];counts=[]
            for group in component.groups:
                value=local[:,group];size=value.new_ones(len(value),len(group),1)
                value,size=merge_group(value,size,component.merge_rounds)
                features.append(value);counts.append(size)
            coarse=torch.cat(features,1);sizes=torch.cat(counts,1)
            for layer in self.encoder:coarse=layer(coarse,coarse,sizes)
            value=local
            for layer in self.decoder:
                # The unmerged hyperedge bypass is injected at every decoder.
                value=layer(value+bypass[:,component.ids],coarse,sizes)
            decoded=decoded.index_copy(1,component.ids,value)
        out=torch.zeros_like(h)
        for ids,head in zip(plan.type_ids,self.readout):
            if len(ids):out=out.index_copy(1,ids,.25*head(decoded[:,ids]).tanh())
        return out
