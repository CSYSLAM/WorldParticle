"""Experimental redesign: a shared, root-preserving relation response operator.

The caller supplies the assembled discrete residual and audited Jacobians.
The default configuration remains an algebra prototype; MixedSolver uses the
metric-normalized, learned-baseline variant with a trained shared checkpoint.
Neither relation error nor a learned displacement replaces the full residual.
"""

from dataclasses import dataclass

import torch
from torch import nn


def skew(vector):
    x, y, z = vector.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero,-z,y,z,zero,-x,-y,x,zero), -1).reshape(*vector.shape[:-1],3,3)


def point_pair_jacobian(left_lever, right_lever):
    """d(point_right-point_left)/d(world COM translation, world rotation).

    Output [...,3,2,6]. Callers zero the rotation columns for particle nodes.
    Reactions scatter to BOTH bodies, including lever-arm torques.
    """
    eye = torch.eye(3, device=left_lever.device, dtype=left_lever.dtype).expand(*left_lever.shape[:-1],3,3)
    left = torch.cat((-eye, skew(left_lever)), -1)
    right = torch.cat((eye, -skew(right_lever)), -1)
    return torch.stack((left,right), -2)


@dataclass
class RelationLayout:
    incidence: torch.Tensor       # [R,S], -1 padding, no implicit world entity
    entity_kind: torch.Tensor     # [N], node-soft/node-cloth/body/rod = 0..3
    relation_kind: torch.Tensor   # [R], tet/membrane/bend/joint/rod/attachment/contact
    pool: torch.Tensor            # [R], a static or step-frozen geometric merge
    pool_component: torch.Tensor # [P], physical interaction component, not scene ID
    conditioning_mask: torch.Tensor | None = None

    def __post_init__(self):
        r, _ = self.incidence.shape
        if r != len(self.relation_kind) or r != len(self.pool):
            raise ValueError("Inconsistent relation layout")
        if bool(((self.incidence < -1) | (self.incidence >= len(self.entity_kind))).any()):
            raise ValueError("Invalid incidence")
        if bool(((self.entity_kind < 0) | (self.entity_kind > 3)).any()):
            raise ValueError("Invalid entity kind")
        if bool(((self.relation_kind < 0) | (self.relation_kind > 6)).any()):
            raise ValueError("Invalid relation kind")
        if (r and not len(self.pool_component)) or bool(((self.pool < 0) | (self.pool >= len(self.pool_component))).any()):
            raise ValueError("Invalid coarse pool")
        self.slots = []
        for slot in range(self.incidence.shape[1]):
            rows = torch.where(self.incidence[:,slot] >= 0)[0]
            self.slots.append((rows, self.incidence[rows,slot]))
        if self.conditioning_mask is None:self.conditioning_mask=torch.ones_like(self.entity_kind,dtype=torch.bool)
        if self.conditioning_mask.shape!=self.entity_kind.shape:raise ValueError('Invalid entity conditioning mask')
        self.entity_degree=torch.zeros_like(self.entity_kind)
        for rows,ids in self.slots:self.entity_degree=self.entity_degree.index_add(0,ids,torch.ones_like(ids))
        self.log_entity_degree=self.entity_degree.float().log1p()
        self.components = [torch.where(self.pool_component == label)[0] for label in torch.unique(self.pool_component, sorted=True).tolist()]
        self.counts = torch.bincount(self.pool, minlength=len(self.pool_component)).clamp_min(1)
        self.relation_overlap=torch.ones(r,device=self.incidence.device)
        for pools in self.components:
            member=torch.isin(self.pool,pools)
            incidence=self.incidence[member].flatten()
            degree=torch.bincount(incidence[incidence>=0],minlength=len(self.entity_kind))
            self.relation_overlap[member]=max(1,int(degree.max()))


class RelationResponse(nn.Module):
    """P(r)=B r + T^T(A_local + S^T U U^T S)T r.

    B is a caller-supplied symmetric positive local metric in mass-whitened
    coordinates. T is the normalized, mass-whitened relation Jacobian.
    Local A and each component's coarse U U^T are positive semidefinite.
    Consequently r=0 gives EXACTLY zero update for arbitrary network weights.
    Positive definiteness alone does not authorize an unbounded physical step.
    """

    architecture_id = "relation_response_v0"

    def __init__(self, entity_features=24, relation_features=32, width=64, rank=2, channels=9, learned_baseline=False, metric_normalization=False, material_film=False):
        super().__init__()
        self.width, self.rank, self.channels = width, rank, channels
        self.metric_normalization=metric_normalization
        self.ablation='full'
        encoder = lambda size: nn.Sequential(nn.Linear(size,width),nn.SiLU(),nn.Linear(width,width))
        self.entity_encoders = nn.ModuleList([encoder(entity_features) for _ in range(4)])
        self.relation_encoder = encoder(relation_features)
        self.relation_type = nn.Embedding(7,width)
        self.fine = nn.Sequential(nn.LayerNorm(width),nn.Linear(width,width),nn.SiLU())
        layer = nn.TransformerEncoderLayer(width,4,2*width,dropout=0,batch_first=True,norm_first=True)
        self.coarse = nn.TransformerEncoder(layer,2,enable_nested_tensor=False)
        self.local_head = nn.Linear(width,channels*(rank+1))
        self.coarse_head = nn.Linear(width,channels*rank)
        self.entity_gain=nn.Linear(width,1) if learned_baseline else None
        self.material_film=None
        if material_film:
            self.material_film=nn.Sequential(nn.Linear(18,32),nn.SiLU(),nn.Linear(32,2*width))
            nn.init.zeros_(self.material_film[-1].weight);nn.init.zeros_(self.material_film[-1].bias)
        if self.entity_gain is not None:
            nn.init.zeros_(self.entity_gain.weight);nn.init.zeros_(self.entity_gain.bias)

    def encode_entities(self, layout, entity_features, relation_features=None):
        encoded=sum(module(entity_features)*(layout.entity_kind == kind)[None,:,None] for kind,module in enumerate(self.entity_encoders))
        if self.material_film is None:return encoded
        if relation_features is None:raise ValueError('Material FiLM requires relation parameter features')
        tags=torch.nn.functional.one_hot(layout.relation_kind,7).to(encoded.dtype)[None].expand(encoded.shape[0],-1,-1)
        relation=torch.cat((relation_features[...,9:18],tags,relation_features[...,27:28]),-1)
        context=encoded.new_zeros(encoded.shape[0],encoded.shape[1],17)
        for rows,ids in layout.slots:context=context.index_add(1,ids,relation[:,rows])
        context=context/layout.entity_degree.clamp_min(1)[None,:,None]
        degree=layout.log_entity_degree.to(encoded.dtype)[None,:,None].expand(encoded.shape[0],-1,-1)
        gamma,beta=self.material_film(torch.cat((context,degree),-1)).chunk(2,-1)
        mask=layout.conditioning_mask[None,:,None]
        return encoded*(1+.25*gamma.tanh()*mask)+.25*beta.tanh()*mask

    def factors(self, layout, entity_features, relation_features, encoded=None):
        if encoded is None:encoded=self.encode_entities(layout,entity_features,relation_features)
        b, r, _ = relation_features.shape
        incident = encoded.new_zeros(b,r,self.width)
        degree = encoded.new_zeros(r)
        for rows, ids in layout.slots:
            incident[:,rows] = incident[:,rows]+encoded[:,ids]
            degree[rows] = degree[rows]+1
        h = self.fine(self.relation_encoder(relation_features)+self.relation_type(layout.relation_kind)+incident/degree.clamp_min(1)[None,:,None])
        p = len(layout.pool_component)
        pooled = h.new_zeros(b,p,self.width).index_add(1,layout.pool,h)/layout.counts[None,:,None]
        separated = layout.pool_component[:,None] != layout.pool_component[None,:]
        latent = self.coarse(pooled, mask=separated, is_causal=False)
        local = self.local_head(h).reshape(b,r,self.channels,self.rank+1)
        diagonal = .1*local[...,0].sigmoid()
        low_rank = local[...,1:].tanh()
        low_rank = .2*low_rank/low_rank.square().sum((-1,-2),keepdim=True).clamp_min(1).sqrt()
        coarse = self.coarse_head(latent).reshape(b,p,self.channels,self.rank).tanh()
        # Normalize each component independently: another isolated scene cannot
        # change either features or the size of this component's response.
        blocks = []
        for ids in layout.components:
            block = coarse[:,ids]
            norm = block.square().sum((1,2,3),keepdim=True).clamp_min(1).sqrt()
            blocks.append((ids,.2*block/norm))
        gain=None if self.entity_gain is None else .1+1.8*self.entity_gain(encoded).sigmoid()
        return diagonal, low_rank, blocks, gain

    @staticmethod
    def whiten_jacobian(layout, jacobian, mass_inverse_sqrt):
        safe = layout.incidence.clamp_min(0)
        mask = (layout.incidence >= 0)[None,:,None,:,None]
        # Relation channels and entity DOFs have different dimensions; e.g.
        # a tet can expose nine objective F^T F entries; each node still has 3 DOF.
        weighted = torch.einsum('brdsu,brsuv->brdsv',jacobian,mass_inverse_sqrt[:,safe])*mask
        normalization = weighted.square().sum((-1,-2),keepdim=True).sqrt().clamp_min(1e-12)
        return weighted/normalization

    @staticmethod
    def gather(layout, matrix, vector):
        safe = layout.incidence.clamp_min(0)
        return torch.einsum('brdsu,brsu->brd',matrix,vector[:,safe])

    @staticmethod
    def scatter(layout, matrix, value, count):
        slots = torch.einsum('brdsu,brd->brsu',matrix,value)
        result = slots.new_zeros(slots.shape[0],count,6)
        for slot,(rows,ids) in enumerate(layout.slots):
            result = result.index_add(1,ids,slots[:,rows,slot])
        return result

    def forward(self, layout, residual, mass_inverse_sqrt, baseline, jacobian, entity_features, relation_features, *, encoded=None):
        if not len(layout.incidence):
            r=torch.einsum('bnij,bnj->bni',mass_inverse_sqrt,residual)
            local=torch.einsum('bnij,bnj->bni',baseline,r)
            return -torch.einsum('bnij,bnj->bni',mass_inverse_sqrt,local)
        if jacobian.shape[2] > self.channels:
            raise ValueError("Relation Jacobian exceeds the configured channel budget")
        if jacobian.shape[2] < self.channels:
            jacobian=torch.nn.functional.pad(jacobian,(0,0,0,0,0,self.channels-jacobian.shape[2]))
        diagonal, low_rank, coarse, gain = self.factors(layout,entity_features,relation_features,encoded)
        t = self.whiten_jacobian(layout,jacobian,mass_inverse_sqrt)
        if self.metric_normalization:
            # Normalize virtual relation directions in the inverse local solve
            # metric, rather than mass alone. A stiff angular row must not
            # acquire an O(1) displacement response. The overlap bound gives
            # T^T T <= B component-wise, without breaking J^T reaction scatter.
            inverse=torch.linalg.inv_ex(baseline+1e-9*torch.eye(6,device=baseline.device,dtype=baseline.dtype),check_errors=False).inverse
            metric=inverse[:,layout.incidence.clamp_min(0)]
            norm=torch.einsum('brdsi,brsij,brdsj->brd',t,metric,t).clamp_min(1e-20).sqrt()
            overlap=(self.channels*layout.relation_overlap).sqrt()
            t=t/norm[...,None,None]/overlap[None,:,None,None,None]
        r = torch.einsum('bnij,bnj->bni',mass_inverse_sqrt,residual)
        edge = self.gather(layout,t,r)
        response = diagonal*edge+torch.einsum('brdk,brk->brd',low_rank,torch.einsum('brdk,brd->brk',low_rank,edge))
        # S is normalized sparse pooling. Its transpose is the exact unpool;
        # no dense entity-to-super-token trace or global square matrix exists.
        scale = layout.counts[layout.pool].to(edge.dtype).sqrt()
        pooled = edge.new_zeros(edge.shape[0],len(layout.pool_component),self.channels).index_add(1,layout.pool,edge/scale[None,:,None])
        coarse_response = torch.zeros_like(pooled)
        for ids,u in coarse:
            coefficient = torch.einsum('bpdk,bpd->bk',u,pooled[:,ids])
            coarse_response[:,ids] = torch.einsum('bpdk,bk->bpd',u,coefficient)
        if self.ablation!='no_coarse':response=response+coarse_response[:,layout.pool]/scale[None,:,None]
        response = self.scatter(layout,t,response,len(layout.entity_kind))
        local=torch.einsum('bnij,bnj->bni',baseline,r)
        if self.ablation=='local_gain':response=torch.zeros_like(response)
        if self.ablation=='relation_only':gain=None
        total = local*(1 if gain is None else gain)+response
        update = -torch.einsum('bnij,bnj->bni',mass_inverse_sqrt,total)
        return update
