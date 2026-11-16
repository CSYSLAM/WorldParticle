"""WorldParticle-style typed tokenizer, hierarchical merge, and cross decoder.

Research implementation trained from scratch. It does not claim compatibility
with the original checkpoint. Merges use canonical geometry ordering within
object/type groups, and propagate weighted centers without a dense trace matrix.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .data import State
from . import math3d as so3
from .feasibility import accepted_scale
from .physics import inertial_target, local_update


def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.SiLU(), nn.Linear(b, c))


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, query, context, keep):
        b, n, width = query.shape
        reshape = lambda x: x.reshape(b, -1, self.heads, width//self.heads).transpose(1, 2)
        q, k, v = reshape(self.q(query)), reshape(self.k(context)), reshape(self.v(context))
        value = F.scaled_dot_product_attention(q, k, v, attn_mask=keep[None, None])
        return self.output(value.transpose(1, 2).reshape(b, n, width))


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.context_norm = nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.ffn_norm = nn.LayerNorm(width)
        self.ffn = mlp(width, 2*width, width)

    def forward(self, h, context, keep):
        h = h+self.attention(self.norm(h), self.context_norm(context), keep)
        return h+self.ffn(self.ffn_norm(h))


def merge_tokens(h, positions, sizes, objects, components, plan=None):
    features, centers, weights, labels, groups = [], [], [], [], []
    indices = plan[0] if plan is not None else [torch.where(objects == group)[0] for group in torch.unique(objects, sorted=True).tolist()]
    for ids in indices:
        feat, pos, size = h[:, ids], positions[:, ids], sizes[:, ids]
        n = len(ids)
        if n > 1:
            order = torch.arange(n, device=h.device)[None].expand(h.shape[0], -1)
            with torch.no_grad():
                for axis in (2, 1, 0):
                    coordinate = pos[..., axis].gather(1, order)
                    order = order.gather(1, torch.argsort(coordinate, dim=1, stable=True))
            feat = feat.gather(1, order[..., None].expand(-1, -1, feat.shape[-1]))
            pos = pos.gather(1, order[..., None].expand(-1, -1, 3))
            size = size.gather(1, order[..., None])
            a, dest = feat[:, ::2], feat[:, 1::2]
            with torch.no_grad():
                similarity = F.normalize(a, dim=-1) @ F.normalize(dest, dim=-1).transpose(-1, -2)
                match = similarity.argmax(-1)
            sa, sd = size[:, ::2], size[:, 1::2]
            # Sorted prefix sums avoid floating atomic reductions and a dense
            # merge trace. Shapes remain fixed for CUDA graph capture, including
            # when deterministic PyTorch scatter_add would synchronize the CPU.
            order = torch.argsort(match, dim=1, stable=True)
            sorted_match = match.gather(1,order).contiguous()
            boundaries = torch.arange(dest.shape[1]+1,device=h.device)[None].expand(h.shape[0],-1).contiguous()
            boundaries = torch.searchsorted(sorted_match,boundaries)
            def accumulate(target, source):
                ordered = source.gather(1,order[...,None].expand_as(source))
                prefix = torch.cat((torch.zeros_like(ordered[:,:1]),ordered.cumsum(1)),1)
                low=prefix.gather(1,boundaries[:,:-1,None].expand(-1,-1,source.shape[-1]))
                high=prefix.gather(1,boundaries[:,1:,None].expand(-1,-1,source.shape[-1]))
                return target+high-low
            total = accumulate(sd,sa)
            feat = accumulate(dest*sd,a*sa)/total
            pos = accumulate(pos[:,1::2]*sd,pos[:,::2]*sa)/total
            size = total
        features.append(feat); centers.append(pos); weights.append(size)
        if plan is None:
            labels.append(components[ids[0]].expand(feat.shape[1]))
            groups.append(objects[ids[0]].expand(feat.shape[1]))
    return torch.cat(features, 1), torch.cat(centers, 1), torch.cat(weights, 1), plan[1] if plan is not None else torch.cat(groups), plan[2] if plan is not None else torch.cat(labels)


class SharedSolver(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        width, heads = config["width"], config["heads"]
        self.encoders = nn.ModuleList([mlp(36, width, width) for _ in range(4)])
        self.relation_encoder = mlp(81, width, width)
        self.film = mlp(3, width, 2*width)
        self.position = nn.Linear(3, width)
        self.encoder = nn.ModuleList([Block(width, heads) for _ in range(config["encoder_layers"])])
        self.decoder = nn.ModuleList([Block(width, heads) for _ in range(config["decoder_layers"])])
        self.heads = nn.ModuleList([mlp(width, width, 6) for _ in range(4)])
        for head in self.heads:
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)
        nn.init.zeros_(self.film[-1].weight); nn.init.zeros_(self.film[-1].bias)

    def relation_features(self, scene, state, dt):
        ids = scene.relations
        b, n = state.x.shape[:2]
        if not len(ids):
            return state.x.new_zeros(b, n, self.config["width"])
        mask = (ids >= 0).to(state.x.dtype)
        safe = ids.clamp_min(0)
        x = state.x[:, safe]
        rest = scene.rest_x[safe]
        mean = (x*mask[None, ..., None]).sum(2, keepdim=True)/mask.sum(-1)[None, :, None, None]
        restmean = (rest*mask[..., None]).sum(1, keepdim=True)/mask.sum(-1)[:, None, None]
        expand = lambda x: x[None].expand(b, *x.shape)
        extra = expand(scene.relation_extra)
        if scene.has_plane:
            extra = extra.clone()
            start = scene.plane_relation_start
            extra[:, start:, 5] = ((state.x*scene.plane_normal).sum(-1)-scene.plane_offset)/scene.length
            extra[:, start:, 6:9] = state.v*dt/scene.length
            extra[:, start:, 9] = (extra[:, start:, 5] < scene.contact_radius/scene.length).to(x.dtype)
        inputs = torch.cat((((x-mean)*mask[None, ..., None]/scene.length).flatten(2), expand(((rest-restmean)*mask[..., None]/scene.length).flatten(1)), (state.v[:, safe]*dt/scene.length*mask[None, ..., None]).flatten(2), (state.q[:, safe]*mask[None, ..., None]).flatten(2), expand(torch.log1p(scene.relation_material)/10), expand(mask), expand(F.one_hot(scene.relation_kind, 5).to(x.dtype)), extra), -1)
        message = self.relation_encoder(inputs)
        out = state.x.new_zeros(b, n, message.shape[-1])
        # Sum incidence messages, normalized by degree. Multi-node relations stay
        # intact until their complete state has been encoded.
        degree = state.x.new_zeros(n)
        for rows, selected in scene.relation_slots:
            out = out.index_add(1, selected, message[:, rows])
            degree = degree.index_add(0, selected, torch.ones_like(selected, dtype=x.dtype))
        return out/degree.clamp_min(1)[None, :, None]

    def correction(self, scene, state, target_x, dx, dr, dt):
        b, n, _ = state.x.shape
        expand = lambda x: x[None].expand(b, *x.shape)
        mass = torch.log1p(scene.mass)[:, None]
        inertia = torch.log1p(scene.inertia.diagonal(dim1=-2, dim2=-1).abs()/scene.length**2)
        features = torch.cat(((state.x-scene.rest_x)/scene.length, state.v*dt/scene.length, so3.matrix(state.q).flatten(-2), state.omega*dt, (target_x-state.x)/scene.length, dx/scene.length, dr, expand(mass), expand(inertia), expand(scene.movable[:, None].to(state.x.dtype)), expand((scene.rest_x-scene.origin)/scene.length), state.x.new_full((b, n, 1), math.log(dt))), -1)
        h = sum(encoder(features)*(scene.kind == kind)[None, :, None] for kind, encoder in enumerate(self.encoders))
        relation = self.relation_features(scene, state, dt)
        h = h+relation
        condition = torch.cat((state.x.new_full((n, 1), dt), mass, inertia.sum(-1, keepdim=True)), -1)
        gain, bias = self.film(condition).chunk(2, -1)
        h = (1+0.1*gain.tanh())*h+0.1*bias.tanh()
        positions = (state.x-scene.origin)/scene.length
        latent = h+self.position(positions)
        coarse_pos, sizes = positions, state.x.new_ones(b, n, 1)
        objects = scene.object_id*4+scene.kind
        components = scene.component
        for depth, layer in enumerate(self.encoder):
            latent = layer(latent, latent, components[:, None] == components[None, :])
            latent, coarse_pos, sizes, objects, components = merge_tokens(latent, coarse_pos, sizes, objects, components, scene.merge_plans[depth])
        decoded = h
        for layer in self.decoder:
            decoded = layer(decoded+relation, latent, scene.component[:, None] == components[None, :])
        raw = sum(head(decoded)*(scene.kind == kind)[None, :, None] for kind, head in enumerate(self.heads))
        # Component-level root-preserving gate for the implemented local model.
        local_size = (dx/scene.length).square().sum(-1)+dr.square().sum(-1)
        gate = torch.zeros_like(local_size)
        for ids in scene.component_indices:
            norm = (local_size[:, ids].mean(-1)+1e-12).sqrt()-1e-6
            gate = gate.index_copy(1,ids,(norm/(norm+1e-3))[:,None].expand(-1,len(ids)))
        raw = raw.tanh()*gate[..., None]
        return raw[..., :3]*scene.length*self.config["max_translation_over_L"], raw[..., 3:]*self.config["max_rotation"]

    def forward(self, scene, state, dt, *, learned=True, force=None, torque=None):
        target_x, target_q = inertial_target(scene, state, dt, force, torque)
        x, q = state.x, state.q
        accepted = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        min_alpha = x.new_ones(x.shape[0])
        for _ in range(self.config["unroll"]):
            dx, dr = local_update(scene, x, q, target_x, target_q, dt, state.x)
            if learned:
                current = State(x, q, state.v, state.omega)
                nx, nr = self.correction(scene, current, target_x, dx, dr, dt)
                dx, dr = dx+nx, dr+nr
            dx, dr = self.config["relaxation"]*dx, self.config["relaxation"]*dr
            move = scene.movable[None, :, None]
            dx = torch.where(move, dx, torch.zeros_like(dx))
            dr = torch.where(move & (scene.kind >= 2)[None, :, None], dr, torch.zeros_like(dr))
            alpha, valid = accepted_scale(scene, x, x+dx)
            min_alpha = torch.minimum(min_alpha, alpha)
            accepted = accepted & valid
            x = x+alpha[:, None, None]*dx
            q = so3.multiply(so3.exp(alpha[:, None, None]*dr), q)
        velocity = (x-state.x)/dt
        omega = so3.log(so3.multiply(q, so3.conjugate(state.q)))/dt
        return State(x, q, velocity, omega), {"accepted": accepted, "min_alpha": min_alpha, "contact_safety": "ANALYTIC_PLANE" if scene.has_plane else "NO_CONTACT"}
