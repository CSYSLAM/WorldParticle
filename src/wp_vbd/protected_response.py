"""Protected Galerkin space and a shared invariant learned factor response.

Physical columns are orthogonalized in a sparse positive energy embedding.
The neural branch changes only the complementary local inverse through
S (I + Z T Z^T)^2 S^T. T=0 gives an exact physical initialization with a
nonzero gate gradient. No global dynamics matrix or convergence loop exists.
"""
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from torch import nn
from . import math3d as so3
from .rod_rows import rotate_pinned
from .response_operator import skew


def metric_terms(scene, metric):
    if hasattr(metric, 'terms'):
        return metric.terms
    return [SimpleNamespace(ids=scene.joints, jacobian=metric.jacobian, weight=metric.weights)]


def metric_action(metric, value):
    # Both interfaces expose the same sparse operator; rod accepts vectors.
    if hasattr(metric, 'terms'):
        return metric(value[..., None])[..., 0] if value.ndim == 3 else metric(value)
    if value.ndim == 3:
        return metric(value)
    return torch.stack([metric(value[..., i]) for i in range(value.shape[-1])], -1)


@dataclass
class ComponentTopology:
    ids: torch.Tensor
    blocks: list
    terms: list
    diagonal_exact: bool = True


def topology(scene, metric, blocks):
    """Freeze index maps during eager warmup; never copy CUDA data in capture."""
    terms = metric_terms(scene, metric)
    signature = (tuple((tuple(p.ids.shape), z.shape[-1]) for p, z, *_ in blocks),
                 tuple(tuple(t.ids.shape) for t in terms),
                 tuple(getattr(t,'topology_key',None) for t in terms))
    cache = getattr(scene, '_protected_topology', None)
    if cache is not None and cache[0] == signature:
        return cache[1]
    groups = []
    term_ids = [t.ids.cpu().tolist() for t in terms]
    # A repeated owner contributes cross terms within J^T W J. The generic
    # block-diagonal preconditioner omits those terms, so it cannot serve as
    # the exact six-DOF inverse for such an entity. Keep its sparse energy
    # embedding instead, including both roles of the owner.
    repeated = set()
    for rows in term_ids:
        for row in rows:
            seen = set()
            for entity in row:
                if entity in seen: repeated.add(entity)
                seen.add(entity)
    block_ids = [p.ids.cpu().tolist() for p, *_ in blocks]
    for entities in scene.interaction_entities:
        ids = entities.cpu().tolist()
        lookup = {entity: i for i, entity in enumerate(ids)}
        selected = []
        for index, members in enumerate(block_ids):
            if set(members).issubset(lookup):
                selected.append((index, torch.tensor([lookup[i] for i in members], device=scene.device)))
        mapped = []
        for index, rows in enumerate(term_ids):
            retained = [i for i, row in enumerate(rows) if any(entity in lookup for entity in row)]
            if not retained:
                continue
            slots = [[lookup.get(entity, -1) for entity in rows[i]] for i in retained]
            slots = torch.tensor(slots, device=scene.device)
            mapped.append((index, torch.tensor(retained, device=scene.device), slots.clamp_min(0), slots >= 0))
        groups.append(ComponentTopology(entities, selected, mapped, not bool(set(ids) & repeated)))
    scene._protected_topology = signature, groups
    return groups


class ProtectedCoarse:
    """Q Q^T with Q^T H Q=I on retained physical directions.

    Two fixed modified Gram--Schmidt passes use scalar normalizations. The
    total retained component space is explicit (up to 12 in this prototype),
    rather than pretending the union of small overlapping blocks is a projector.
    Redundant columns are masked; there is no enlarged dense reduced solve.
    Physical H/Q are frozen in backward, matching the stated solver policy.
    """
    def __init__(self, scene, metric, blocks, *, maximum_columns=12, rank_relative=1e-12, exact_rigid_singletons=False):
        self.scene, self.metric = scene, metric
        self.topology = topology(scene, metric, blocks)
        self.groups = []
        self.ranks = []
        batch = len(metric.local)
        self.valid = torch.ones(batch, device=scene.device, dtype=torch.bool)
        terms = metric_terms(scene, metric)
        self.singleton_group_mask=[];self.singleton_ids=None;self.all_mobile_singletons=False
        if exact_rigid_singletons and scene.metadata.get('contract') not in ('rigid_contact_primal_dual_v21','rigid_joint_contact_finite_v26'):raise ValueError('Exact singleton factorization requires a six-DOF rigid contract')
        with torch.no_grad():
            local = metric.local.double()
            ghost = torch.diag_embed((~scene.dof).double())[None]
            root, info = torch.linalg.cholesky_ex(local + ghost, check_errors=False)
            self.valid &= (info == 0).all(-1) & torch.isfinite(root).flatten(1).all(-1)
            if exact_rigid_singletons:
                for term in terms:
                    self.valid&=torch.isfinite(term.weight).flatten(1).all(-1)&(term.weight>=0).flatten(1).all(-1)&torch.isfinite(term.jacobian).flatten(1).all(-1)
                diagonal=metric.diagonal.double()+ghost
                diagonal_root,diagonal_info=torch.linalg.cholesky_ex(diagonal,check_errors=False)
                identity=torch.eye(6,device=scene.device,dtype=torch.float64).expand_as(diagonal_root)
                singleton_q=torch.linalg.solve_triangular(diagonal_root.transpose(-1,-2),identity,upper=True)
                self.valid&=(diagonal_info==0).all(-1)&torch.isfinite(singleton_q).flatten(1).all(-1)
            singleton_ids=[]
            for group in self.topology:
                exact=exact_rigid_singletons and group.diagonal_exact and len(group.ids)==1 and len(group.blocks)==1 and blocks[group.blocks[0][0]][1].shape[-1]==6
                self.singleton_group_mask.append(exact)
                if not group.blocks:
                    self.groups.append((group.ids, local.new_zeros(batch, len(group.ids), 6, 0)))
                    self.ranks.append(torch.zeros(batch, device=scene.device, dtype=torch.long))
                    continue
                if exact:
                    self.groups.append((group.ids,singleton_q[:,group.ids]))
                    self.ranks.append(torch.full((batch,),6,device=scene.device,dtype=torch.long))
                    singleton_ids.append(group.ids)
                    continue
                raw = []
                for index, mapped in group.blocks:
                    z = blocks[index][1].double()
                    raw.append(z.new_zeros(batch, len(group.ids), 6, z.shape[-1]).index_copy(1, mapped, z))
                z = torch.cat(raw, -1) * scene.dof[group.ids][None, :, :, None]
                if z.shape[-1] > maximum_columns:
                    raise ValueError('Protected component exceeds its declared column budget')
                embedding = [(root[:, group.ids].transpose(-1, -2) @ z).flatten(1, 2)]
                for index, rows, ids, valid in group.terms:
                    term = terms[index]
                    jacobian = term.jacobian[:, rows].double() * valid[None, :, None, :, None]
                    image = torch.einsum('brcsi,brsip->brcp', jacobian, z[:, ids])
                    weight = term.weight[:, rows].double()
                    self.valid &= torch.isfinite(weight).flatten(1).all(-1) & (weight >= 0).flatten(1).all(-1)
                    embedding.append((image * weight.clamp_min(0).sqrt()[..., None]).flatten(1, 2))
                energy = torch.cat(embedding, 1)
                self.valid &= torch.isfinite(z).flatten(1).all(-1) & torch.isfinite(energy).flatten(1).all(-1)
                basis, images, retained = [], [], []
                for column in range(z.shape[-1]):
                    value, image = z[..., column], energy[..., column]
                    original = image.square().sum(-1)
                    if basis:
                        q, e = torch.stack(basis, -1), torch.stack(images, -1)
                        for _ in range(2):
                            coefficient = torch.einsum('bdp,bd->bp', e, image)
                            value = value - torch.einsum('bnip,bp->bni', q, coefficient)
                            image = image - torch.einsum('bdp,bp->bd', e, coefficient)
                    norm = image.square().sum(-1)
                    keep = (norm > rank_relative * original) & (original > 0) & torch.isfinite(norm)
                    scale = torch.where(keep, norm.clamp_min(1e-300).rsqrt(), 0.)
                    basis.append(value * scale[:, None, None])
                    images.append(image * scale[:, None])
                    retained.append(keep)
                self.groups.append((group.ids, torch.stack(basis, -1)))
                self.ranks.append(torch.stack(retained, -1).sum(-1))
            if singleton_ids:
                self.singleton_ids=torch.cat(singleton_ids);self.singleton_q=singleton_q[:,self.singleton_ids]
            self.all_mobile_singletons=exact_rigid_singletons and all(exact or not group.blocks for exact,group in zip(self.singleton_group_mask,self.topology))

    def __call__(self, value):
        vector = value.ndim == 3
        source = value.double()[..., None] if vector else value.double()
        result = torch.zeros_like(source)
        if self.singleton_ids is not None:
            q=self.singleton_q
            value=q@(q.transpose(-1,-2)@source[:,self.singleton_ids])
            result=result.index_copy(1,self.singleton_ids,value)
        for index,(ids, q) in enumerate(self.groups):
            if self.singleton_group_mask[index]:continue
            if not q.shape[-1]:continue
            coefficient = torch.einsum('bnip,bniq->bpq', q, source[:, ids])
            result = result.index_add(1, ids, torch.einsum('bnip,bpq->bniq', q, coefficient))
        return (result[..., 0] if vector else result).to(value.dtype)


def centers(scene, state):
    position = state.x
    if scene.metadata.get('body_position_representation') == 'origin':
        position = position + rotate_pinned(state.q, scene.body_com)
    return position


class InvariantPortResponse(nn.Module):
    """One node/relation network shared by particle, cloth, body and rod.

    Scalars encode invariant geometry/physics; two nodal shape fields modulate
    translation/rotation ports. A bounded isotropic 2x2 block T (Kronecker I3)
    preserves the arbitrary world-axis gauge. Fixed nodes cannot relay incoming
    conditioning. This is a frozen-feature SPD response, not a claim about
    symmetry of the full nonlinear neural Jacobian.
    """
    def __init__(self, width=32, bound=.4):
        super().__init__()
        if not 0 < bound < 1:
            raise ValueError('Factor modulation bound must lie in (0,1)')
        self.bound = bound
        self.encoder = nn.Sequential(nn.Linear(28, width), nn.SiLU(), nn.Linear(width, width))
        self.message = nn.Sequential(nn.Linear(width + 19, width), nn.SiLU(), nn.Linear(width, width))
        self.update = nn.Sequential(nn.Linear(2 * width, width), nn.SiLU(), nn.LayerNorm(width))
        self.shape = nn.Linear(width, 2)
        self.factor = nn.Linear(width, 3)
        nn.init.zeros_(self.factor.weight)
        nn.init.zeros_(self.factor.bias)

    def stream_local_messages(self,h,row_features,active,slot_counts,positions):
        """Complete the local relation MLP in bounded checkpointed chunks."""
        from contextlib import nullcontext
        from torch.utils.checkpoint import checkpoint
        from .heterogeneous_backbone import relation_members,relation_messages
        bound=getattr(self,'relation_chunk_rows',0)
        if bound<=0:raise ValueError('Positive complete local relation chunk bound required')
        host=bool(getattr(self,'checkpoint_inputs_on_cpu',False));device=h.device
        node=h.to('cpu') if host else h
        gathered=torch.zeros_like(h);count=row_features.shape[1]
        for begin in tuple(range(0,count,bound)) or (0,):
            end=min(begin+bound,count);local=[]
            for rows,ids in positions:
                selected=(rows>=begin)&(rows<end)
                local.append((rows[selected]-begin,ids[selected]))
            def evaluate(value,features,enabled,denominator,*,slots=tuple(local)):
                value=value.to(device)
                pooled=relation_members(value,slots,features.shape[1],chunk_rows=bound)
                message=self.message(torch.cat((pooled/denominator.clamp_min(1)[None,:,None],features),-1))
                message=message*enabled[...,None]
                return relation_messages(message,slots,value.shape[1],torch.ones_like(enabled,dtype=torch.bool),chunk_rows=bound)
            storage=torch.autograd.graph.save_on_cpu(pin_memory=False) if host else nullcontext()
            with storage:
                partial=checkpoint(evaluate,node,row_features[:,begin:end].contiguous(),
                    active[:,begin:end].to(h.dtype).contiguous(),slot_counts[begin:end].to(h.dtype).contiguous(),
                    use_reentrant=False,preserve_rng_state=False)
            gathered=gathered+partial
        return gathered

    def encode(self, scene, state, physical, baseline, relation, dt):
        layout = scene.layout
        if getattr(self,'compact_relations',False):
            from .neural_relations import active_layout
            layout,selected=active_layout(layout,relation[...,27])
            relation=relation[:,selected]
        position = centers(scene, state)
        batch = len(position)
        mobile = scene.movable
        active = relation[..., 27]
        degree = position.new_zeros(batch, scene.n)
        material = position.new_zeros(batch, scene.n, 9)
        row_center = position.new_zeros(batch, len(layout.incidence), 3)
        rest = scene.rest_x
        if scene.metadata.get('body_position_representation') == 'origin':
            rest = rest + rotate_pinned(scene.rest_q, scene.body_com)
        rest_center = torch.zeros_like(row_center)
        slots = position.new_zeros(len(layout.incidence))
        for rows, ids in layout.slots:
            row_center = row_center.index_add(1, rows, position[:, ids])
            rest_center = rest_center.index_add(1, rows, rest[ids][None].expand(batch, -1, -1))
            slots = slots.index_add(0, rows, torch.ones_like(rows, dtype=position.dtype))
            degree = degree.index_add(1, ids, active[:, rows])
            material = material.index_add(1, ids, relation[:, rows, 9:18] * active[:, rows, None])
        row_center = row_center / slots.clamp_min(1)[None, :, None]
        rest_center = rest_center / slots.clamp_min(1)[None, :, None]
        geometry = []
        for points, middle in ((position, row_center), (rest[None].expand(batch, -1, -1), rest_center)):
            average = torch.zeros_like(position)
            square = position.new_zeros(batch, scene.n, 1)
            for rows, ids in layout.slots:
                arm = (middle[:, rows] - points[:, ids]) / scene.length
                average = average.index_add(1, ids, arm * active[:, rows, None])
                square = square.index_add(1, ids, arm.square().sum(-1, keepdim=True) * active[:, rows, None])
            geometry.extend((average.norm(dim=-1, keepdim=True) / degree.clamp_min(1)[..., None], square / degree.clamp_min(1)[..., None]))
        radius, mass_ratio = position.new_zeros(batch, scene.n, 1), position.new_zeros(batch, scene.n, 1)
        batched=bool(getattr(scene,'component_batch',False))
        if batched:
            from .component_batch import layout as component_layout
            groups=component_layout(scene);center=groups.centers(position)
            radius=(position-center[:,groups.labels]).norm(dim=-1,keepdim=True)/scene.length
            reference=groups.mass_total/groups.mobile_count.clamp_min(1)
            mass_ratio=torch.log1p(scene.mass/reference[groups.labels].clamp_min(1e-30))[None,:,None].expand(batch,-1,-1)
        for ids in (() if batched else scene.interaction_entities):
            mass = scene.mass[ids] * mobile[ids]
            center = (position[:, ids] * mass[None, :, None]).sum(1, keepdim=True) / mass.sum().clamp_min(1e-30)
            radius[:, ids] = (position[:, ids] - center).norm(dim=-1, keepdim=True) / scene.length
            reference = mass.sum() / mobile[ids].sum().clamp_min(1)
            mass_ratio[:, ids] = torch.log1p(scene.mass[ids] / reference.clamp_min(1e-30))[None, :, None].expand(batch,-1,-1)
        tags = torch.nn.functional.one_hot(scene.kind, 4).to(position.dtype)[None].expand(batch, -1, -1)
        trace = scene.inertia.diagonal(dim1=-2, dim2=-1).sum(-1) / (scene.mass.clamp_min(1e-30) * scene.length**2)
        features = torch.cat((tags, scene.dof.float().mean(-1)[None, :, None].expand(batch, -1, -1), mass_ratio,
            torch.log1p(trace)[None, :, None].expand(batch, -1, -1), radius,
            state.v.norm(dim=-1, keepdim=True) * dt / scene.length, state.omega.norm(dim=-1, keepdim=True) * dt,
            physical[..., :3].norm(dim=-1, keepdim=True) / scene.length, physical[..., 3:].norm(dim=-1, keepdim=True),
            baseline.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).to(position.dtype),
            baseline.square().sum((-1, -2)).sqrt()[..., None].to(position.dtype),
            material / degree.clamp_min(1)[..., None], degree.log1p()[..., None], *geometry), -1)
        features = torch.where(mobile[None, :, None], features, 0.)
        dtype = self.factor.weight.dtype
        h = self.encoder(features.to(dtype))
        initial = h
        kind = torch.nn.functional.one_hot(getattr(layout,'legacy_kind',layout.relation_kind), 7).to(dtype)[None].expand(batch, -1, -1)
        extent = row_center.new_zeros(batch, len(layout.incidence), 1)
        for rows, ids in layout.slots:
            extent = extent.index_add(1, rows, (position[:, ids] - row_center[:, rows]).norm(dim=-1, keepdim=True) / scene.length)
        extent = extent / slots.clamp_min(1)[None, :, None]
        row_features = torch.cat((relation[..., 9:18], kind, relation[..., :9].norm(dim=-1, keepdim=True), relation[..., 18:27].norm(dim=-1, keepdim=True), extent), -1).to(dtype)
        for _ in range(2):
            if getattr(self,'relation_chunk_rows',0) and self.training and torch.is_grad_enabled():
                gathered=self.stream_local_messages(h,row_features,active,slots,layout.slots)
            else:
                pooled = h.new_zeros(batch, len(layout.incidence), h.shape[-1])
                for rows, ids in layout.slots:
                    pooled = pooled.index_add(1, rows, h[:, ids])
                message = self.message(torch.cat((pooled / slots.clamp_min(1).to(dtype)[None, :, None], row_features), -1))
                message = message * active.to(dtype)[..., None]
                gathered = torch.zeros_like(h)
                for rows, ids in layout.slots:
                    gathered = gathered.index_add(1, ids, message[:, rows])
            updated = h + self.update(torch.cat((h, gathered / degree.clamp_min(1).to(dtype)[..., None]), -1))
            h = torch.where(mobile[None, :, None], updated, initial)
        return h, features

    def forward(self, scene, state, residual, whitening, baseline, relation, physical, coarse, metric, dt):
        h, _ = self.encode(scene, state, physical, baseline, relation, dt)
        position = centers(scene, state).double()
        dof = scene.dof
        w = whitening.double()
        safe_w = w + torch.diag_embed((~dof).double())[None]
        root, info = torch.linalg.cholesky_ex(baseline.double(), check_errors=False)
        valid = (info == 0).all(-1) & torch.isfinite(root).flatten(1).all(-1)
        result = -(w @ baseline.double() @ (w @ residual.double()[..., None]))[..., 0]
        for ids in scene.interaction_entities:
            mask = scene.movable[ids]
            mass = scene.mass[ids].double() * mask
            center = (position[:, ids] * mass[None, :, None]).sum(1, keepdim=True) / mass.sum().clamp_min(1e-30)
            shape = 1 + .5 * self.shape(h[:, ids]).tanh().double()
            eye = torch.eye(3, device=position.device, dtype=torch.float64)
            raw = position.new_zeros(len(position), len(ids), 6, 6)
            raw[..., :3, :3] = scene.length * shape[..., :1, None] * eye
            raw[..., :3, 3:] = -shape[..., 1:, None] * skew(position[:, ids] - center)
            raw[..., 3:, 3:] = shape[..., 1:, None] * eye
            raw = raw * dof[ids][None, :, :, None]
            full = raw.new_zeros(len(position), scene.n, 6, 6).index_copy(1, ids, raw)
            full = full - coarse(metric_action(metric, full))
            local = full[:, ids]
            transformed = torch.linalg.solve_ex(safe_w[:, ids], local, check_errors=False)
            z = torch.linalg.solve_triangular(root[:, ids], transformed.result, upper=False)
            z = z / z.square().sum((1, 2, 3), keepdim=True).clamp_min(1e-24).sqrt()
            pooled = (h[:, ids] * mask[None, :, None]).sum(1) / mask.sum().clamp_min(1)
            a, b, c = (.5 * self.bound * self.factor(pooled).tanh().double()).unbind(-1)
            two = torch.stack((torch.stack((a, b), -1), torch.stack((b, c), -1)), -2)
            t = torch.einsum('bij,uv->biujv', two, eye).reshape(len(position), 6, 6)
            s = safe_w[:, ids] @ root[:, ids]
            y = (s.transpose(-1, -2) @ residual[:, ids].double()[..., None])[..., 0]
            def modulation(value):
                coefficient = torch.einsum('bnip,bni->bp', z, value)
                return torch.einsum('bnip,bp->bni', z, (t @ coefficient[..., None])[..., 0])
            first = modulation(y)
            delta = (s @ (first + modulation(y + first))[..., None])[..., 0]
            result = result.index_add(1, ids, -delta)
            valid &= (transformed.info == 0).all(-1) & torch.isfinite(z).flatten(1).all(-1) & torch.isfinite(delta).flatten(1).all(-1)
        return result * dof, valid


def protected_update(model, scene, state, metric, raw_coarse, residual, whitening, baseline, relation, physical, dt, learned, *, return_parts=False):
    independent=bool(scene.metadata.get('independent_object_motion_modes'))
    if independent and (scene.metadata.get('contract')!='elastic_surface_joint_contact_v38' or not scene.metadata.get('joint_tree_coarse_modes')):
        raise ValueError('Independent object modes require the surface/joint tree contract')
    # The packed six-column factor cannot represent two independent objects
    # plus tree columns. Use the declared 18-column energy embedding even
    # when the surrounding sparse operations are batched by component.
    if getattr(scene,'component_batch',False) and not independent:
        from .component_batch import BatchedRigidCoarse
        coarse=BatchedRigidCoarse(scene,metric,raw_coarse.blocks)
    else:
        maximum=18 if scene.metadata.get('contract') in ('rod_shape_contact_com_v40','elastic_rod_surface_joint_self_contact_v222') and model.config.get('rod_gravity_material_ports',False) else 12
        if independent:maximum=18
        coarse = ProtectedCoarse(scene, metric, raw_coarse.blocks,maximum_columns=maximum,
                                exact_rigid_singletons=bool(getattr(scene,'fast_rigid_contact',False)) and not independent)
    source = residual.double()
    coarse_r = coarse(source)
    complement = source - metric_action(metric, coarse_r)
    w = whitening.double()
    gain=1.
    if learned:
        if model.config.get('shared_constitutive_coarse'):
            local,valid,gain=model.protected_response(scene,state,complement,w,baseline,relation,physical,coarse,metric,dt,return_coarse_gain=True)
            gain=gain[...,None]
        else:
            local, valid = model.protected_response(scene, state, complement, w, baseline, relation, physical, coarse, metric, dt)
    else:
        local = -(w @ baseline.double() @ (w @ complement[..., None]))[..., 0]
        valid = torch.ones(len(source), device=source.device, dtype=torch.bool)
    protected_part = -gain*coarse_r
    complementary_part = local - coarse(metric_action(metric, local))
    if scene.metadata.get('independent_object_motion_modes'):
        # A complete coarse basis leaves an exactly empty complement.
        # Forming B r - Q H B r numerically leaves signed roundoff that
        # can spuriously fail the strict descent test on a FREE singleton.
        for index,ids in enumerate(scene.interaction_entities):
            if not coarse.groups[index][1].shape[-1]:continue
            complete=coarse.ranks[index]>=scene.dof[ids].sum()
            complementary_part=complementary_part.index_copy(1,ids,
                torch.where(complete[:,None,None],0.,complementary_part[:,ids]))
    update = protected_part + complementary_part
    if return_parts:
        return update.to(residual.dtype), coarse_r.to(residual.dtype), coarse.valid & valid, protected_part.to(residual.dtype)
    return update.to(residual.dtype), coarse_r.to(residual.dtype), coarse.valid & valid
