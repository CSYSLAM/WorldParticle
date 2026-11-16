"""Disjoint rigid component operators with one sparse metric traversal.

Each component has at most six columns and independent small factors. Shared
column numbers never imply a shared coefficient across physical components.
"""
import torch


class ComponentLayout:
    def __init__(self, scene):
        mixed=scene.metadata.get('contract') in ('elastic_shape_contact_v33','elastic_surface_joint_contact_v38','elastic_surface_joint_self_contact_v58','elastic_surface_joint_dynamic_proxy_v300')
        if not mixed and scene.metadata.get('contract') not in ('rigid_contact_primal_dual_v21', 'rigid_joint_contact_finite_v26'):
            raise ValueError('Component batching currently requires a rigid six-DOF contract')
        if mixed:
            expected=scene.movable[:,None]&torch.cat((torch.ones(scene.n,3,device=scene.device,dtype=torch.bool),(scene.kind==2)[:,None].expand(-1,3)),1)
            if not bool(((scene.kind>=0)&(scene.kind<=2)).all()) or not torch.equal(scene.dof,expected):
                raise ValueError('Mixed component batching requires particle translation and rigid six-DOF masks')
        elif not bool((scene.kind == 2).all()) or not torch.equal(scene.dof, scene.movable[:, None].expand(-1, 6)):
            raise ValueError('Component batching requires complete rigid response DOFs')
        self.scene = scene
        self.keys, self.labels = torch.unique(scene.interaction_id, sorted=True, return_inverse=True)
        self.count = len(self.keys)
        self.mobile_count = torch.zeros(self.count, device=scene.device, dtype=torch.float64).index_add(0, self.labels, scene.movable.double())
        self.active = self.mobile_count > 0
        self.mass_total = torch.zeros_like(self.mobile_count).index_add(0, self.labels, scene.mass * scene.movable)
        self.fixed_count = torch.zeros_like(self.mobile_count).index_add(0, self.labels, (~scene.movable).double())

    def sum(self, value):
        return value.new_zeros(value.shape[0], self.count, *value.shape[2:]).index_add(1, self.labels, value)

    def pool(self, value):
        return self.sum(value * self.scene.movable[None, :, None]) / self.mobile_count.to(value.dtype).clamp_min(1)[None, :, None]

    def centers(self, position):
        mass = (self.scene.mass * self.scene.movable).to(position.dtype)
        return self.sum(position * mass[None, :, None]) / self.mass_total.to(position.dtype).clamp_min(1e-30)[None, :, None]

    def row_sum(self, value, labels):
        indices = labels.reshape(*labels.shape, *((1,) * (value.ndim - 2))).expand_as(value)
        return value.new_zeros(value.shape[0], self.count, *value.shape[2:]).scatter_add(1, indices, value)

    def gram(self, metric, columns):
        """Positive component Gram; repeated owner roles combine before JtWJ."""
        from .protected_response import metric_terms
        value = columns.double() * self.scene.dof[None, :, :, None]
        local = torch.einsum('bnip,bnij,bnjq->bnpq', value, metric.local.double(), value)
        gram = self.sum(local)
        valid = torch.isfinite(metric.local).flatten(1).all(-1) & torch.isfinite(value).flatten(1).all(-1)
        for term in metric_terms(self.scene, metric):
            jacobian, weight = term.jacobian.double(), term.weight.double()
            valid &= torch.isfinite(jacobian).flatten(1).all(-1) & torch.isfinite(weight).flatten(1).all(-1) & (weight >= 0).flatten(1).all(-1)
            if not len(term.ids): continue
            effective = ((jacobian != 0) & (weight[..., None, None] != 0)).any(-1).any(2) & self.scene.movable[term.ids][None]
            groups = self.labels[term.ids][None].expand(len(value), -1, -1)
            first = effective.long().argmax(-1, keepdim=True)
            selected = groups.gather(-1, first)[..., 0]
            valid &= ~(effective & (groups != selected[..., None])).flatten(1).any(-1)
            image = torch.einsum('brcsi,brsip->brcp', jacobian, value[:, term.ids])
            row = torch.einsum('brcp,brc,brcq->brpq', image, weight, image)
            gram = gram + self.row_sum(row, selected)
        return gram, valid & torch.isfinite(gram).flatten(1).all(-1)


def layout(scene):
    key = (int(scene.topology_generation), id(scene.interaction_id), scene.interaction_id._version)
    previous = getattr(scene, '_component_batch_layout', None)
    if previous is None or previous[0] != key:
        previous = (key, ComponentLayout(scene)); scene._component_batch_layout = previous
    return previous[1]


def context_features(scene, state, physical, position, dt):
    groups = layout(scene); labels = groups.labels
    center = groups.centers(position)
    fixed = (~scene.movable).to(position.dtype)
    support = groups.sum(position * fixed[None, :, None]) / groups.fixed_count.to(position.dtype).clamp_min(1)[None, :, None]
    support = torch.where((groups.fixed_count > 0)[None, :, None], support, center)
    activity = (physical[..., :3].norm(dim=-1) / scene.length + physical[..., 3:].norm(dim=-1)) * scene.movable
    total = groups.sum(activity[..., None])
    drive = groups.sum(position * activity[..., None]) / total.clamp_min(1e-30)
    drive = torch.where(total > 1e-20, drive, center)
    axis = drive - support; length = axis.norm(dim=-1, keepdim=True)
    signed = ((position - support[:, labels]) * axis[:, labels]).sum(-1, keepdim=True) / (length[:, labels].clamp_min(1e-12) * scene.length)
    signed = torch.where(length[:, labels] > scene.length * 1e-8, signed, 0.)
    work = (physical[..., :3] * state.v).sum(-1, keepdim=True) * dt / scene.length**2
    work = work + (physical[..., 3:] * state.omega).sum(-1, keepdim=True) * dt
    return torch.cat(((position - support[:, labels]).norm(dim=-1, keepdim=True) / scene.length * (groups.fixed_count[labels] > 0)[None, :, None], (position - drive[:, labels]).norm(dim=-1, keepdim=True) / scene.length, signed, work), -1)


class BatchedRigidCoarse:
    """Independent QQt spaces, each with a six-by-six physical factor.

    Correlation conditioning certifies that all six columns pass the original
    relative rank cutoff. Difficult/rank-deficient spaces use the old energy
    embedding and two MGS passes, without a ridge repair or VBD fallback.
    """
    def __init__(self, scene, metric, blocks, *, rank_relative=1e-12):
        self.layout = layout(scene); self.packed = False; self.fallback = None
        from .protected_response import ProtectedCoarse
        with torch.no_grad():
            batch = len(metric.local)
            modes = metric.local.new_zeros(batch, scene.n, 6, 6)
            if not blocks:
                self.fallback = ProtectedCoarse(scene, metric, blocks); return
            if any(z.shape[-1] != 6 for _, z, *_ in blocks):
                self.fallback = ProtectedCoarse(scene, metric, blocks); return
            ids = torch.cat([plan.ids for plan, *_ in blocks])
            if len(ids) != int(scene.movable.sum()) or len(torch.unique(ids)) != len(ids) or not bool(scene.movable[ids].all()):
                raise ValueError('Rigid component modes must cover every mobile entity exactly once')
            expected = torch.cat([self.layout.labels[plan.ids[:1]].expand(len(plan.ids)) for plan, *_ in blocks])
            if not torch.equal(expected, self.layout.labels[ids]): raise ValueError('A coarse block crosses interaction components')
            modes = modes.index_copy(1, ids, torch.cat([z.double() for _, z, *_ in blocks], dim=1))
            gram, valid = self.layout.gram(metric, modes)
            active = self.layout.active
            eye = torch.eye(6, device=scene.device, dtype=torch.float64)
            ghost = (~active)[None, :, None, None] * eye
            root, info = torch.linalg.cholesky_ex(gram + ghost, check_errors=False)
            diagonal = gram.diagonal(dim1=-2, dim2=-1)
            scale = diagonal.clamp_min(1e-300).rsqrt()
            correlation = gram * scale[..., :, None] * scale[..., None, :]
            correlation = torch.where(active[None, :, None, None], correlation, eye)
            # Avoid eigensolver nonfinite input even for a rejected metric.
            finite = torch.isfinite(correlation).flatten(1).all(-1)
            self.valid = valid & finite
            safe_correlation = torch.where(finite[:, None, None, None], correlation, eye)
            eigen = torch.linalg.eigvalsh(safe_correlation)
            safely_full_rank = (eigen[..., 0] > max(1e-8, 100 * rank_relative)) | ~active[None]
            if not bool((safely_full_rank & (info == 0)).all()) or not bool(valid.all() & finite.all()):
                self.fallback = ProtectedCoarse(scene, metric, blocks)
                # Rank fallback must never erase a stale graph/invalid metric
                # rejection discovered by the component ownership audit.
                self.valid &= self.fallback.valid
                return
            self.q = torch.linalg.solve_triangular(root[:, self.layout.labels], modes.transpose(-1, -2), upper=False).transpose(-1, -2)
            self.valid = valid & torch.isfinite(self.q).flatten(1).all(-1)
            self.packed = True
            self.complete = (self.layout.mobile_count <= 1) & active
            self.all_mobile_singletons = bool((self.complete | ~active).all())

    def __getattr__(self, name):
        if name in ('fallback', 'packed'): raise AttributeError(name)
        fallback = self.__dict__.get('fallback')
        if fallback is not None: return getattr(fallback, name)
        raise AttributeError(name)

    def __call__(self, value):
        if self.fallback is not None: return self.fallback(value)
        vector = value.ndim == 3
        source = value.double()[..., None] if vector else value.double()
        coefficient = self.layout.sum(self.q.transpose(-1, -2) @ source)
        result = self.q @ coefficient[:, self.layout.labels]
        return (result[..., 0] if vector else result).to(value.dtype)


def vector_response(module, scene, state, residual, whitening, baseline, relation, physical, coarse, metric, dt, return_coarse_gain):
    """The existing positive inverse-replacement formula, reduced by group."""
    from .protected_response import metric_action
    groups = coarse.layout; labels = groups.labels
    w = whitening.double(); source = residual.double()
    local = -(w @ baseline.double() @ (w @ source[..., None]))[..., 0]
    h, _ = module.encode(scene, state, physical, baseline, relation, dt) if module.mode == 'learned' else (physical.new_zeros(len(physical), scene.n, module.width), None)
    raw = module.make_ports(scene, state, -local, h, relation, dt, unit_shapes=module.mode == 'physical_vector')
    projected = raw - coarse(metric_action(metric, raw))
    safe_w = w + torch.diag_embed((~scene.dof).double())[None]
    root, info = torch.linalg.cholesky_ex(baseline.double(), check_errors=False)
    factor = safe_w @ root
    before, ib = torch.linalg.solve_ex(factor, raw, check_errors=False)
    after, ia = torch.linalg.solve_ex(factor, projected, check_errors=False)
    original = groups.sum(before.square().sum(-2))
    norm = groups.sum(after.square().sum(-2))
    keep = (norm > module.rank_relative * original) & (original > 0) & torch.isfinite(norm) & ~coarse.complete[None, :, None]
    scale = torch.where(keep, norm.clamp_min(1e-300).rsqrt(), 0.).detach()
    full = projected * scale[:, labels, None, :]
    gh, metric_valid = groups.gram(metric, full)
    z, iz = torch.linalg.solve_ex(factor, full, check_errors=False)
    gb = groups.sum(z.transpose(-1, -2) @ z)
    normalization = gb.diagonal(dim1=-2, dim2=-1).sum(-1).clamp_min(1e-24).rsqrt()
    u = full * normalization[:, labels, None, None]
    gb = gb * normalization[..., None, None].square(); gh = gh * normalization[..., None, None].square()
    eye = torch.eye(6, device=source.device, dtype=source.dtype)
    b = gb + module.ridge * eye; positive_h = gh + module.ridge * eye
    lb, i_b = torch.linalg.cholesky_ex(b, check_errors=False)
    trace = torch.cholesky_solve(positive_h, lb).diagonal(dim1=-2, dim2=-1).sum(-1)
    finite = torch.isfinite(trace) & (trace > 0)
    negative = .5 / torch.where(finite, trace, 1.).clamp_min(1.)
    logit = module.factor(groups.pool(h))[..., 0].double() if module.mode == 'learned' else trace.new_full(trace.shape, 30.)
    positive = -torch.expm1(-logit.clamp_min(0))
    below = negative * torch.expm1(logit.clamp_max(0) / negative)
    gate = torch.where(logit >= 0, positive, below)
    mix = b + gate[..., None, None] * (gh - gb)
    lm, i_m = torch.linalg.cholesky_ex(mix, check_errors=False)
    restricted = groups.sum((u.transpose(-1, -2) @ source[..., None])[..., 0])
    first = torch.cholesky_solve(restricted[..., None], lb)
    second = torch.cholesky_solve((gh - gb) @ first, lm)[..., 0]
    delta = -gate[:, labels, None] * (u @ second[:, labels, :, None])[..., 0]
    result = (local - delta) * scene.dof
    valid = metric_valid & finite.all(-1) & (i_b == 0).all(-1) & (i_m == 0).all(-1)
    valid &= (info == 0).all(-1) & (ib == 0).all(-1) & (ia == 0).all(-1) & (iz == 0).all(-1)
    valid &= torch.isfinite(root).flatten(1).all(-1) & torch.isfinite(delta).flatten(1).all(-1)
    if return_coarse_gain:
        from .complete_response import coarse_gain
        gain = coarse_gain(module,scene,h,relation,coarse) if module.mode == 'learned' else h.new_ones(*h.shape[:2])
        return result, valid, gain.double()
    return result, valid
