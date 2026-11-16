"""Sparse real-owner point ports for weighted surface and self contact.

Endpoint interpolation introduces no independent virtual-particle DOF.
"""
import torch

from . import math3d as so3
from .mixed_metric import MetricTerm
from .response_operator import skew


class PointPorts:
    """Two contact points mapped to at most four real physical owners.

    owners: [rows, slots], -1 for padding; weights: [rows, 2, slots].
    A rigid endpoint is one real body-local point. A material endpoint is
    a nonnegative affine combination of particle positions. Geometry and
    barycentric weights are frozen by default. Particle self-contact may
    explicitly retain derivatives of the current closest-point weights;
    its positive metric uses a separate frozen port set.
    """
    def __init__(self, owners, weights, rigid, local_points, body_mask, *, differentiable_weights=False):
        if type(differentiable_weights) is not bool or differentiable_weights and bool(rigid.any()):
            raise ValueError('Differentiable weights require explicit particle-only ports')
        if owners.dtype != torch.long or owners.ndim != 2 or not 1 <= owners.shape[1] <= 4:
            raise ValueError('One to four integer physical owner slots required')
        rows, slots = owners.shape
        if weights.shape != (rows, 2, slots) or weights.dtype not in (torch.float32, torch.float64):
            raise ValueError('Two affine endpoint weight vectors required')
        if rigid.shape != (rows, 2) or rigid.dtype != torch.bool:
            raise ValueError('Explicit rigid endpoint types required')
        if local_points.shape != (rows, 2, 3) or not local_points.is_floating_point():
            raise ValueError('Body-local point coordinates required')
        if body_mask.ndim != 1 or body_mask.dtype != torch.bool or not len(body_mask):
            raise ValueError('Physical node ownership mask required')
        if any(value.device != owners.device for value in (weights, rigid, local_points, body_mask)):
            raise ValueError('Port arrays must share a device')
        if bool(((owners < -1) | (owners >= len(body_mask))).any()):
            raise ValueError('Contact owner is outside the physical scene')
        if not bool(torch.isfinite(weights).all() & torch.isfinite(local_points).all()):
            raise ValueError('Nonfinite contact port data')
        total = weights.double().sum(-1, keepdim=True)
        tolerance = max(1e-12, 8*torch.finfo(weights.dtype).eps)
        if bool((weights < 0).any()) or bool(((total-1).abs() > tolerance).any()):
            raise ValueError('Nonnegative endpoint weights must sum to one')
        if bool(((owners[:, None] < 0) & (weights != 0)).any()):
            raise ValueError('Padding cannot carry a contact weight')
        active = weights > 0
        if bool((active & (body_mask[owners.clamp_min(0)][:, None] != rigid[..., None])).any()):
            raise ValueError('A port cannot mix body and material ownership')
        if bool((rigid & (active.sum(-1) != 1)).any()):
            raise ValueError('A rigid point has exactly one physical body')
        self.owners = owners.detach().clone()
        self.ids = self.owners.clamp_min(0)
        # Native barycentric weights are FP32. Restore affine partition in
        # FP64 within their representation error, so common translation and
        # reciprocal forces do not inherit a sum-of-weights drift.
        affine=weights.double()/total
        self.weights = affine.clone() if differentiable_weights else affine.detach().clone()
        self.rigid = rigid.detach().clone()
        self.has_rigid = bool(self.rigid.any())
        self.local_points = local_points.detach().clone()
        self.body_mask = body_mask.detach().clone()
        self.node_count = len(body_mask)
        same=(self.owners[:,:,None]==self.owners[:,None,:])&(self.owners[:,:,None]>=0)
        self.has_repeated_owners=bool(torch.tril(same,diagonal=-1).any())

    def _validate_state(self, state, body_com):
        batch = len(state.x)
        if state.x.shape != (batch, self.node_count, 3) or state.q.shape != (batch, self.node_count, 4):
            raise ValueError('Physical state shape does not match ports')
        if body_com.shape != (self.node_count, 3):
            raise ValueError('COM offsets must use physical owner indexing')
        if not bool(torch.isfinite(state.x).all() & torch.isfinite(state.q).all() & torch.isfinite(body_com).all()):
            raise ValueError('Nonfinite physical port state')
        if bool((state.q[:, self.body_mask].norm(dim=-1) < 1e-12).any()):
            raise ValueError('Degenerate body orientation')

    def geometry(self, state, body_com, *, jacobian=True, point_chunk_rows=65536):
        if point_chunk_rows is not None and (type(point_chunk_rows) is not int or point_chunk_rows<=0):
            raise ValueError('Positive point chunk size or explicit unchunked reference required')
        self._validate_state(state, body_com)
        batch = len(state.x); rows, slots = self.owners.shape
        weights = self.weights.to(state.x)
        particle_only = not self.has_rigid and state.q.dtype==state.x.dtype==body_com.dtype
        if particle_only:
            # Particle ports have exactly zero angular columns. Keep the
            # previous zero-gradient connection to q/COM (and hence Adam's
            # zero-vs-None behavior) without repeating rotations per port.
            zero=state.q[:,0,0]*0+body_com[0,0]*0
            if not jacobian and not torch.is_grad_enabled() and point_chunk_rows is not None and rows>point_chunk_rows:
                # Merit/path checks need endpoint values, not a full broadcast
                # [batch,rows,2,slots,3] temporary. Preserve the exact original
                # row arithmetic, evaluating every row into the final buffer.
                # Gradient-enabled and Jacobian paths retain the old graph.
                point=state.x.new_empty(batch,rows,2,3)
                for first in range(0,rows,point_chunk_rows):
                    last=min(first+point_chunk_rows,rows)
                    point[:,first:last]=((state.x[:,self.ids[first:last]][:,:,None]+zero[:,None,None,None,None])*weights[None,first:last,...,None]).sum(-2)
            else:
                point=((state.x[:,self.ids][:,:,None]+zero[:,None,None,None,None])*weights[None,...,None]).sum(-2)
        else:
            q = so3.normalize(state.q[:, self.ids])[:, :, None].expand(-1, -1, 2, -1, -1)
            offset = self.local_points.to(state.x)[:, :, None] - body_com[self.ids][:, None]
            lever = so3.rotate(q, offset[None].expand(batch, -1, -1, -1, -1))
            lever = lever * self.rigid[None, :, :, None, None]
            point = ((state.x[:, self.ids][:, :, None] + lever) * weights[None, ..., None]).sum(-2)
        if not jacobian:return point,None
        identity = torch.eye(3, device=state.x.device, dtype=state.x.dtype)
        translation = weights[None, ..., None, None] * identity
        translation = translation.expand(batch, rows, 2, slots, 3, 3)
        rotation = torch.zeros_like(translation) if particle_only else -skew(lever) * weights[None, ..., None, None]
        return point, torch.cat((translation, rotation), -1)

    def relative_normal_jacobian(self, state, body_com, normal):
        """Project particle port derivatives without the two full 3x6 arrays.

        Distinct particle owners have scalar translation weights and exactly
        zero angular columns. Rigid or repeated owners keep the original
        contraction, including its coalescing and floating point order.
        """
        if normal.shape != (len(self.owners), 3) or normal.dtype != state.x.dtype or normal.device != state.x.device:
            raise ValueError('Normal rows must match the physical state dtype and device')
        if not bool(torch.isfinite(normal).all()):
            raise ValueError('Finite contact normals required')
        if self.has_rigid or self.has_repeated_owners or not state.q.dtype == state.x.dtype == body_com.dtype:
            jacobian = self.geometry(state, body_com)[1]
            return torch.einsum('ri,brsij->brsj', normal, self.relative_jacobian(jacobian))
        self._validate_state(state, body_com)
        weights = self.weights.to(state.x)
        translation = (weights[:, 1]-weights[:, 0])[..., None]*normal[:, None]
        row = torch.cat((translation, torch.zeros_like(translation)), -1)
        return row[None].expand(len(state.x), -1, -1, -1)

    def relative_jacobian(self, jacobian, *, rows=None):
        relative = jacobian[:, :, 1] - jacobian[:, :, 0]
        # Native nondegenerate contact features normally have distinct real
        # owners. Their padded columns are already exactly zero. Avoid a
        # second dense row tensor and an identity coalescing contraction.
        if not self.has_repeated_owners:return relative
        # Coalesce repeated physical owners before diagonal compliance or
        # metric construction. Opposite endpoint weights can cancel exactly.
        owners=self.owners if rows is None else self.owners[rows]
        same = (owners[:, :, None] == owners[:, None, :]) & (owners[:, :, None] >= 0)
        first = ~torch.tril(same, diagonal=-1).any(-1) & (owners >= 0)
        combined = torch.einsum('rst,brtci->brsci', same.to(relative.dtype), relative)
        return combined * first[None, :, :, None, None]

    def owner_wrenches(self, jacobian, covectors):
        if jacobian is not None:
            return torch.einsum('brpsci,brpc->brsi', jacobian, covectors)
        if self.has_rigid or covectors.shape[1:] != (len(self.owners), 2, 3):
            raise ValueError('Factorized pullback requires complete particle port covectors')
        force = torch.einsum('rps,brpc->brsc', self.weights.to(covectors), covectors)
        return torch.cat((force, torch.zeros_like(force)), -1)

    def point_velocity(self, jacobian, velocity, *, rows=None):
        if velocity.shape[1:] != (self.node_count, 6):
            raise ValueError('Complete physical six-DOF velocities required')
        if rows is not None:
            if rows.ndim!=1 or rows.dtype!=torch.long or rows.device!=self.owners.device or bool(((rows<0)|(rows>=len(self.owners))).any()):
                raise ValueError('Matched contact row indices required')
            ids=self.ids[rows]
            if jacobian is not None:jacobian=jacobian[:,rows]
        else:ids=self.ids
        if jacobian is not None:
            return torch.einsum('brpsci,brsi->brpc', jacobian, velocity[:, ids])
        if self.has_rigid:
            raise ValueError('Rigid port velocity requires its full Jacobian')
        weights=self.weights if rows is None else self.weights[rows]
        return torch.einsum('rps,brsc->brpc', weights.to(velocity), velocity[:, ids, :3])

    def pullback(self, jacobian, covectors):
        pair = self.owner_wrenches(jacobian, covectors)
        result = pair.new_zeros(len(pair), self.node_count, 6)
        return result.index_add(1, self.ids.flatten(), pair.flatten(1, 2))

    def metric_term(self, jacobian, port_metric, *, compact=False):
        if jacobian is None and (not compact or self.has_rigid or self.has_repeated_owners):
            raise ValueError('Factorized metric requires compact distinct particle ports')
        shape = (len(port_metric), len(self.owners), 3, 3) if jacobian is None else (*jacobian.shape[:2], 3, 3)
        if port_metric.shape != shape or not bool(torch.isfinite(port_metric).all()):
            raise ValueError('A finite 3x3 relative-point metric is required')
        scale = port_metric.abs().amax((-1, -2)).clamp_min(1.)
        if bool(((port_metric-port_metric.transpose(-1, -2)).abs().amax((-1, -2)) > 1e-12*scale).any()):
            raise ValueError('Point metric must be symmetric')
        symmetric=(port_metric+port_metric.transpose(-1, -2))/2
        selected=None;topology_key=None;ids=self.ids
        if compact:
            if port_metric.requires_grad:raise ValueError('Compaction requires a frozen contact metric')
            selected=torch.where((symmetric!=0).any((-1,-2)).any(0))[0]
            cache=getattr(self,'_metric_selection_cache',None)
            if cache is None or not torch.equal(cache[0],selected):
                cache=(selected.detach().clone(),object(),self.ids[selected])
                self._metric_selection_cache=cache
            selected,topology_key,ids=cache
            symmetric=symmetric[:,selected];scale=scale[:,selected]
            if jacobian is not None:jacobian=jacobian[:,selected]
        if port_metric.requires_grad:
            eigen,frame=so3.eigh_zero_safe(symmetric)
        else:
            # Frozen inactive contact blocks are exactly zero. Retain their
            # canonical zero spectrum/identity basis and every physical row,
            # without asking the CUDA eigensolver for a large zero batch.
            flat=symmetric.reshape(-1,3,3);active=torch.where((flat!=0).any(-1).any(-1))[0]
            eigen=flat.new_zeros(len(flat),3)
            frame=torch.eye(3,device=flat.device,dtype=flat.dtype).expand(len(flat),3,3)
            if len(active):
                values,vectors=so3.eigh_zero_safe(flat[active])
                eigen=eigen.index_copy(0,active,values);frame=frame.index_copy(0,active,vectors)
            eigen=eigen.reshape(*symmetric.shape[:-2],3);frame=frame.reshape(symmetric.shape)
        if bool((eigen < -1e-12*scale[..., None]).any()):
            raise ValueError('Point metric must be positive semidefinite')
        if jacobian is None:
            weights = self.weights[selected].to(port_metric)
            translation = (weights[:, 1]-weights[:, 0])[..., None, None]*torch.eye(3,device=port_metric.device,dtype=port_metric.dtype)
            relative = torch.cat((translation, torch.zeros_like(translation)), -1)[None].expand(len(port_metric), -1, -1, -1, -1)
        else:
            relative = self.relative_jacobian(jacobian,rows=selected)
        rows = torch.einsum('brac,brsai->brcsi', frame, relative)
        return MetricTerm(ids, rows, eigen.clamp_min(0),topology_key)

    def normal_compliance(self, jacobian, normal, inverse_blocks):
        row = torch.einsum('brc,brsci->brsi', normal, self.relative_jacobian(jacobian))
        return torch.einsum('brsi,brsij,brsj->br', row, inverse_blocks[:, self.ids], row)
