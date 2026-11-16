"""Owned native mesh-feature records, evaluated at each current configuration.

This calls only Newton's geometry/ownership predicate, never a dynamics
solver. All candidate records remain available for penetration prevention.
"""
import torch
import warp as wp

from newton._src.geometry.soft_contacts_mesh import (
    _SoftContactFilterData, _make_soft_contact_filter_data, _soft_contact_force_enabled,
)

from . import math3d as so3


@wp.kernel
def select_mesh_forces(data: _SoftContactFilterData, enabled: wp.array[wp.int32]):
    i = wp.tid()
    enabled[i] = wp.int32(_soft_contact_force_enabled(i, data))


class MeshForceFilter:
    def __init__(self, model, contacts, count):
        self.model = model
        self.mesh_data = contacts._soft_contact_mesh_data
        if self.mesh_data is None:
            raise ValueError('Native mesh contact topology is required')
        overflow = contacts._soft_contact_source_overflow
        if overflow is not None and bool(wp.to_torch(overflow).any()):
            raise ValueError('Incomplete mesh candidates: native contact source overflow')
        self.data = _make_soft_contact_filter_data(model, contacts, model.particle_q, model.body_q)
        # Native collision buffers are reusable. Freeze record provenance,
        # preserving the immutable model/topology arrays and their owner.
        self.owned = {}
        for name in ('features', 'contact_count', 'contact_shapes', 'source_overflow', 'soft_contact_indices'):
            original = getattr(self.data, name)
            if original is not None:
                if name in ('features', 'contact_shapes', 'soft_contact_indices'):
                    self.owned[name] = wp.from_torch(wp.to_torch(original)[:count].clone(), dtype=original.dtype)
                else: self.owned[name] = wp.clone(original)
                setattr(self.data, name, self.owned[name])
        self.data.contact_max = count
        self.count = count

    def __call__(self, state, body_com):
        if len(state.x) != 1:
            raise ValueError('Native mesh filtering requires a single physical scene')
        n, b = self.model.particle_count, self.model.body_count
        # Native collision geometry is FP32. Conversions stay on the same
        # device as the shared physical state and never transport via CPU.
        particle = state.x[0, :n].detach().float().contiguous()
        rotation = state.q[0, n:n+b]
        origin = state.x[0, n:n+b]-so3.rotate(rotation, body_com[n:n+b])
        pose = torch.cat((origin, rotation), -1).detach().float().contiguous()
        self.data.particle_q = wp.from_torch(particle, dtype=wp.vec3)
        self.data.body_q = wp.from_torch(pose, dtype=wp.transform)
        result = wp.empty(self.count, dtype=wp.int32, device=self.model.device)
        if self.count:
            wp.launch(select_mesh_forces, dim=self.count, inputs=[self.data, result], device=self.model.device)
        return wp.to_torch(result).bool()[None]
