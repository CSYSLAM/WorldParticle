"""Owned native soft-feature geometry mapped to prototype point ports.

This is a geometry snapshot, not force eligibility, CCD or solver admission.
Rigid primitive metadata is retained even when its contact weight is zero.
"""
from dataclasses import dataclass

import torch

from . import math3d as so3
from .contact_ports import PointPorts
from .data import State
from .native_array import native_tensor


def native_state(model, native, wp):
    n, b = model.particle_count, model.body_count
    if n < 1:
        raise ValueError('A soft-port state requires actual material particles')
    particle = wp.to_torch(native.particle_q).double()
    pose = wp.to_torch(native.body_q).double() if b else None
    device = particle.device
    x = torch.zeros(1, n+b+1, 3, device=device, dtype=torch.float64)
    q = torch.zeros(1, n+b+1, 4, device=device, dtype=torch.float64); q[..., 3] = 1.
    com = torch.zeros(n+b+1, 3, device=device, dtype=torch.float64)
    if n: x[0, :n] = particle
    if b:
        if bool((pose[:, 3:].norm(dim=-1) < 1e-12).any()):
            raise ValueError('Degenerate native body orientation')
        com[n:n+b] = wp.to_torch(model.body_com).double()
        q[0, n:n+b] = so3.normalize(pose[:, 3:])
        x[0, n:n+b] = pose[:, :3]+so3.rotate(q[0, n:n+b], com[n:n+b])
    body = torch.arange(n+b+1, device=device) >= n
    return State(x, q, torch.zeros_like(x), torch.zeros_like(x)), com, body


@dataclass
class SoftSurfacePorts:
    ports: PointPorts
    source_slots: torch.Tensor
    shape: torch.Tensor
    corners: torch.Tensor
    feature_size: torch.Tensor
    rigid_primitive: torch.Tensor
    normal: torch.Tensor
    radius: torch.Tensor
    shape_margin: torch.Tensor
    body_local_velocity: torch.Tensor
    mesh_force_filter: object = None

    @classmethod
    def snapshot(cls, model, contacts, wp):
        count = int(wp.to_torch(contacts.soft_contact_count)[0])
        capacity = len(contacts.soft_contact_particle)
        if not 0 <= count <= capacity:
            raise ValueError('Native soft feature capacity overflow')
        def take(name):
            return native_tensor(wp, getattr(contacts, name))[:count].clone()
        corners = take('soft_contact_indices').long()
        bary = take('soft_contact_barycentric')
        shape = take('soft_contact_shape').long()
        particle_view = take('soft_contact_particle').long()
        valid = corners >= 0
        if bool(((corners < -1) | (corners >= model.particle_count)).any()):
            raise ValueError('Soft feature owner is outside the particle scene')
        if bool((~valid[:, 0] | (~valid[:, 1] & valid[:, 2])).any()):
            raise ValueError('Soft feature corners must use trailing -1 padding')
        feature_size = valid.sum(-1)
        expected = torch.where(feature_size == 1, corners[:, 0], -torch.ones_like(particle_view))
        if not torch.equal(particle_view, expected):
            raise ValueError('Particle-only view disagrees with full feature metadata')
        if bool(((shape < 0) | (shape >= model.shape_count)).any()):
            raise ValueError('Invalid native soft feature shape')
        body = native_tensor(wp, model.shape_body).long()[shape]
        if bool(((body < -1) | (body >= model.body_count)).any()):
            raise ValueError('Invalid rigid soft-feature owner')
        n, b = model.particle_count, model.body_count
        body = torch.where(body < 0, n+b, n+body)
        owners = torch.cat((body[:, None], corners), -1)
        weights = bary.new_zeros(count, 2, 4)
        weights[:, 0, 0] = 1.; weights[:, 1, 1:] = bary
        rigid = torch.zeros(count, 2, device=bary.device, dtype=torch.bool); rigid[:, 0] = True
        local = bary.new_zeros(count, 2, 3); local[:, 0] = take('soft_contact_body_pos')
        mask = torch.arange(n+b+1, device=bary.device) >= n
        ports = PointPorts(owners, weights, rigid, local, mask)
        normal = take('soft_contact_normal').double()
        if not bool(torch.isfinite(normal).all()) or bool(((normal.square().sum(-1)-1).abs() > 1e-5).any()):
            raise ValueError('Invalid native soft-feature normal')
        normal = normal/normal.norm(dim=-1, keepdim=True)
        # Native edge/face force uses MAX radius over all primitive corners,
        # including corners whose barycentric contact weight is zero.
        radii = wp.to_torch(model.particle_radius).double()
        radius = torch.where(valid, radii[corners.clamp_min(0)], 0.).amax(-1)
        margin = native_tensor(wp, model.shape_margin)[shape].clone().double()
        velocity = take('soft_contact_body_vel').double()
        if not bool(torch.isfinite(radius).all() & torch.isfinite(margin).all() & torch.isfinite(velocity).all()) or bool((radius < 0).any() | (margin < 0).any()):
            raise ValueError('Invalid soft-feature radius, margin or velocity')
        mesh_filter = None
        if getattr(contacts, '_soft_contact_mesh_data', None) is not None:
            from .surface_mesh_filter import MeshForceFilter
            mesh_filter = MeshForceFilter(model, contacts, count)
        return cls(ports, torch.arange(count, device=bary.device), shape, corners, feature_size,
                   take('soft_contact_rigid_indices').long(), normal, radius, margin, velocity, mesh_filter)

    def geometry(self, state, body_com, *, jacobian=True):
        point, jacobian = self.ports.geometry(state, body_com,jacobian=jacobian)
        distance = ((point[:, :, 1]-point[:, :, 0])*self.normal).sum(-1)
        depth = self.radius+self.shape_margin-distance
        return point, jacobian, depth
