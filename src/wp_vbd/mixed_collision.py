"""Owned particle/shape and rigid/rigid collision rows in one COM index space.

Newton's collision pipeline supplies geometry only. Particle/shape records
use the shape as endpoint zero and the particle as endpoint one. Normals
therefore point from the shape to the particle, as in the native source.
Full-surface edge/face contacts require a separate barycentric operator.
"""
from dataclasses import dataclass, fields

import torch

from . import math3d as so3
from .rigid_collision import RigidContacts


@dataclass
class MixedContacts(RigidContacts):
    relation_kind: torch.Tensor
    source_kind: torch.Tensor

    @classmethod
    def snapshot(cls, model, contacts, wp, state, *, dtype=torch.float64):
        rigid = RigidContacts.snapshot(model, contacts, wp, dtype=dtype)
        particle_count = model.particle_count
        world = particle_count + model.body_count
        count = int(wp.to_torch(contacts.soft_contact_count)[0])
        capacity = len(contacts.soft_contact_particle)
        if count < 0 or count > capacity:
            raise ValueError(f'Soft collision overflow: {count}/{capacity}')
        def take(name):
            return wp.to_torch(getattr(contacts, name))[:count].clone()
        particle = take('soft_contact_particle').long()
        shape = take('soft_contact_shape').long()
        indices = take('soft_contact_indices').long()
        barycentric = take('soft_contact_barycentric').to(dtype)
        if bool(((particle < 0) | (particle >= particle_count)).any()):
            raise ValueError('Edge/face soft contact requires a barycentric physical operator')
        if bool(((shape < 0) | (shape >= model.shape_count)).any()):
            raise ValueError('Invalid soft collision shape')
        expected = torch.stack((particle, -torch.ones_like(particle), -torch.ones_like(particle)), -1)
        if not torch.equal(indices, expected) or not torch.equal(barycentric, barycentric.new_tensor([1., 0., 0.]).expand_as(barycentric)):
            raise ValueError('Invalid particle-only contact metadata')
        body = wp.to_torch(model.shape_body).long()[shape]
        if bool(((body < -1) | (body >= model.body_count)).any()):
            raise ValueError('Invalid soft collision owner')
        owner = torch.stack((torch.where(body < 0, world, body + particle_count), particle), -1)
        position = take('soft_contact_body_pos').to(dtype)
        velocity = take('soft_contact_body_vel').to(dtype)
        normal = take('soft_contact_normal').to(dtype)
        if any(not bool(torch.isfinite(v).all()) for v in (position, velocity, normal)):
            raise ValueError('Nonfinite soft collision geometry')
        if not bool(((normal.square().sum(-1) - 1).abs() < 1e-5).all()):
            raise ValueError('Invalid soft collision normal')
        normal = normal / normal.norm(dim=-1, keepdim=True)
        def material(name):
            return wp.to_torch(getattr(model, name))[shape].clone().to(dtype)
        ke = .5 * (material('shape_material_ke') + model.soft_contact_ke)
        kd = .5 * (material('shape_material_kd') + model.soft_contact_kd)
        soft_mu = normal.new_tensor(model.soft_contact_mu)
        shape_mu = material('shape_material_mu')
        margins = torch.stack((material('shape_margin'), wp.to_torch(model.particle_radius)[particle].clone().to(dtype)), -1)
        if any(not bool(torch.isfinite(v).all()) or bool((v < 0).any()) for v in (ke, kd, soft_mu, shape_mu, margins)):
            raise ValueError('Invalid soft collision material or radius')
        mu = (shape_mu * soft_mu).sqrt()
        # Surface velocities in source soft records are body-local. The
        # shared row consumes the world velocity frozen at detection time.
        velocity = so3.rotate(state.q[0, owner[:, 0]], velocity)
        rigid_values = {f.name: getattr(rigid, f.name) for f in fields(RigidContacts)}
        rigid_values['owners'] = rigid.owners + particle_count
        def join(name, value):
            rigid_values[name] = torch.cat((rigid_values[name], value), 0)
        join('owners', owner)
        join('shapes', torch.stack((shape, -torch.ones_like(shape)), -1))
        join('points', torch.stack((position, torch.zeros_like(position)), 1))
        join('offsets', normal.new_zeros(count, 2, 3))
        join('normal', normal)
        join('margins', margins)
        join('surface_velocity', velocity)
        join('material', ke); join('damping', kd); join('mu', mu)
        # The particle radius belongs to physical geometry. Only the shape
        # activation margin is removed when reporting physical separation:
        # shape_margin - depth = centre_distance - particle_radius.
        join('physical_margin', margins[:, 0])
        join('source_slots', torch.arange(count, device=normal.device) + rigid.capacity)
        join('point_ids', torch.full((count,), -1, device=normal.device, dtype=torch.long))
        rigid_values['source_count'] += count
        rigid_values['capacity'] += capacity
        rigid_rows = len(rigid.owners)
        kinds = torch.cat((torch.full((rigid_rows,), 7, device=normal.device, dtype=torch.long), torch.full((count,), 8, device=normal.device, dtype=torch.long)))
        sources = torch.cat((torch.zeros(rigid_rows, device=normal.device, dtype=torch.long), torch.ones(count, device=normal.device, dtype=torch.long)))
        return cls(**rigid_values, relation_kind=kinds, source_kind=sources)
