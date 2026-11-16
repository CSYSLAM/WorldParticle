"""Owned rigid and weighted soft rows for one shared contact iteration."""
import torch

from . import math3d as so3
from .response_operator import skew
from .rigid_collision import RigidContacts
from .soft_surface_ports import SoftSurfacePorts
from .surface_contact_rows import export_material


class SurfaceJointContacts:
    def __init__(self, rigid, soft, material, particle_count):
        self.rigid = rigid; self.soft = soft; self.soft_material = material
        self.rigid_count = len(rigid.owners)
        self.owners = torch.cat((torch.nn.functional.pad(rigid.owners, (0, 2), value=-1), soft.ports.owners))
        self.material = torch.cat((rigid.material, material.stiffness))
        self.damping = torch.cat((rigid.damping, material.damping))
        self.mu = torch.cat((rigid.mu, material.friction))
        self.normal = torch.cat((rigid.normal, soft.normal))
        self.physical_margin = torch.cat((rigid.physical_margin, soft.shape_margin))
        self.generation = rigid.generation
        self.relation_kind = torch.cat((torch.full_like(rigid.material, 7, dtype=torch.long), torch.full_like(soft.radius, 8, dtype=torch.long)))
        self.source_kind = torch.cat((torch.zeros_like(rigid.material, dtype=torch.long), torch.ones_like(soft.radius, dtype=torch.long)))
        self.soft_feature_size = torch.cat((torch.zeros_like(rigid.material, dtype=torch.long), soft.feature_size))
        self.particle_count = particle_count

    @classmethod
    def snapshot(cls, model, contacts, wp, newton):
        rigid = RigidContacts.snapshot(model, contacts, wp)
        rigid.owners = rigid.owners+model.particle_count
        soft = SoftSurfacePorts.snapshot(model, contacts, wp)
        material = export_material(model, soft, wp, newton)
        return cls(rigid, soft, material, model.particle_count)

    def depth(self, scene, state):
        return torch.cat((self.rigid.geometry(scene, state)[-1], self.soft.geometry(state, scene.body_com,jacobian=False)[-1]), 1)

    def normal_jacobian(self, scene, state):
        skeleton = self.rigid.geometry(scene, state)[2]
        lever = skeleton-state.x[:, self.rigid.owners]
        eye = torch.eye(3, device=scene.device, dtype=state.x.dtype)
        rigid_j = torch.cat((eye.expand(*lever.shape[:-1], 3, 3), -skew(lever)), -1)
        rigid_j = rigid_j*state.x.new_tensor([-1., 1.])[None, None, :, None, None]
        rigid_j = torch.nn.functional.pad(rigid_j, (0, 0, 0, 0, 0, 2))
        soft_j = self.soft.ports.relative_jacobian(self.soft.geometry(state, scene.body_com)[1])
        relative = torch.cat((rigid_j, soft_j), 1)
        same = (self.owners[:, :, None] == self.owners[:, None, :]) & (self.owners[:, :, None] >= 0)
        first = ~torch.tril(same, diagonal=-1).any(-1) & (self.owners >= 0)
        relative = torch.einsum('rst,brtci->brsci', same.to(relative.dtype), relative)*first[None, :, :, None, None]
        return torch.einsum('ri,brsij->brsj', self.normal, relative)
