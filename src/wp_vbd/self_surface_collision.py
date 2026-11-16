"""One ordered rigid/soft/VF/EE contact snapshot in real-owner indexing."""
import torch


class SelfSurfaceContacts:
    def __init__(self, base, snapshot, *, stiffness, damping, friction):
        self.base = base
        self.self_snapshot = snapshot
        self.self_contacts = snapshot.contacts
        self.rigid, self.soft = base.rigid, base.soft
        self.soft_material = base.soft_material
        self.rigid_count, self.surface_count = base.rigid_count, len(base.owners)
        self.particle_count, self.generation = base.particle_count, base.generation
        c = self.self_contacts
        count = len(c.kind)
        scalar = c.normal.new_tensor([stiffness, damping, friction])
        if not bool(torch.isfinite(scalar).all()) or bool((scalar < 0).any()):
            raise ValueError('Finite nonnegative self contact material required')
        self.self_material = tuple(scalar[i].expand(count) for i in range(3))
        self.owners = torch.cat((base.owners, c.ports.owners))
        self.material = torch.cat((base.material, self.self_material[0]))
        self.damping = torch.cat((base.damping, self.self_material[1]))
        self.mu = torch.cat((base.mu, self.self_material[2]))
        self.normal = torch.cat((base.normal, c.normal))
        self.physical_margin = torch.cat((base.physical_margin, c.margin))
        self.relation_kind = torch.cat((base.relation_kind, c.kind))
        self.source_kind = torch.cat((base.source_kind, c.kind-7))
        self.soft_feature_size = torch.cat((base.soft_feature_size, torch.zeros_like(c.kind)))

    def depth(self, scene, state):
        if getattr(scene,'self_contact_refresh_geometry',False):
            from .self_contact_refresh import fields
            distance=fields(self.self_contacts,state)[2]
        else:distance = self.self_contacts.geometry(state, scene.body_com,jacobian=False)[-1]
        return torch.cat((self.base.depth(scene, state), self.self_contacts.margin[None]-distance), 1)

    def normal_jacobian(self, scene, state):
        c = self.self_contacts
        if getattr(scene,'self_contact_refresh_geometry',False):
            from .self_contact_refresh import snapshots
            _,c=snapshots(c,state)
        row = c.ports.relative_normal_jacobian(state, scene.body_com, c.normal)
        return torch.cat((self.base.normal_jacobian(scene, state), row), 1)
