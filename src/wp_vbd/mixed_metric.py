"""Frozen SPD mixed preconditioning metric, applied through sparse relations.

This is not the nonlinear residual Jacobian. Tet/membrane use their projected
positive metric, joints and bending use Gauss--Newton, and contact retains the
existing lagged positive local block. The physical residual is unchanged.
"""

from dataclasses import dataclass

import torch

from . import math3d as so3
from .data import State
from .mixed_physics import plane_system
from .response_operator import skew


@dataclass
class MetricTerm:
    ids: torch.Tensor
    jacobian: torch.Tensor
    weight: torch.Tensor
    topology_key: object = None


def frozen_contact_term(contacts, jacobian, weight, *, compact=False, cache_name):
    """Omit only exactly zero frozen metric rows, keeping candidate order.

    Geometry, constitutive forces and accepted contact history still retain
    every candidate. Cache identity must change when the selected incidence
    changes, even if its shape stays the same.
    """
    ids = contacts.owners
    if not compact:
        return MetricTerm(ids, jacobian, weight)
    if jacobian.requires_grad or weight.requires_grad:
        raise ValueError('Contact metric compaction requires frozen rows and weights')
    if not bool(torch.isfinite(jacobian).all()) or not bool(torch.isfinite(weight).all()):
        raise ValueError('Contact metric compaction requires finite rows and weights')
    selected = torch.where((weight != 0).any(0).any(-1))[0]
    selected_ids = ids[selected]
    cache = getattr(contacts, cache_name, None)
    if cache is None or not torch.equal(cache[0], selected) or not torch.equal(cache[2], selected_ids):
        cache = (selected.detach().clone(), object(), selected_ids.clone())
        setattr(contacts, cache_name, cache)
    selected, topology_key, ids = cache
    return MetricTerm(ids, jacobian[:, selected], weight[:, selected], topology_key)


def metric_topology(scene):
    base=[scene.tets,scene.triangles,scene.bend_edges[scene.bending_indices],scene.joints,scene.attachments]
    result=base+[ids for enabled,ids in zip(scene.material_damping_flags,(scene.tets,scene.triangles,scene.bend_edges[scene.bending_indices])) if enabled]
    if getattr(scene,'elastic_prestress_metric',False):result.extend((scene.tets,scene.triangles))
    return result


class FrozenMixedMetric:
    def __init__(self, scene, state, start, target_x, target_q, dt, extras, relation_jacobian, *, base=None,elastic_mode='projected'):
        self.compiled_response_kernels=bool(getattr(scene,'compiled_response_kernels',False))
        if elastic_mode not in ('projected','objective'):raise ValueError('Unknown coarse elastic metric')
        self.dof = scene.dof
        batch = state.x.shape[0]
        if base is None:
            local = state.x.new_zeros(batch, scene.n, 6, 6)
            local[..., :3, :3] = scene.mass[None, :, None, None] / dt**2 * torch.eye(3, device=state.x.device)
            rotation = so3.matrix(state.q)
            local[..., 3:, 3:] = rotation @ scene.inertia @ rotation.transpose(-1, -2) / dt**2
            forecast = State(target_x, target_q, start.v, start.omega)
            base = local + plane_system(scene, state, start, dt, forecast)[1]
        self.local = base.detach()
        self.terms = [];prestress=[]
        for key in ('tets', 'triangles'):
            ids = getattr(scene, key)
            dm = scene.tet_Dm if key == 'tets' else scene.tri_Dm
            points = state.x[:, ids]
            f = (points[:, :, 1:] - points[:, :, :1]).transpose(-1, -2) @ dm
            m = torch.cat((-dm.sum(-2, keepdim=True), dm), -2)
            dimensions = dm.shape[-1]
            jf = torch.einsum('ij,rsa->riasj', torch.eye(3, device=f.device, dtype=f.dtype), m)
            jf = jf.reshape(len(ids), 3*dimensions, len(ids[0]), 3) if len(ids) else f.new_zeros(0, 3*dimensions, dimensions+1, 3)
            jf = jf[None].expand(batch, -1, -1, -1, -1)
            material = scene.tet_material if key == 'tets' else scene.tri_material
            mu, lam = material[:, 0], material[:, :2].sum(-1)
            measure = scene.tet_volume if key == 'tets' else scene.tri_area
            if key == 'tets':
                derivative = torch.einsum('bria,rsa->brsi', so3.cofactor(f), m)
                jacobian = torch.cat((jf, derivative[:, :, None]), 2)
                weight = torch.cat((mu[:, None].expand(-1, 9), lam[:, None]), -1) * measure[:, None]
                weight = weight[None].expand(batch, -1, -1)
            else:
                f0, f1 = f[..., 0], f[..., 1]
                raw = torch.linalg.cross(f0, f1)
                js = raw.norm(dim=-1).clamp_min(1e-10)
                normal = raw / js[..., None]
                arm = f1[:, :, None] * m[None, :, :, :1] - f0[:, :, None] * m[None, :, :, 1:]
                cross_j = -skew(arm).permute(0, 1, 3, 2, 4)
                area_j = torch.einsum('bri,brisj->brsj', normal, cross_j)
                r = (lam[None]*(js-1)-mu[None]).clamp_min(0) / js
                jacobian = torch.cat((jf, area_j[:, :, None], cross_j), 2)
                weight = torch.cat((mu[None, :, None].expand(batch, -1, 6), (lam[None]-r)[..., None], r[..., None].expand(-1, -1, 3)), -1) * measure[None, :, None]
            if elastic_mode=='objective':
                # Frame-indifferent positive strain metric. Unlike projected
                # J_F, J_(F^T F) annihilates infinitesimal rigid rotations.
                # At rest this is the isotropic linear elastic Hessian.
                from .material_damping import metric_jacobian
                jc=metric_jacobian(scene,f,key)
                volume_j=derivative if key=='tets' else area_j
                jacobian=torch.cat((jc,volume_j[:,:,None]),2)
                weight=torch.cat((.5*mu[:,None].expand(-1,dimensions**2),material[:,1:2]),-1)*measure[:,None]
                weight=weight[None].expand(batch,-1,-1)
            padded = torch.nn.functional.pad(jacobian, (0, 3))
            self.terms.append(MetricTerm(ids, padded.detach(), weight.detach()))
            if getattr(scene,'elastic_prestress_metric',False):
                if elastic_mode!='objective':raise ValueError('Prestress correction requires the objective strain metric')
                from .elastic_prestress import positive_prestress_factor
                jf_stress,w_stress=positive_prestress_factor(f,dm,material,measure)
                prestress.append(MetricTerm(ids,jf_stress,w_stress))
        selected = scene.bending_indices
        if self.compiled_response_kernels and getattr(scene,'response_topology_cache',True) and not torch.is_grad_enabled():
            from .response_topology import bending_ids
            ids=bending_ids(scene)
        else:ids = scene.bend_edges[selected]
        sl, retained = scene.structural_slices['bend_edges']
        # MixedScene retains exactly the positive, valid bend edges here.
        if relation_jacobian is None:
            # The physical metric consumes only bending rows from the full
            # neural-relation Jacobian. Do not assemble discarded tetrahedral
            # feature Jacobians, parameters, rates and neural condition rows.
            if len(selected):
                with torch.enable_grad():
                    points=state.x[:,ids].detach().requires_grad_(True)
                    n1=so3.normalize(torch.linalg.cross(points[:,:,2]-points[:,:,0],points[:,:,3]-points[:,:,0]))
                    n2=so3.normalize(torch.linalg.cross(points[:,:,3]-points[:,:,1],points[:,:,2]-points[:,:,1]))
                    edge=so3.normalize(points[:,:,3]-points[:,:,2])
                    theta=torch.atan2((torch.linalg.cross(n1,n2)*edge).sum(-1),(n1*n2).sum(-1))
                    derivative=torch.autograd.grad(theta.sum(),points)[0]
                bend=torch.nn.functional.pad(derivative[:,:,None],(0,3))
            else:bend=state.x.new_zeros(batch,0,1,4,6)
        else:bend = relation_jacobian[:, sl, :1, :, :]
        weight = (scene.bend_material[selected, 0] * scene.bend_length[selected])[None, :, None].expand(batch, -1, -1)
        self.terms.append(MetricTerm(ids, bend.detach(), weight))
        self.terms.append(MetricTerm(scene.joints, extras[1].detach(), scene.joint_stiffness[None].expand(batch, -1, -1)))
        self.terms.append(MetricTerm(scene.attachments,extras[7].detach(),scene.attachment_stiffness[None,:,None].expand(batch,-1,3)))
        self.terms.extend(extras[8])
        self.terms.extend(prestress)

    @property
    def diagonal(self):
        if getattr(self,'compiled_response_kernels',False) and not torch.is_grad_enabled() and self.local.is_cuda:
            from .compiled_contact import evaluate as compiled
            return compiled(_diagonal_tensor,self.local,self.terms,self.dof)
        return _diagonal_tensor(self.local,self.terms,self.dof)

    def __call__(self, value):
        if getattr(self,'compiled_response_kernels',False) and not torch.is_grad_enabled() and value.is_cuda:
            from .compiled_contact import evaluate as compiled
            return compiled(_apply_tensor,self.local,self.terms,self.dof,value)
        return _apply_tensor(self.local,self.terms,self.dof,value)

    def gram(self, plan, modes):
        """Z^T H Z from sparse overlaps; no global entity-to-pool matrix."""
        local = self.local[:, plan.ids].to(modes.dtype)
        gram = torch.einsum('bnip,bnij,bnjq->bpq', modes, local, modes)
        for index, rows, ids, valid in plan.overlaps:
            term = self.terms[index]
            value = modes[:, ids] * valid[None, :, :, None, None]
            restricted = torch.einsum('brcsi,brsip->brcp', term.jacobian[:, rows].to(modes.dtype), value)
            gram = gram + torch.einsum('brcp,brc,brcq->bpq', restricted, term.weight[:, rows].to(modes.dtype), restricted)
        return gram


def _diagonal_tensor(local,terms,dof):
    diagonal=local.clone()
    for term in terms:
        weighted=term.jacobian*term.weight[...,None,None]
        contribution=torch.einsum('brcsi,brcsj->brsij',weighted,term.jacobian)
        diagonal=diagonal.index_add(1,term.ids.flatten(),contribution.flatten(1,2))
    return diagonal*dof[None,:,:,None]*dof[None,:,None,:]


def _apply_tensor(local,terms,dof,value):
    """Same ordered sparse J^T W J action; no dense global matrix."""
    mask=dof[None,:,:,None]
    value=value*mask
    result=local.to(value.dtype)@value
    for term in terms:
        jacobian=term.jacobian.to(value.dtype)
        restricted=torch.einsum('brcsi,brsiq->brcq',jacobian,value[:,term.ids])
        # Contract W with Jv first to retain the bounded-memory path.
        weighted=restricted*term.weight.to(value.dtype)[...,None]
        reaction=torch.einsum('brcsi,brcq->brsiq',jacobian,weighted)
        result=result.index_add(1,term.ids.flatten(),reaction.flatten(1,2))
    return result*mask
