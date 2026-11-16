"""Experimental balanced component/patch response for real mixed residuals."""

from dataclasses import dataclass

import numpy as np
import torch

from .coarse_response import rigid_component_modes
from .mixed_metric import metric_topology


@dataclass
class CoarsePlan:
    ids: torch.Tensor
    rigid: bool
    weight: float
    overlaps: list
    columns: torch.Tensor
    bending: object = None


class MixedCoarseLayout:
    """Build once outside CUDA capture; only sparse incidence overlaps persist."""
    def __init__(self, scene, mode='rigid', patch_size=4):
        if mode not in ('rigid', 'patch','attachment','bending'):
            raise ValueError('Unknown coarse layout')
        self.plans = []
        self.mode=mode
        topology = metric_topology(scene)

        def add(ids, rigid, weight,bending=None):
            position=scene.rest_x[None,ids]
            mass=scene.mass[ids]
            center=(position*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
            basis=rigid_component_modes(position,center,scene.dof[ids])[0].double().cpu().numpy().reshape(-1,6)
            columns=list(range(bending.functions.shape[1])) if bending is not None else []
            if bending is None:
                for column in range(6 if rigid else 3):
                    if np.linalg.matrix_rank(basis[:,columns+[column]],tol=1e-9)>len(columns):columns.append(column)
            if not columns:return
            lookup = torch.full((scene.n,), -1, device=scene.device, dtype=torch.long)
            lookup[ids] = torch.arange(len(ids), device=scene.device)
            overlaps = []
            for index, incidence in enumerate(topology):
                mapped = lookup[incidence]
                rows = torch.where((mapped >= 0).any(-1))[0]
                if len(rows):
                    overlaps.append((index, rows, mapped[rows].clamp_min(0), mapped[rows] >= 0))
            self.plans.append(CoarsePlan(ids, rigid, weight, overlaps,torch.tensor(columns,device=scene.device),bending))

        for component in scene.interaction_entities:
            ids = component[scene.movable[component]]
            if not len(ids):
                continue
            bending=next((plan for plan in scene.bending_plans if torch.equal(plan.mobile,ids)),None) if mode=='bending' else None
            if bending is not None:
                add(ids,False,1.,bending);continue
            patches = mode == 'patch' and bool((~scene.body[ids]).all()) and len(ids) > patch_size
            clusters=[]
            if mode=='attachment':
                for body in torch.unique(scene.attachments[:,1]).tolist():
                    rows=(scene.attachments[:,1]==body)&(scene.attachment_stiffness>0)
                    if body in ids.tolist() and bool(rows.any()):
                        cluster=torch.unique(torch.cat((scene.attachments[rows,0],scene.attachments.new_tensor([body]))),sorted=True)
                        clusters.append(cluster[scene.movable[cluster]])
            add(ids, True, .5 if patches or clusters else 1.)
            for cluster in clusters:add(cluster,True,.5/len(clusters))
            if patches:
                points = scene.rest_x[ids].cpu().numpy()
                axis = int(np.ptp(points, axis=0).argmax())
                order = np.lexsort((points[:, (axis+2)%3], points[:, (axis+1)%3], points[:, axis]))
                ordered = ids[torch.as_tensor(order.copy(), device=scene.device)]
                for patch in torch.tensor_split(ordered, max(1, int(np.ceil(len(ids)/patch_size)))):
                    add(patch, False, .5)


class PreparedMixedCoarse:
    def __init__(self, scene, state, metric, layout, gains=None, composition='balanced'):
        if composition not in ('balanced','additive'):raise ValueError('Unknown coarse composition')
        self.metric = metric
        self.composition=composition
        self.blocks = []
        self.valid = torch.ones(state.x.shape[0], device=scene.device, dtype=torch.bool)
        for index,plan in enumerate(layout.plans):
            ids = plan.ids
            mass = scene.mass[ids]
            position = state.x[:, ids].detach()
            center = (position*mass[None, :, None]).sum(1, keepdim=True)/mass.sum()
            modes = plan.bending.modes(state).detach() if plan.bending is not None else rigid_component_modes(position, center, scene.dof[ids])[...,plan.columns]
            modes = modes.double()
            gram = metric.gram(plan, modes)
            scale = gram.diagonal(dim1=-2, dim2=-1).rsqrt()
            normalized = gram * scale[:, :, None] * scale[:, None, :]
            factor, info = torch.linalg.cholesky_ex(normalized, check_errors=False)
            self.valid = self.valid & (info == 0) & torch.isfinite(factor).flatten(1).all(-1)
            gain=None if gains is None else gains[index][...,plan.columns].double()
            self.blocks.append((plan, modes, scale, factor, gain))

    def __call__(self, value):
        result = torch.zeros_like(value)
        for plan, modes, scale, factor, gain in self.blocks:
            restricted = torch.einsum('bnip,bniq->bpq', modes, value[:, plan.ids].double())
            if gain is None:
                coefficient = torch.cholesky_solve(restricted*scale[..., None], factor)*scale[..., None]
            else:
                white=torch.linalg.solve_triangular(factor,restricted*scale[...,None],upper=False)
                coefficient=torch.linalg.solve_triangular(factor.transpose(-1,-2),white*gain[...,None],upper=True)*scale[...,None]
            response = torch.einsum('bnip,bpq->bniq', modes, coefficient).to(value.dtype)*plan.weight
            result = result.index_add(1, plan.ids, response)
        return result

    def residual_complement(self, residual):
        coarse = self(residual[..., None])
        return coarse, residual if self.composition=='additive' else residual-self.metric(coarse)[..., 0]

    def combine_negative_local(self, coarse, negative_local):
        local = negative_local[..., None]
        if self.composition=='additive':return (-coarse+local)[...,0]
        return (-coarse + local - self(self.metric(local)))[..., 0]
