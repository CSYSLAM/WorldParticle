"""Bounded joint-tree motion columns for the shared Cartesian response.

Each scalar joint moves its child subtree coherently. A maximum of six
columns per interaction is selected by frozen physical work/curvature, in
addition to the existing six rigid columns. No target poses are assigned,
no dynamics solver is called, and the existing twelve-column budget stays.
"""
from types import SimpleNamespace

import torch

from . import math3d as so3
from .protected_response import metric_action
from .coarse_response import rigid_component_modes


def topology(scene):
    cached = getattr(scene, '_articulation_coarse_topology', None)
    spec = scene.joint_spec
    signature = (tuple(map(tuple, spec.owners.tolist())), tuple(spec.enabled.tolist()))
    mimic = getattr(scene, 'mimic_spec', None)
    if mimic is not None:
        signature += (tuple(map(tuple, mimic.pairs.tolist())), tuple(map(tuple, mimic.coefficients.tolist())))
    if cached is not None and cached[0] == signature:
        return cached[1]
    if not bool(spec.enabled.all()):
        raise ValueError('Joint-tree columns require enabled tree joints; rebuild after ownership changes')
    linear = (spec.linear_axes.square().sum(-1) > .5).sum(-1)
    angular = (spec.angular_axes.square().sum(-1) > .5).sum(-1)
    scalar = (linear+angular) == 1
    fixed = (linear+angular == 0) & (spec.angular_projector.square().sum((-1,-2)) > .5)
    if not bool((scalar | fixed).all()) or bool((spec.angular_mode > 1).any()):
        raise ValueError('This bounded mode path admits scalar REVOLUTE/PRISMATIC and FIXED tree joints')
    parent_of = {}
    for parent, child in spec.owners.tolist():
        if child in parent_of:
            raise ValueError('Closed loops or multiple parent joints need a separate mode construction')
        parent_of[child] = parent
    for child in parent_of:
        seen = set(); node = child
        while node in parent_of:
            if node in seen: raise ValueError('Cyclic articulation is outside the joint-tree mode contract')
            seen.add(node); node = parent_of[node]
    source = torch.where(scalar)[0].tolist()
    if len(source)>64:raise ValueError('This prototype bounds tree preprocessing to 64 scalar joints')
    descendants = []
    for joint in source:
        root = int(spec.owners[joint, 1]); members = {root}
        for _ in range(len(parent_of)):
            members.update(child for child, parent in parent_of.items() if parent in members)
        descendants.append(torch.tensor(sorted(members), device=scene.device, dtype=torch.long))
    followers = {}
    if mimic is not None:
        for pair, coefficient in zip(mimic.pairs.tolist(), mimic.coefficients.tolist()):
            follower, reference = pair
            if follower not in source or reference not in source:
                raise ValueError('Mimic tree modes require scalar joint endpoints')
            followers[follower] = (reference, coefficient[1])
    independent = [joint for joint in source if joint not in followers]
    if len(independent)>24:raise ValueError('Articulation candidate pool exceeds its declared 24-column budget')
    lookup = {joint: i for i, joint in enumerate(independent)}
    memo = {}
    def coefficients(joint, seen):
        if joint in memo:return memo[joint]
        if joint in seen: raise ValueError('Cyclic mimic dependencies require a separate mode construction')
        value = torch.zeros(len(independent), device=scene.device, dtype=torch.float64)
        if joint in followers:
            reference, multiplier = followers[joint]
            if multiplier: value = multiplier*coefficients(reference, seen | {joint})
        else:value[lookup[joint]] = 1.
        memo[joint] = value
        return value
    mapping = torch.stack([coefficients(joint, set()) for joint in source]) if source else scene.rest_x.new_zeros(0,0)
    anchored={}
    for child in parent_of:
        node=child
        while node in parent_of:node=parent_of[node]
        anchored[child]=not bool(scene.movable[node])
    result = SimpleNamespace(source=source, descendants=descendants, linear=linear.bool(), mapping=mapping,
                             independent_joint_ids=independent,anchored=anchored)
    scene._articulation_coarse_topology = (signature, result)
    return result


def raw_modes(scene, state):
    plan = topology(scene); spec = scene.joint_spec
    raw = state.x.new_zeros(len(state.x), scene.n, 6, len(plan.independent_joint_ids))
    for column, (joint, ids) in enumerate(zip(plan.source, plan.descendants)):
        parent, child = spec.owners[joint]
        parent_frame = so3.normalize(so3.multiply(state.q[:, parent], spec.frames[joint,0,3:]))
        axis_local = spec.linear_axes[joint,0] if bool(plan.linear[joint]) else spec.angular_axes[joint,0]
        axis = so3.rotate(parent_frame, axis_local)
        if bool(plan.linear[joint]):
            translation=axis[:,None].expand(-1,len(ids),-1)
            angular=torch.zeros_like(translation)
        else:
            # Use the current child anchor: a finite pre-existing anchor
            # error is not increased by the root joint's proposed rotation.
            anchor = state.x[:,child]+so3.rotate(state.q[:,child], spec.frames[joint,1,:3]-scene.body_com[child])
            angular=axis[:,None].expand(-1,len(ids),-1)
            translation=torch.linalg.cross(angular, state.x[:,ids]-anchor[:,None])
        value=torch.cat((translation,angular),-1)[...,None]*plan.mapping[column]
        raw=raw.index_add(1,ids,value)
    if getattr(scene, 'mimic_spec', None) is not None:
        from .mimic_constraint import Tangent
        tangent = Tangent(scene, state)
        if not bool(tangent.valid.all()):raise ValueError('Invalid mimic tangent in articulation modes')
        # Project all columns together, reusing the same small mass Gram.
        for _, ids, jacobian, lift, inverse, _ in tangent.groups:
            coordinate = torch.einsum('brni,bnip->brp', jacobian, raw[:,ids])
            multiplier = torch.einsum('brs,bsp->brp', inverse, coordinate)
            raw = raw.index_add(1, ids, -torch.einsum('brni,brp->bnip', lift, multiplier))
    return raw*scene.dof[None,:,:,None]


def augment(scene, state, base, metric, residual, *, maximum_extra=6):
    if not 1<=maximum_extra<=6:raise ValueError('At most six extra tree columns fit the declared twelve-column budget')
    with torch.no_grad():
        modes = raw_modes(scene, state)
        if not modes.shape[-1]:return base
        image = metric_action(metric, modes)
        blocks = []; selected = []
        for ports, rigid, *tail in base.blocks:
            ids = ports.ids
            base_space='interaction_rigid';material_id=None
            material_mask=scene.kind[ids]<2;rigid_ids=ids[scene.kind[ids]==2]
            if scene.metadata.get('articulation_material_motion_modes') and bool(material_mask.any()) and len(rigid_ids):
                groups=torch.unique(scene.object_id[ids[material_mask]])
                plan=topology(scene)
                if len(groups)!=1 or any(not plan.anchored.get(i,False) for i in rigid_ids.tolist()):
                    raise ValueError('Material/robot motion allocation currently requires one material object and anchored tree bodies per interaction')
                local=torch.where(material_mask)[0];material=ids[local]
                mass=scene.mass[material]
                center=(state.x[:,material]*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
                value=rigid_component_modes(state.x[:,material],center,scene.dof[material]).to(rigid.dtype)
                rigid=torch.zeros_like(rigid).index_copy(1,local,value)
                base_space='material_rigid';material_id=int(groups[0])
            gradient = torch.einsum('bnip,bni->bp', modes[:,ids], residual[:,ids])
            curvature = (modes[:,ids]*image[:,ids]).sum((1,2))
            valid = torch.isfinite(gradient) & torch.isfinite(curvature) & (curvature>0)
            score = torch.where(valid, gradient.square()/curvature.clamp_min(1e-300), 0.).amax(0)
            order = torch.argsort(score, descending=True, stable=True)[:maximum_extra]
            order = order[score[order]>0]
            addition = modes[:,ids][:,:,:,order]
            blocks.append((ports, torch.cat((rigid, addition),-1), *tail))
            selected.append({'joint_ids':[topology(scene).independent_joint_ids[i] for i in order.tolist()],
                             'estimated_physical_work':score[order].tolist(),
                             'base_space':base_space,'material_object_id':material_id})
        scene.articulation_coarse_diagnostic = {'maximum_extra_columns':maximum_extra,
                                               'maximum_total_columns':12,'maximum_candidate_columns':24,
                                               'maximum_scalar_tree_joints':64,'selected':selected}
        return SimpleNamespace(blocks=blocks)
