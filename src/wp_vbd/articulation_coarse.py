"""Bounded joint-tree motion columns for the shared Cartesian response.

Each scalar joint moves its child subtree coherently. A maximum of six
columns per interaction is selected by frozen physical work/curvature, in
addition to the existing six rigid columns. No target poses are assigned,
no dynamics solver is called. The default budget is twelve columns. The
explicit independent-object path admits two free objects plus tree modes
within eighteen columns, without changing the learned architecture.
That path can request eight tree columns when space permits; the combined
eighteen-column budget still limits the selection for each interaction.
"""
from types import SimpleNamespace

import torch

from . import math3d as so3
from .protected_response import metric_action
from .coarse_response import rigid_component_modes
from .static_topology import tensor_versions


def topology(scene):
    cached = getattr(scene, '_articulation_coarse_topology', None)
    spec = scene.joint_spec
    mimic = getattr(scene, 'mimic_spec', None)
    sources=(spec.owners,spec.enabled,spec.linear_axes,spec.angular_axes,
             spec.angular_projector,spec.angular_mode,scene.movable)
    if mimic is not None:
        sources += (mimic.pairs,mimic.coefficients)
    signature=tensor_versions(sources)
    if signature is not None and cached is not None and cached[0] == signature:
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
                             independent_joint_ids=independent,anchored=anchored,
                             parent_children=tuple(tuple(int(k) for k in spec.owners[joint].tolist()) for joint in source),
                             linear_modes=tuple(bool(linear[joint]) for joint in source),source_tensors=sources)
    scene._articulation_coarse_topology = (signature, result)
    return result


def raw_modes(scene, state):
    plan = topology(scene); spec = scene.joint_spec
    raw = state.x.new_zeros(len(state.x), scene.n, 6, len(plan.independent_joint_ids))
    for column, (joint, ids) in enumerate(zip(plan.source, plan.descendants)):
        parent, child = plan.parent_children[column]
        parent_frame = so3.normalize(so3.multiply(state.q[:, parent], spec.frames[joint,0,3:]))
        axis_local = spec.linear_axes[joint,0] if plan.linear_modes[column] else spec.angular_axes[joint,0]
        axis = so3.rotate(parent_frame, axis_local)
        if plan.linear_modes[column]:
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


def augment(scene, state, base, metric, residual, *, maximum_extra=None):
    independent=bool(scene.metadata.get('independent_object_motion_modes'))
    if maximum_extra is None:maximum_extra=scene.metadata.get('joint_tree_extra_columns',6)
    limit=8 if independent else 6
    if not isinstance(maximum_extra,int) or not 1<=maximum_extra<=limit:
        raise ValueError(f'Joint-tree extra-column budget must be an integer in 1..{limit}')
    with torch.no_grad():
        modes = raw_modes(scene, state)
        if not modes.shape[-1] and not scene.metadata.get('elastic_shape_response',False):return base
        image = metric_action(metric, modes)
        blocks = []; selected = []
        for ports, rigid, *tail in base.blocks:
            ids = ports.ids
            base_space='interaction_rigid';material_id=None
            material_mask=scene.kind[ids]<2;rigid_ids=ids[scene.kind[ids]==2]
            if scene.metadata.get('independent_object_motion_modes'):
                # A free payload must translate independently of both the
                # anchored arm and the cloth. Allocate physical modes by
                # authored object, retaining the same learned response.
                plan=topology(scene)
                if any(i in plan.anchored and not plan.anchored[i] for i in rigid_ids.tolist()):
                    raise ValueError('Floating articulated trees need root-coherent object modes')
                free_mask=material_mask.clone()
                free_mask[scene.kind[ids]==2]=torch.tensor(
                    [i not in plan.anchored for i in rigid_ids.tolist()],device=ids.device,dtype=torch.bool)
                groups=torch.unique(scene.object_id[ids[free_mask]])
                if len(groups)>2:
                    raise ValueError('Independent motion allocation admits at most two free objects per interaction')
                object_modes=[]
                for group in groups:
                    local=torch.where(free_mask & (scene.object_id[ids]==group))[0]
                    entity=ids[local];mass=scene.mass[entity]
                    center=(state.x[:,entity]*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
                    value=rigid_component_modes(state.x[:,entity],center,scene.dof[entity]).to(rigid.dtype)
                    object_modes.append(rigid.new_zeros(*rigid.shape[:-1],6).index_copy(1,local,value))
                if object_modes:rigid=torch.cat(object_modes,-1)
                base_space='independent_objects' if object_modes else 'interaction_rigid'
                material_id=groups.tolist()
            elif scene.metadata.get('articulation_material_motion_modes') and bool(material_mask.any()) and len(rigid_ids):
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
            shape_columns=modes.new_zeros(len(state.x),scene.n,6,0)
            if scene.metadata.get('elastic_shape_response',False):
                from .elastic_shape_modes import response_columns
                shape_columns=response_columns(scene,state,metric,residual,ids)
            gradient = torch.einsum('bnip,bni->bp', modes[:,ids], residual[:,ids])
            curvature = (modes[:,ids]*image[:,ids]).sum((1,2))
            valid = torch.isfinite(gradient) & torch.isfinite(curvature) & (curvature>0)
            score = torch.where(valid, gradient.square()/curvature.clamp_min(1e-300), 0.).amax(0)
            # Two 7-axis arms with mimic grippers each need eight tree
            # coordinates. Keep every block inside the existing embedding;
            # two free objects still leave room for only six tree columns.
            extra=min(maximum_extra,(18 if independent else 12)-rigid.shape[-1]-shape_columns.shape[-1])
            if extra<0:raise ValueError('Elastic material columns exceed the declared coarse budget')
            order = torch.argsort(score, descending=True, stable=True)[:extra]
            order = order[score[order]>0]
            addition = modes[:,ids][:,:,:,order]
            blocks.append((ports, torch.cat((rigid, addition,shape_columns[:,ids]),-1), *tail))
            selected.append({'joint_ids':[topology(scene).independent_joint_ids[i] for i in order.tolist()],
                             'estimated_physical_work':score[order].tolist(),
                             'base_space':base_space,'material_object_id':material_id,
                             'elastic_shape_columns':shape_columns.shape[-1]})
        scene.articulation_coarse_diagnostic = {'maximum_extra_columns':maximum_extra,
                                               'maximum_total_columns':18 if scene.metadata.get('independent_object_motion_modes') else 12,'maximum_candidate_columns':24,
                                               'maximum_scalar_tree_joints':64,'selected':selected}
        return SimpleNamespace(blocks=blocks)
