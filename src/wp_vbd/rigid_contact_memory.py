"""Accepted matched-contact geometry and tangent numerical history.

The pinned collision matcher is a geometry operator. Restore its complete
accepted cache before every query, including retries and restarts; only an
accepted physical step publishes the proposed cache into State.history.
"""
import math

import torch

BASE_CACHE = {'keys': '_prev_sorted_keys', 'positions': '_prev_pos_world', 'normals': '_prev_normal',
              'count': '_prev_count', 'frame': '_prev_frame'}
STICKY_CACHE = {'point0': '_prev_point0', 'point1': '_prev_point1',
                'offset0': '_prev_offset0', 'offset1': '_prev_offset1'}


def authored_policy(example):
    contacts = example.contacts
    mode = contacts.contact_matching_mode
    matcher = example.collision_pipeline._contact_matcher
    if mode not in ('latest', 'sticky') or matcher is None:
        raise ValueError('Authored matched contacts require latest/sticky geometry matching')
    return validate_policy({'mode': mode, 'position_threshold': math.sqrt(matcher._pos_threshold_sq),
                            'normal_dot_threshold': matcher._normal_dot_threshold})


def validate_policy(policy):
    if not isinstance(policy, dict) or set(policy) != {'mode', 'position_threshold', 'normal_dot_threshold'}:
        raise ValueError('Complete explicit contact matching policy required')
    if policy['mode'] not in ('latest', 'sticky'):
        raise ValueError('Only latest/sticky matched contact policies are implemented')
    for name in ('position_threshold', 'normal_dot_threshold'):
        value = policy[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError('Finite contact matching policy required: '+name)
    if policy['position_threshold'] < 0 or not -1 <= policy['normal_dot_threshold'] <= 1:
        raise ValueError('Invalid contact matching thresholds')
    return dict(policy)


def enabled(scene):
    return bool(scene.metadata.get('cross_step_contact_history', False))


def cache_fields(scene):
    return {**BASE_CACHE, **(STICKY_CACHE if scene.metadata['contact_matching_policy']['mode'] == 'sticky' else {})}


def initial_history(scene, state, *, capacity=None):
    count = scene.metadata.get('initial_contact_history_storage_capacity',scene.metadata['contact_history_capacity']) if capacity is None else capacity
    if type(count) is not int or not 1<=count<=scene.metadata['contact_history_capacity'] or len(state.x) != 1:
        raise ValueError('Matched rigid history requires declared capacity and batch1')
    result = {'contact_match_keys': torch.full((1, count), torch.iinfo(torch.int64).max, device=scene.device, dtype=torch.int64),
              'contact_match_count': torch.zeros(1, 1, device=scene.device, dtype=torch.int32),
              'contact_match_frame': torch.zeros(1, 2, device=scene.device, dtype=torch.int32),
              'contact_query_generation': torch.zeros(1, 1, device=scene.device, dtype=torch.int32),
              'contact_dual': state.x.new_zeros(1, count, 3), 'contact_normal': state.x.new_zeros(1, count, 3)}
    for name in cache_fields(scene):
        if name in ('keys', 'count', 'frame'):
            continue
        result['contact_match_'+name] = torch.zeros(1, count, 3, device=scene.device, dtype=torch.float32)
    return result


def history_keys(scene):
    return {'contact_match_'+name for name in cache_fields(scene)} | {'contact_query_generation', 'contact_dual', 'contact_normal'}


def reset(scene, state):
    """Explicitly invalidate the matching/numeric cache, retaining rod state.

    Episode teleports still need the full rod/native reset contract. This
    operation only cold-starts contact history at the next fresh query.
    """
    validate(scene,state)
    from .data import State
    history={**state.history,**initial_history(scene,state)}
    return State(state.x,state.q,state.v,state.omega,history)


def validate(scene, state):
    history = state.history
    if not isinstance(history, dict) or not history_keys(scene) <= set(history):
        raise ValueError('Complete accepted matched-contact history required')
    keys=history['contact_match_keys']
    if keys.ndim!=2 or keys.shape[0]!=1:raise ValueError('One complete matched cache sequence required')
    expected = initial_history(scene,state,capacity=keys.shape[1])
    for name, value in expected.items():
        incoming = history[name]
        if incoming.shape != value.shape or incoming.dtype != value.dtype or incoming.device != value.device or not bool(torch.isfinite(incoming).all()):
            raise ValueError('Invalid matched-contact history: '+name)
    count = int(history['contact_match_count'][0, 0])
    capacity = keys.shape[1]
    frame = history['contact_match_frame'][0]
    generation = int(history['contact_query_generation'][0, 0])
    if not 0 <= count <= capacity or int(frame[0]) not in (0, 1) or generation < 0:
        raise ValueError('Invalid matched-contact count/frame')
    if (count and int(frame[0]) != 1) or int(frame[1]) != generation:
        raise ValueError('Matched-contact cache is from another accepted generation')
    keys = history['contact_match_keys'][0, :count]
    if bool((keys < 0).any()) or bool((keys[1:] < keys[:-1]).any()):
        raise ValueError('Accepted contact cache keys must retain the sorted stream')
    if count and not bool(((history['contact_normal'][0, :count].square().sum(-1)-1).abs() < 1e-8).all()):
        raise ValueError('Accepted physical contact normals must be normalized')


def compact(scene,state):
    """Keep every accepted row, removing only the unused capacity suffix."""
    validate(scene,state)
    from .data import State
    count=max(1,int(state.history['contact_match_count'][0,0]))
    per_row=history_keys(scene)-{'contact_match_count','contact_match_frame','contact_query_generation'}
    history={name:(value[:,:count].clone() if name in per_row else value) for name,value in state.history.items()}
    return State(state.x,state.q,state.v,state.omega,history)


def restore_query(oracle, state):
    scene = oracle.scene
    validate(scene, state)
    matcher = oracle.pipeline._contact_matcher
    if matcher is None:
        raise ValueError('Matched-contact history needs its geometry matcher')
    for name, attribute in cache_fields(scene).items():
        value = state.history['contact_match_'+name][0]
        if name == 'frame':
            value = value.clone()
            # History carries one logical stream; native weak-object buffer
            # identities are rebound explicitly when constructing a restart.
            if int(value[0]) == 1:
                value[0] = matcher.buffer_id(oracle.contacts)
        destination=oracle.wp.to_torch(getattr(matcher, attribute))
        if name not in ('count','frame') and len(value)>len(destination):
            count=int(state.history['contact_match_count'][0,0])
            if not scene.metadata.get('bounded_analytic_contact_buffers') or count>len(destination):
                raise ValueError('Native query capacity cannot restore all accepted contact rows')
            # Only unused input padding is removed. Every accepted row and
            # its numeric multiplier remain available in the physical state.
            value=value[:count]
        if destination.shape!=value.shape:
            if name in ('count','frame'):raise ValueError('Invalid contact cache scalar shape')
            destination.fill_(torch.iinfo(torch.int64).max if name=='keys' else 0)
            destination[:len(value)].copy_(value)
        else:destination.copy_(value)
    matcher._prev_claim.fill_(0x7FFFFFFFFFFFFFFF)
    matcher._reset_world_mask.zero_()
    oracle.wp.to_torch(oracle.contacts.contact_generation).copy_(state.history['contact_query_generation'][0])


def capture_query(oracle):
    scene, wp = oracle.scene, oracle.wp
    matcher = oracle.pipeline._contact_matcher
    prefix=None
    if scene.metadata.get('bounded_analytic_contact_buffers'):
        prefix=max(1,int(wp.to_torch(matcher._prev_count)[0]))
    result={}
    for name,attribute in cache_fields(scene).items():
        value=wp.to_torch(getattr(matcher,attribute))
        if prefix is not None and name not in ('count','frame'):value=value[:prefix]
        result['contact_match_'+name]=value.clone()[None]
    frame = result['contact_match_frame']
    if int(frame[0, 0]) != matcher.buffer_id(oracle.contacts):
        raise ValueError('Geometry matcher saved another contact stream')
    frame[0, 0] = 1
    result['contact_query_generation'] = wp.to_torch(oracle.contacts.contact_generation).clone()[None]
    return result


def tangent_warm_start(scene, state, contacts, rho_n):
    """Compressive, cone-clamped accepted tangent transport and retention.

    The existing finite normal constitutive fixed point stays eliminated.
    This history stores that accepted normal load to gate tangent retention;
    it does not add another normal-force term to the physical residual.
    Sticky anchors carry tangent memory in geometry, so their dual starts0.
    """
    validate(scene, state)
    policy = scene.metadata['contact_matching_policy']
    if contacts.matching_mode != policy['mode'] or contacts.match_index is None:
        raise ValueError('Collision packet has no matching provenance')
    indices = contacts.match_index
    old_count = int(state.history['contact_match_count'][0, 0])
    frame = state.history['contact_match_frame'][0]
    provenance = (int(frame[0]) == 1 and contacts.match_generation == int(frame[1])
                  and contacts.stream_id==id(scene.geometry_oracle.contacts))
    if not provenance:
        indices=torch.full_like(indices,-1)
    if bool((indices >= old_count).any()):
        raise ValueError('Matched index exceeds the accepted contact set')
    matched = (indices >= 0) & provenance
    slots = indices.clamp_min(0)
    old = state.history['contact_dual'][:, slots]
    old_normal = state.history['contact_normal'][:, slots]
    normal_load = (old*old_normal).sum(-1)
    tangent = old-normal_load[..., None]*old_normal
    tangent = tangent-(tangent*contacts.normal).sum(-1, keepdim=True)*contacts.normal
    from .rigid_contact_rows import project_disk
    tangent = project_disk(tangent, contacts.mu[None]*normal_load.clamp_min(0))
    live = matched[None] & (normal_load > 0) & (contacts.material[None] > 0) & (rho_n > 0)
    if policy['mode'] == 'sticky':
        live = torch.zeros_like(live)
    return torch.where(live[..., None], tangent*scene.metadata['joint_gamma'], 0.)


def commit(scene, context, output):
    contacts = context.contacts
    capacity = max(1,contacts.source_count)
    if contacts.generation != int(context.contact_cache['contact_query_generation'][0, 0]):
        raise ValueError('Contact proposal/cache generation differs at commit')
    _, _, _, _, depth = contacts.geometry(scene, output)
    alpha = scene.metadata['contact_alpha']
    effective = torch.where(depth < 0, depth, depth-alpha*context.depth0)
    normal_load = contacts.material[None]*effective.clamp_min(0)
    dual = context.dual_t+normal_load[..., None]*contacts.normal
    result={name:(value[:,:capacity] if name not in ('contact_match_count','contact_match_frame','contact_query_generation') else value)
            for name,value in context.contact_cache.items()}
    result['contact_dual'] = output.x.new_zeros(1, capacity, 3).index_copy(1, contacts.source_slots, dual)
    result['contact_normal'] = output.x.new_zeros(1, capacity, 3).index_copy(1, contacts.source_slots, contacts.normal[None])
    return result
