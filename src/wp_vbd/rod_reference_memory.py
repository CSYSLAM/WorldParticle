"""Offline conversion of already accepted source matching/numeric history."""
import numpy as np
import torch

from . import rigid_contact_memory as memory


def accepted_cache(scene, state, fields, capacity):
    """Take post-call state for the NEXT call, never current/future labels."""
    result=memory.initial_history(scene,state,capacity=capacity)
    prefix='matching_after__pipeline0__'
    count=int(fields[prefix+'_prev_count'][0])
    if not 0<=count<=capacity:raise ValueError('Accepted source contact cache exceeds complete storage')
    if count!=int(fields['contacts_after__rigid_contact_count'][0]):raise ValueError('Matcher/source contact count differs')
    frame=fields[prefix+'_prev_frame']
    numeric_frame=fields['aux_after___prev_contact_frame']
    generation=int(fields['contacts_after__contact_generation'][0])
    if frame[0]<=0 or numeric_frame[0]<=0 or int(frame[1])!=generation or int(numeric_frame[1])!=generation:
        raise ValueError('Accepted geometry and numeric history refer to different source contact sets')
    for name,attribute in memory.cache_fields(scene).items():
        value=fields[prefix+attribute]
        if not np.array_equal(value,fields['matching_before__pipeline0__'+attribute]):
            raise ValueError('Unexpected geometry-matcher edit inside source dynamics: '+attribute)
        target=result['contact_match_'+name]
        if name=='frame':target[0]=torch.tensor([1,generation],dtype=target.dtype)
        elif name=='count':target[0,0]=count
        else:target[0,:count]=torch.as_tensor(value[:count],dtype=target.dtype)
    result['contact_query_generation'][0,0]=generation
    result['contact_dual'][0,:count]=torch.as_tensor(fields['aux_after___prev_contact_lambda'][:count]).double()
    if count:
        normal=torch.as_tensor(fields['aux_after___prev_contact_normal'][:count]).double()
        if bool((normal.norm(dim=-1)<=1e-12).any()):raise ValueError('Degenerate source accepted contact normal')
        result['contact_normal'][0,:count]=normal/normal.norm(dim=-1,keepdim=True)
    from .data import State
    memory.validate(scene,State(state.x,state.q,state.v,state.omega,result))
    return {name:value[0].numpy() for name,value in result.items()}
