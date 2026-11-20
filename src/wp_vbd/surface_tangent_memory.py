"""Accepted surface-contact tangent memory and a bounded learned force seed."""

import math
import torch

from .rigid_contact_rows import project_disk


HISTORY_KEYS={
    'surface_contact_shape','surface_contact_corners','surface_contact_weights',
    'surface_contact_local_anchor','surface_contact_normal','surface_contact_tangent',
}

ANCHOR_QUANTIZATION_M=0.005
ANCHOR_BINS=128
ANCHOR_OFFSET_M=0.32


def initial_history(state,capacity):
    device=state.x.device
    return {
        'surface_contact_shape':torch.full((1,capacity),-1,device=device,dtype=torch.long),
        'surface_contact_corners':torch.full((1,capacity,3),-1,device=device,dtype=torch.long),
        'surface_contact_weights':state.x.new_zeros(1,capacity,3),
        'surface_contact_local_anchor':state.x.new_zeros(1,capacity,3),
        'surface_contact_normal':state.x.new_zeros(1,capacity,3),
        'surface_contact_tangent':state.x.new_zeros(1,capacity,3),
    }


def validate(history,capacity):
    template={
        'surface_contact_shape':((1,capacity),torch.long),
        'surface_contact_corners':((1,capacity,3),torch.long),
    }
    for name,(shape,dtype) in template.items():
        value=history[name]
        if value.shape!=shape or value.dtype!=dtype:raise ValueError('Invalid surface contact cache: '+name)
    for name in ('surface_contact_weights','surface_contact_local_anchor',
                 'surface_contact_normal','surface_contact_tangent'):
        value=history[name]
        if value.shape!=(1,capacity,3) or not value.is_floating_point() or not bool(torch.isfinite(value).all()):
            raise ValueError('Invalid surface contact cache: '+name)
    count=(history['surface_contact_shape']>=0).sum()
    if int(count)>capacity:raise ValueError('Surface contact cache exceeds its fixed capacity')


def match_force(snapshot,history,load,mu):
    """One-to-one feature matching, current-plane projection and cone clamp."""
    rows=len(snapshot.shape);capacity=history['surface_contact_shape'].shape[1]
    if rows>capacity:raise ValueError('Surface contact history capacity overflow')
    zero=load.new_zeros(1,rows,3)
    if not rows or not capacity:return zero
    # Include the rigid local contact anchor in the sorted identity. Native
    # collision can emit several points for one (shape, triangle); those rows
    # have different moment arms and may not share each other's tangent force.
    base=snapshot.ports.node_count+1
    anchor=snapshot.ports.local_points[:,0].to(load)
    old_anchor=history['surface_contact_local_anchor'][0].to(load)
    anchor_code=torch.round((anchor+ANCHOR_OFFSET_M)/ANCHOR_QUANTIZATION_M).long()
    old_anchor_code=torch.round((old_anchor+ANCHOR_OFFSET_M)/ANCHOR_QUANTIZATION_M).long()
    anchor_code=anchor_code.clamp(0,ANCHOR_BINS-1)
    old_anchor_code=old_anchor_code.clamp(0,ANCHOR_BINS-1)
    def key(shape,corners):
        value=shape.to(torch.int64)
        for corner in corners.unbind(-1):value=value*base+(corner.to(torch.int64)+1)
        return value
    def feature_key(shape,corners,anchor_code):
        value=key(shape,corners)
        for coordinate in anchor_code.unbind(-1):value=value*ANCHOR_BINS+coordinate
        return value
    old_shape=history['surface_contact_shape'][0]
    old_live=old_shape>=0
    old_slots=torch.where(old_live)[0]
    if not len(old_slots):return zero
    old_key=feature_key(old_shape[old_slots],history['surface_contact_corners'][0,old_slots],old_anchor_code[old_slots])
    sorted_key,order=old_key.sort(stable=True)
    sorted_slots=old_slots[order]
    # Query the 27 neighboring quantization cells, avoiding a dense pairwise
    # contact matrix while remaining robust to millimeter anchor jitter.
    offsets=torch.tensor([(x,y,z) for x in (-1,0,1) for y in (-1,0,1) for z in (-1,0,1)],
                         device=load.device,dtype=torch.long)
    codes=anchor_code[:,None,:]+offsets[None]
    cell_valid=((codes>=0)&(codes<ANCHOR_BINS)).all(-1)
    query_key=feature_key(snapshot.shape[:,None].expand(-1,len(offsets)),
        snapshot.corners[:,None,:].expand(-1,len(offsets),-1),codes)
    flat_key=query_key.reshape(-1)
    offset=torch.searchsorted(sorted_key,flat_key)
    safe=offset.clamp_max(len(sorted_key)-1)
    matched=(sorted_key[safe]==flat_key)&cell_valid.reshape(-1)
    candidate_slots=sorted_slots[safe].reshape(rows,-1)
    matched=matched.reshape(rows,-1)
    bary=snapshot.ports.weights[:,1,1:].to(load)
    normal=snapshot.normal.to(load)
    old_bary=history['surface_contact_weights'][0,candidate_slots].to(load)
    old_normal=history['surface_contact_normal'][0,candidate_slots].to(load)
    distance=(bary[:,None]-old_bary).square().sum(-1)
    anchor_distance=(anchor[:,None]-old_anchor[candidate_slots]).norm(dim=-1)
    alignment=(normal[:,None]*old_normal).sum(-1)
    eligible=matched & (distance<.05) & (anchor_distance<.008) & (alignment>.85)
    # If the native query reports duplicate feature keys, permit only the
    # closest current row to reuse the accepted multiplier.
    cost=torch.where(eligible,distance+anchor_distance.square()/(ANCHOR_QUANTIZATION_M**2)
                     +.1*(1-alignment),torch.full_like(distance,float('inf')))
    chosen=cost.argmin(-1)
    chosen_cost=cost.gather(1,chosen[:,None])[:,0]
    best=candidate_slots.gather(1,chosen[:,None])[:,0]
    live=eligible.any(-1)
    per_old=torch.full((capacity,),float('inf'),device=load.device,dtype=load.dtype)
    per_old.scatter_reduce_(0,best,chosen_cost,reduce='amin',include_self=True)
    unique=live & (chosen_cost<=per_old[best])
    row_ids=torch.arange(rows,device=load.device)
    tie=torch.full((capacity,),rows,device=load.device,dtype=torch.long)
    tie.scatter_reduce_(0,best,torch.where(unique,row_ids,rows),reduce='amin',include_self=True)
    unique &= tie[best]==row_ids
    old=history['surface_contact_tangent'][0,best].to(load)
    old=old-(old*normal).sum(-1,keepdim=True)*normal
    carried=project_disk(old,mu*load[0].clamp_min(0))
    return torch.where(unique[None,:,None],carried[None],zero)


def commit(snapshot,tangent,accepted,history):
    result={key:value.clone() for key,value in history.items()}
    if not bool(accepted.all()):return result
    capacity=result['surface_contact_shape'].shape[1]
    rows=len(snapshot.shape)
    if rows>capacity:raise ValueError('Surface contact history capacity overflow')
    shape=torch.full_like(result['surface_contact_shape'],-1)
    corners=torch.full_like(result['surface_contact_corners'],-1)
    weights=torch.zeros_like(result['surface_contact_weights'])
    anchor=torch.zeros_like(result['surface_contact_local_anchor'])
    normal=torch.zeros_like(result['surface_contact_normal'])
    force=torch.zeros_like(result['surface_contact_tangent'])
    if rows:
        shape[0,:rows]=snapshot.shape
        corners[0,:rows]=snapshot.corners
        weights[0,:rows]=snapshot.ports.weights[:,1,1:].to(weights)
        anchor[0,:rows]=snapshot.ports.local_points[:,0].to(anchor)
        normal[0,:rows]=snapshot.normal.to(normal)
        value=tangent.detach()[0]
        value=value-(value*normal[0,:rows]).sum(-1,keepdim=True)*normal[0,:rows]
        force[0,:rows]=value
    result.update(surface_contact_shape=shape,surface_contact_corners=corners,
                  surface_contact_weights=weights,surface_contact_local_anchor=anchor,surface_contact_normal=normal,
                  surface_contact_tangent=force)
    return result


def seed_learned(model,scene,start,context,learned):
    head=getattr(model,'contact_tangent_head',None)
    if head is None or not learned or not len(context.contacts.soft.shape):return
    snapshot=context.contacts.soft
    point,jac,depth=snapshot.geometry(start,scene.body_com)
    velocity=torch.cat((start.v,start.omega),-1)[:,snapshot.ports.ids]
    endpoint_velocity=torch.einsum('brpsci,brsi->brpc',jac,velocity)
    relative_velocity=endpoint_velocity[:,:,1]-endpoint_velocity[:,:,0]
    relative_velocity=relative_velocity+context.soft.surface_velocity
    n=snapshot.normal[None].to(start.x)
    tangent_velocity=relative_velocity-(relative_velocity*n).sum(-1,keepdim=True)*n
    axis=torch.nn.functional.one_hot(n[0].abs().argmin(-1),3).to(n)
    t1=torch.nn.functional.normalize(torch.linalg.cross(n,axis[None]),dim=-1,eps=1e-12)
    t2=torch.linalg.cross(n,t1)
    frame=torch.stack((t1,t2),-1)
    gravity=scene.values.get('gravity',start.x.new_zeros(3)).to(start.x)
    gravity=gravity[None,None].expand_as(tangent_velocity)
    tangent_gravity=gravity-(gravity*n).sum(-1,keepdim=True)*n
    dt=context.dt;length=scene.length
    mass=(scene.mass[snapshot.ports.ids][:,None]*snapshot.ports.weights).sum((-1,-2)).clamp_min(1e-30)
    load=snapshot.shape_margin.new_zeros(1,len(snapshot.shape))
    load=(context.soft.material.stiffness[None]*depth.clamp_min(0))
    velocity_feature=torch.einsum('brc,brck->brk',tangent_velocity,frame)*dt/length
    gravity_feature=torch.einsum('brc,brck->brk',tangent_gravity,frame)*dt*dt/length
    support=torch.log1p(load*dt*dt/mass[None])
    dt_feature=load.new_full(load.shape,math.log(dt*60))
    features=torch.cat((velocity_feature,gravity_feature,support[...,None],
        context.soft.material.friction[None,:,None],dt_feature[...,None]),-1)
    fraction=torch.tanh(head(features.to(head[0].weight.dtype))).to(start.x.dtype)
    seed_scale=float(model.config.get('contact_tangent_seed_scale',.1))
    if not math.isfinite(seed_scale) or not 0<=seed_scale<=.25:
        raise ValueError('Learned tangent seed scale must be in [0, .25]')
    predicted=seed_scale*torch.einsum('brck,brk->brc',frame,fraction)*(
        context.soft.material.friction[None,:,None]*load[...,None])
    carried=context.soft.dual_t
    seeded=project_disk(carried+predicted,context.soft.material.friction[None]*load)
    context.soft.dual_t=seeded
    context.dual_t=torch.cat((context.rigid.dual_t,seeded),1)
    context.learned_tangent_seed=predicted
