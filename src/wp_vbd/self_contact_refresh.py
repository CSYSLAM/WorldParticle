"""Current self witnesses on fixed complete primitive-pair identities.

Forces use current closest geometry. The positive metric freezes its ports;
merit uses true closest normal energy and lagged damping/friction geometry.
No broad phase, dynamics solver, candidate mutation or history commit occurs.
"""
import torch

from .contact_ports import PointPorts
from .self_contact_geometry import vertex_face,edge_edge
from .self_contact_rows import SelfPointContacts,begin,normal_potential,friction_potential


def enabled(configuration,contract):
    """Allow new composed material rows without migrating legacy self demos."""
    key='elastic_rod_refresh_self_geometry'
    if contract=='elastic_rod_surface_joint_self_contact_v222' and key in configuration:
        value=configuration[key]
    else:value=configuration.get('self_contact_refresh_geometry',False)
    if type(value) is not bool:raise ValueError('Explicit Boolean self-contact geometry policy required')
    return value


def fields(snapshot,state):
    if len(state.x)!=1:raise ValueError('Refreshed self witnesses require the explicit batch1 contact contract')
    count=len(snapshot.kind)
    weights=state.x.new_zeros(count,2,4);normal=state.x.new_zeros(count,3);distance=state.x.new_zeros(count)
    for kind,compute in ((9,vertex_face),(10,edge_edge)):
        ids=torch.where(snapshot.kind==kind)[0]
        geometry=compute(state.x[0,snapshot.ports.ids[ids]])
        if getattr(snapshot,'allow_finite_overlap',False):
            geometry.normal=torch.where((geometry.distance>1e-12)[:,None],geometry.normal,snapshot.normal[ids])
        weights=weights.index_copy(0,ids,geometry.weights)
        normal=normal.index_copy(0,ids,geometry.normal)
        distance=distance.index_copy(0,ids,geometry.distance)
    return weights,normal,distance[None]


def distances(snapshot,state):
    """Closest distances for merit; force ports and path certificates stay full."""
    if len(state.x)!=1:raise ValueError('Refreshed self witnesses require the explicit batch1 contact contract')
    distance=state.x.new_zeros(len(snapshot.kind))
    for kind,compute in ((9,vertex_face),(10,edge_edge)):
        ids=torch.where(snapshot.kind==kind)[0]
        values=compute(state.x[0,snapshot.ports.ids[ids]],distance_only=True)
        distance=distance.index_copy(0,ids,values)
    return distance[None]


def snapshots(snapshot,state):
    weights,normal,_=fields(snapshot,state)
    old=snapshot.ports
    def ports(differentiable):
        return PointPorts(old.owners,weights,old.rigid,old.local_points,old.body_mask,
                           differentiable_weights=differentiable)
    current=SelfPointContacts(ports(True),normal,snapshot.margin,snapshot.kind)
    frozen=SelfPointContacts(ports(False),normal.detach(),snapshot.margin,snapshot.kind)
    current.allow_finite_overlap=frozen.allow_finite_overlap=getattr(snapshot,'allow_finite_overlap',False)
    current.metric_ports=frozen.ports
    return current,frozen


def contexts(snapshot,state,body_com,context,material_start):
    current,frozen=snapshots(snapshot,state)
    def make(contacts):
        result=begin(contacts,state,body_com,context.dt,context.stiffness,context.damping,context.friction,
                     context.epsilon/context.dt,material_start=material_start)
        result.compact_frozen_contact_metric=getattr(context,'compact_frozen_contact_metric',False)
        result.contact_row_checkpointing=getattr(context,'contact_row_checkpointing',False)
        if hasattr(context,'allowed_overlap'):result.allowed_overlap=context.allowed_overlap
        return result
    return current,frozen,make(current),make(frozen)


def merit(snapshot,state,body_com,context,frozen):
    contacts=frozen['frozen_geometry'];lagged=frozen['merit_context']
    point=contacts.geometry(state,body_com,jacobian=False)[0]
    displacement=point[:,:,1]-lagged.start_point[:,:,1]-point[:,:,0]+lagged.start_point[:,:,0]
    normal=contacts.normal;dn=(displacement*normal).sum(-1)
    tangent=displacement-dn[...,None]*normal
    normal_energy=normal_potential(distances(snapshot,state),snapshot.margin,context.stiffness)[0]
    damping=.5*frozen['damping_active']/context.dt*dn.clamp_max(0).square()
    friction=context.friction*frozen['normal_load']*friction_potential(tangent,state.x.new_tensor(context.epsilon))
    return normal_energy+damping+friction
