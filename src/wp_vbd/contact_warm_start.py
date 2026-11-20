"""Remove closing free-flight components at imminent static surface contacts.

Only the initial candidate changes. Inertial targets, constitutive forces,
response budgets and all nonlinear acceptance checks remain unchanged.
"""
import torch
from . import math3d as so3


def project_closing(warm,ids,normals):
    """Frozen normal-span projection on particles with inward warm motion."""
    if not len(ids):return warm,0
    with torch.no_grad():
        inward=(warm[0,ids,:3].detach()*normals).sum(-1)<0
        ids=ids[inward];normals=normals[inward]
        if not len(ids):return warm,0
        nodes,mapping=torch.unique(ids,sorted=True,return_inverse=True)
        gram=warm.new_zeros(len(nodes),3,3).index_add(0,mapping,normals[...,None]*normals[:,None,:])
        values,frame=so3.eigh_zero_safe(gram)
        live=values>values.amax(-1,keepdim=True)*1e-12
        projector=(frame*live[:,None,:])@frame.transpose(-1,-2)
    translation=warm[:,nodes,:3]
    projected=translation-(projector[None]@translation[...,None])[...,0]
    value=torch.cat((projected,warm[:,nodes,3:]),-1)
    return warm.index_copy(1,nodes,value),len(nodes)


def prepare(scene,current,warm,context,frozen):
    contacts=context.contacts
    if not hasattr(contacts,'soft') or 'soft_rows' not in frozen:
        return warm,{'enabled':True,'projected_particles':0}
    snapshot=contacts.soft;ports=snapshot.ports;rows=frozen['soft_rows']
    with torch.no_grad():
        velocity=ports.point_velocity(rows['jacobian'],warm.detach())
        closing=-((velocity[:,:,1]-velocity[:,:,0])*snapshot.normal).sum(-1)
        depth=rows['raw_depth']
        threatened=((ports.owners[:,0]==scene.world)[None] & (context.soft.material.stiffness>0)[None]
            & rows.get('force_eligible',torch.ones_like(depth,dtype=torch.bool))
            & (closing>0) & (depth+closing>=0))
        selected=torch.where(threatened[0])[0]
        owners=ports.owners[selected]
        participating=(owners>=0)&(ports.weights[selected]>0).any(1)
        safe=owners.clamp_min(0)
        participating&=(scene.kind[safe]<2)&scene.movable[safe]
        normals=snapshot.normal[selected,None].expand(-1,owners.shape[1],-1)
        ids=owners[participating];normal=normals[participating]
    output,count=project_closing(warm,ids,normal)
    return output,{'enabled':True,'projected_particles':count,'threatened_static_rows':len(selected)}
