"""Direction-preserving shared port prototype; not in the deployed v18 model.

Six scalar-indexed columns carry polar/axial vectors, rather than six world
axis carriers modulated by scalar invariants. Incoming conditioning at a fixed
node is discarded. Projection and inverse scaling remain the caller's job.
"""
import torch
from torch import nn

class SharedVectorPorts(nn.Module):
    def __init__(self,width=32):
        super().__init__()
        self.shape=nn.Linear(width,6)
        nn.init.zeros_(self.shape.weight);nn.init.zeros_(self.shape.bias)

    def forward(self,scene,state,physical,latent,relation,dt,*,unit_shapes=False):
        mobile=scene.movable;layout=scene.layout
        proposal=torch.where(scene.dof[None],physical,0.)
        active=relation[...,27].to(proposal.dtype)
        pooled=proposal.new_zeros(len(proposal),len(layout.incidence),6)
        count=proposal.new_zeros(len(layout.incidence))
        for rows,ids in layout.slots:
            pooled=pooled.index_add(1,rows,proposal[:,ids])
            count=count.index_add(0,rows,mobile[ids].to(count.dtype))
        pooled=pooled/count.clamp_min(1)[None,:,None]
        spread=torch.zeros_like(proposal);degree=proposal.new_zeros(len(proposal),scene.n)
        for rows,ids in layout.slots:
            message=(pooled[:,rows]-proposal[:,ids])*active[:,rows,None]
            spread=spread.index_add(1,ids,message)
            degree=degree.index_add(1,ids,active[:,rows])
        spread=spread/degree.clamp_min(1)[...,None]
        spread=torch.where(scene.dof[None],spread,0.)
        velocity=torch.cat((state.v,state.omega),-1)*dt
        scale=proposal.new_ones(len(proposal),scene.n,6) if unit_shapes else (1+.5*self.shape(latent).tanh()).to(proposal.dtype)
        out=proposal.new_zeros(len(proposal),scene.n,6,6)
        for index,(field,rotation) in enumerate(((proposal,False),(proposal,True),(spread,False),(spread,True),(velocity,False),(velocity,True))):
            dofs=slice(3,6) if rotation else slice(0,3)
            out[...,dofs,index]=field[...,dofs]*scale[...,index,None]
        return out*scene.dof[None,:,:,None]
