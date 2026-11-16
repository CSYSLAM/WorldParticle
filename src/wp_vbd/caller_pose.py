"""Causal caller pose edits between accepted rigid COM solver states.

These are explicit scene controllers, not a dynamics or history update. The
alignment pass reads frozen neighbour poses; followers read its completed
output. The caller must apply each phase exactly once.
"""
from dataclasses import dataclass

import torch

from . import math3d as so3
from .data import State


def _swing(source,target):
    """Newton's robust shortest arc, with its frozen singularity branches."""
    parallel=(source-target).square().sum(-1)<=2e-8
    opposite=(source+target).square().sum(-1)<=2e-8
    helper=torch.zeros_like(source)
    helper[...,0]=1.
    other=torch.zeros_like(source);other[...,1]=1.
    helper=torch.where((source[...,0].abs()>=.9)[...,None],other,helper)
    axis=torch.linalg.cross(source,helper)
    final=torch.zeros_like(source);final[...,2]=1.
    axis=torch.where((axis.norm(dim=-1)<=1e-8)[...,None],torch.linalg.cross(source,final),axis)
    fallback=torch.zeros_like(source);fallback[...,0]=1.
    axis=torch.where((axis.norm(dim=-1)<=1e-8)[...,None],fallback,axis)
    axis=axis/axis.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    angle=source.new_tensor(torch.pi/2)
    half_turn=torch.cat((axis*angle.sin(),angle.cos().expand(*axis.shape[:-1],1)),-1)
    ordinary=so3.normalize(torch.cat((torch.linalg.cross(source,target),1+(source*target).sum(-1,keepdim=True)),-1))
    identity=torch.zeros_like(ordinary);identity[...,3]=1.
    return torch.where(parallel[...,None],identity,torch.where(opposite[...,None],half_turn,ordinary))


@dataclass(frozen=True)
class RigidCallerPose:
    """Two ordered passes on COM state, with immutable physical history.

    Each alignment aims local +Z at a frozen neighbour's local endpoint and
    keeps the body's origin fixed. Each follower copies a parent-relative
    origin/rotation and sets both velocity components to zero. Followers all
    read the same post-alignment snapshot, so no ordering is inferred.
    """
    body_com:torch.Tensor
    align_ids:torch.Tensor
    next_ids:torch.Tensor
    next_offsets:torch.Tensor
    follower_ids:torch.Tensor
    parent_ids:torch.Tensor
    follower_offsets:torch.Tensor
    follower_rotations:torch.Tensor

    def __post_init__(self):
        com=self.body_com
        if com.ndim!=2 or com.shape[1]!=3 or com.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(com).all()):
            raise ValueError('Finite float32/float64 N-by3 body COM coordinates required')
        for ids in (self.align_ids,self.next_ids,self.follower_ids,self.parent_ids):
            if ids.ndim!=1 or ids.dtype!=torch.long or ids.device!=com.device or bool(((ids<0)|(ids>=len(com))).any()):
                raise ValueError('Valid same-device body indices required')
        if len(self.align_ids)!=len(self.next_ids) or len(self.follower_ids)!=len(self.parent_ids):
            raise ValueError('Matched alignment/follower owners required')
        if len(self.align_ids.unique())!=len(self.align_ids) or len(self.follower_ids.unique())!=len(self.follower_ids):
            raise ValueError('Each pass requires unique edited bodies')
        for value,shape in ((self.next_offsets,(len(self.align_ids),3)),
                            (self.follower_offsets,(len(self.follower_ids),3)),
                            (self.follower_rotations,(len(self.follower_ids),4))):
            if value.shape!=shape or value.device!=com.device or value.dtype!=com.dtype or not bool(torch.isfinite(value).all()):
                raise ValueError('Finite same-device/dtype local edit frames required')
        if bool((self.follower_rotations.norm(dim=-1)<=1e-12).any()):
            raise ValueError('Nondegenerate follower quaternions required')

    def apply(self,state,*,align_previous_output,apply_followers=True):
        if type(align_previous_output) is not bool or type(apply_followers) is not bool:
            raise ValueError('Explicit first-input versus post-solve phase required')
        n=len(self.body_com)
        for value,width in ((state.x,3),(state.q,4),(state.v,3),(state.omega,3)):
            if value.ndim!=3 or value.shape[1:]!=(n,width) or len(value)!=len(state.x) or value.device!=self.body_com.device or value.dtype!=self.body_com.dtype:
                raise ValueError('Matched batched rigid COM state required')
            if not bool(torch.isfinite(value).all()):raise ValueError('Finite caller pose input required')
        if bool((state.q.norm(dim=-1)<=1e-12).any()):raise ValueError('Nondegenerate input quaternions required')
        x,q,v,omega=state.x,state.q,state.v,state.omega
        if align_previous_output and len(self.align_ids):
            ids,nxt=self.align_ids,self.next_ids
            origin=x-so3.rotate(q,self.body_com)
            segment=origin[:,nxt]+so3.rotate(q[:,nxt],self.next_offsets)-origin[:,ids]
            length=segment.norm(dim=-1,keepdim=True)
            axis=torch.zeros_like(segment);axis[...,2]=1.
            swing=_swing(so3.rotate(q[:,ids],axis),segment/length.clamp_min(1e-10))
            rotation=so3.normalize(so3.multiply(swing,q[:,ids]))
            rotation=torch.where(length>=1e-10,rotation,q[:,ids])
            position=origin[:,ids]+so3.rotate(rotation,self.body_com[ids])
            position=torch.where(length>=1e-10,position,x[:,ids])
            x=x.index_copy(1,ids,position);q=q.index_copy(1,ids,rotation)
        if apply_followers and len(self.follower_ids):
            ids,parents=self.follower_ids,self.parent_ids
            parent_origin=x[:,parents]-so3.rotate(q[:,parents],self.body_com[parents])
            rotation=so3.normalize(so3.multiply(q[:,parents],self.follower_rotations))
            origin=parent_origin+so3.rotate(q[:,parents],self.follower_offsets)
            x=x.index_copy(1,ids,origin+so3.rotate(rotation,self.body_com[ids]))
            q=q.index_copy(1,ids,rotation)
            v=v.index_copy(1,ids,torch.zeros_like(v[:,ids]))
            omega=omega.index_copy(1,ids,torch.zeros_like(omega[:,ids]))
        # Physical damping/constitutive history is the previous unedited
        # accepted state. Rebaselining it here would change the source model.
        return State(x,q,v,omega,state.history)
