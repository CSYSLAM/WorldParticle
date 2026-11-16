"""Differentiable caller forces and sampled targets on predicted rigid states."""
from dataclasses import dataclass
import math

import torch

from . import math3d as so3


@dataclass(frozen=True)
class RigidTargetFeedback:
    """Origin-target spring with shared acceleration and gravity compensation.

    Linear velocity and force use COM convention; the target tracks the
    primary body's origin. A picked-body mode applies damping to the other
    controlled bodies. This is a caller input operator, not a solver step or
    a material-history update.
    """
    body_com:torch.Tensor
    mass:torch.Tensor
    primary:int
    followers:torch.Tensor
    gravity_compensated:torch.Tensor
    damping_mass_offset:float=10.

    def __post_init__(self):
        com=self.body_com
        if com.ndim!=2 or com.shape[1]!=3 or com.dtype not in (torch.float32,torch.float64) or not bool(torch.isfinite(com).all()):
            raise ValueError('Finite floating body COM coordinates required')
        if self.mass.shape!=(len(com),) or self.mass.dtype!=com.dtype or self.mass.device!=com.device or not bool(torch.isfinite(self.mass).all()) or bool((self.mass<0).any()):
            raise ValueError('Matched finite nonnegative native body masses required')
        if type(self.primary) is not int or not 0<=self.primary<len(com) or not bool(self.mass[self.primary]>0):
            raise ValueError('Positive-mass primary rigid owner required')
        for ids in (self.followers,self.gravity_compensated):
            if ids.ndim!=1 or ids.dtype!=torch.long or ids.device!=com.device or bool(((ids<0)|(ids>=len(com))).any()) or len(ids.unique())!=len(ids):
                raise ValueError('Unique same-device rigid body IDs required')
        if bool((self.followers==self.primary).any()):raise ValueError('Primary cannot also be a follower')
        if isinstance(self.damping_mass_offset,bool) or not math.isfinite(self.damping_mass_offset) or self.damping_mass_offset<0:
            raise ValueError('Finite nonnegative controller mass offset required')

    def _state(self,state):
        shape=state.x.shape
        if len(shape)!=3 or shape[1:]!=self.body_com.shape or state.q.shape!=(*shape[:2],4) or state.v.shape!=shape:
            raise ValueError('Complete batched matched rigid COM state required')
        for value in (state.x,state.q,state.v):
            if value.dtype!=self.body_com.dtype or value.device!=self.body_com.device or not bool(torch.isfinite(value).all()):
                raise ValueError('Finite same-device/dtype rigid caller state required')
        if not bool(((state.q.square().sum(-1)-1).abs()<1e-5).all()):raise ValueError('Normalized caller orientations required')
        return state.x[:,self.primary]-so3.rotate(state.q[:,self.primary],self.body_com[self.primary])

    def _target(self,state,target):
        if target.shape!=(len(state.x),3) or target.dtype!=state.x.dtype or target.device!=state.x.device or not bool(torch.isfinite(target).all()):
            raise ValueError('Finite matched per-batch target position required')

    def sample_target(self,state,held_target,follow_axes,refresh):
        """Refresh selected coordinates only at an explicit caller sample.

        For frame-held RJ45 input, follow_axes=(False,True,False) and refresh
        is true at frame boundaries. The caller carries the returned target
        through its substeps; it never reads a future reference target.
        """
        origin=self._state(state);self._target(state,held_target)
        if follow_axes.shape!=(3,) or follow_axes.dtype!=torch.bool or follow_axes.device!=state.x.device:
            raise ValueError('Explicit same-device three-axis sample mask required')
        if refresh.shape!=(len(state.x),) or refresh.dtype!=torch.bool or refresh.device!=state.x.device:
            raise ValueError('Explicit per-batch sample phase required')
        return torch.where(refresh[:,None]&follow_axes[None],origin,held_target)

    def forces(self,state,target,gravity,stiffness,damping,picked_body):
        origin=self._state(state);self._target(state,target);batch=len(state.x)
        if gravity.shape not in ((3,),(batch,3)) or gravity.dtype!=state.x.dtype or gravity.device!=state.x.device or not bool(torch.isfinite(gravity).all()):
            raise ValueError('Finite matched gravity vector required')
        if picked_body.shape!=(batch,) or picked_body.dtype!=torch.long or picked_body.device!=state.x.device or bool(((picked_body< -1)|(picked_body>=len(self.mass))).any()):
            raise ValueError('Explicit per-batch pick mode/body required')
        def coefficient(value):
            if isinstance(value,bool):raise ValueError('Floating controller coefficient required')
            result=torch.as_tensor(value,device=state.x.device,dtype=state.x.dtype)
            if result.ndim==0:result=result.expand(batch)
            if result.shape!=(batch,) or not bool(torch.isfinite(result).all()) or bool((result<0).any()):
                raise ValueError('Finite nonnegative per-batch controller coefficient required')
            return result[:,None]
        k,d=coefficient(stiffness),coefficient(damping)
        ids=torch.cat((self.followers.new_tensor([self.primary]),self.followers))
        mass=self.mass[ids];offset=self.damping_mass_offset
        error=target-origin;v=state.v[:,ids]
        damp_force=-(offset+mass)[None,:,None]*d[:,None]*v
        acceleration=error*((offset+self.mass[self.primary])*k/self.mass[self.primary])
        spring=acceleration[:,None]*mass[None,:,None]+damp_force
        # Retain the pinned primary expression's arithmetic grouping.
        primary=(offset+self.mass[self.primary])*(k*error-d*state.v[:,self.primary])
        spring=torch.cat((primary[:,None],spring[:,1:]),1)
        picked=ids[None]!=picked_body[:,None]
        controlled=torch.where((picked_body<0)[:,None,None],spring,torch.where(picked[...,None],damp_force,0.))
        force=torch.zeros_like(state.x)
        g=gravity.expand(batch,3)
        force=force.index_add(1,self.gravity_compensated,-g[:,None]*self.mass[self.gravity_compensated][None,:,None])
        force=force.index_add(1,ids,controlled)
        return force,torch.zeros_like(force)
