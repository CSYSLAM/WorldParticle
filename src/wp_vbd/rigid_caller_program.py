"""Functional between-step pose, feedback and frame-held caller memory."""
from dataclasses import dataclass

import torch

from .data import State
from .caller_pose import RigidCallerPose
from .rigid_feedback import RigidTargetFeedback

HISTORY_KEYS=frozenset(('caller_target','caller_phase','caller_align'))
SCHEMA='causal_rigid_pose_feedback_frame_hold_v215'


@dataclass(frozen=True)
class CallerMemory:
    target:torch.Tensor
    phase:torch.Tensor
    align:torch.Tensor

    @classmethod
    def from_history(cls,history):
        if not isinstance(history,dict) or not HISTORY_KEYS<=history.keys():
            raise ValueError('Complete caller target/phase/alignment memory required')
        return cls(*(history['caller_'+name] for name in ('target','phase','align')))

    def history(self):
        return {'caller_'+name:getattr(self,name) for name in ('target','phase','align')}


@dataclass(frozen=True)
class RigidCallerProgram:
    pose:RigidCallerPose
    feedback:RigidTargetFeedback
    steps_per_sample:int
    follow_axes:torch.Tensor

    def __post_init__(self):
        if type(self.steps_per_sample) is not int or self.steps_per_sample<1:
            raise ValueError('Positive integer caller sample period required')
        if not torch.equal(self.pose.body_com,self.feedback.body_com):
            raise ValueError('Pose and feedback must own the same COM coordinate space')
        axes=self.follow_axes
        if axes.shape!=(3,) or axes.dtype!=torch.bool or axes.device!=self.pose.body_com.device:
            raise ValueError('Explicit same-device three-axis target sampling mask required')

    def validate(self,state,memory):
        b=len(state.x);device=state.x.device
        self.feedback._state(state);self.feedback._target(state,memory.target)
        if memory.phase.shape!=(b,) or memory.phase.dtype!=torch.long or memory.phase.device!=device:
            raise ValueError('One same-device integer caller phase per sequence required')
        if bool(((memory.phase<0)|(memory.phase>=self.steps_per_sample)).any()):
            raise ValueError('Caller phase outside its sample period')
        if memory.align.shape!=(b,) or memory.align.dtype!=torch.bool or memory.align.device!=device:
            raise ValueError('One explicit previous-output alignment flag per sequence required')

    def prepare(self,state,memory,gravity,stiffness,damping,picked_body):
        """Align past output, compute feedback, then synchronize followers.

        Source feedback is evaluated before the follower reset. In particular
        the force law must not accidentally read a follower's reset velocity.
        Initial raw input skips alignment through an explicit per-sequence bit.
        Physical history and caller memory are not mutated by preparation.
        """
        self.validate(state,memory)
        aligned=self.pose.apply(state,align_previous_output=True,apply_followers=False)
        mask=memory.align[:,None,None]
        aligned=State(torch.where(mask,aligned.x,state.x),torch.where(mask,aligned.q,state.q),state.v,state.omega,state.history)
        force,torque=self.feedback.forces(aligned,memory.target,gravity,stiffness,damping,picked_body)
        prepared=self.pose.apply(aligned,align_previous_output=False)
        return prepared,force,torque

    def commit(self,output,memory,accepted,*,elapsed_steps=1):
        """Sample only an accepted predicted frame end; otherwise keep memory.

        Alignment preserves body origins, so the origin sampled after the
        native post-solve alignment is the raw accepted output's origin.
        This caller state never replaces damping or constitutive history.
        """
        self.validate(output,memory)
        if accepted.shape!=memory.phase.shape or accepted.dtype!=torch.bool or accepted.device!=output.x.device:
            raise ValueError('One explicit accepted-transaction bit per sequence required')
        if type(elapsed_steps) is not int or elapsed_steps<1:
            raise ValueError('Positive integer elapsed caller ticks required')
        if bool((memory.phase+elapsed_steps>self.steps_per_sample).any()):
            raise ValueError('Split the physical interval at the caller sample boundary')
        phase=(memory.phase+elapsed_steps)%self.steps_per_sample
        target=self.feedback.sample_target(output,memory.target,self.follow_axes,accepted&(phase==0))
        return CallerMemory(target,torch.where(accepted,phase,memory.phase),memory.align|accepted)


def physical_state(state):
    """Keep controller state outside the backend's strict physical history."""
    CallerMemory.from_history(state.history)
    return State(state.x,state.q,state.v,state.omega,{k:v for k,v in state.history.items() if k not in HISTORY_KEYS})


def attach_memory(state,memory):
    if state.history is None or HISTORY_KEYS&state.history.keys():
        raise ValueError('Backend must return only its complete physical history')
    return State(state.x,state.q,state.v,state.omega,{**state.history,**memory.history()})
