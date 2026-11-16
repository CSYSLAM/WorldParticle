"""Serialize static caller programs separately from physical solver state."""
import numpy as np
import torch

from .caller_pose import RigidCallerPose
from .rigid_feedback import RigidTargetFeedback
from .rigid_caller_program import RigidCallerProgram

POSE_FIELDS=('align_ids','next_ids','next_offsets','follower_ids','parent_ids','follower_offsets','follower_rotations')


def from_example(scene,example):
    """Authored RJ45 controller adapter; shared neural weights know no IDs."""
    if scene.metadata.get('official_demo')!='contacts.example_contacts_rj45_plug':
        raise ValueError('No caller-program adapter registered for this example')
    def native(name,dtype):return torch.as_tensor(getattr(example,name).numpy(),device=scene.device,dtype=dtype)
    followers=native('_cable_anchor_indices',torch.long);primary=int(example._plug_body);latch=int(example._latch_body)
    pose=RigidCallerPose(scene.body_com,native('_cable_align_indices',torch.long),native('_cable_align_next',torch.long),
        native('_cable_align_next_start_offsets',torch.float64),followers,torch.full_like(followers,primary),
        native('_cable_anchor_offsets',torch.float64),native('_cable_anchor_rotations',torch.float64))
    feedback=RigidTargetFeedback(scene.body_com,scene.mass,primary,followers.new_tensor([latch]),followers.new_tensor([primary,latch]))
    return RigidCallerProgram(pose,feedback,int(example.sim_substeps),torch.tensor([False,True,False],device=scene.device))


def to_values(program):
    values={'caller__'+name:getattr(program.pose,name).detach().cpu().numpy().copy() for name in POSE_FIELDS}
    for name in ('followers','gravity_compensated'):
        values['caller__feedback_'+name]=getattr(program.feedback,name).detach().cpu().numpy().copy()
    values.update(caller__primary=np.asarray(program.feedback.primary,np.int64),
        caller__steps_per_sample=np.asarray(program.steps_per_sample,np.int64),
        caller__damping_mass_offset=np.asarray(program.feedback.damping_mass_offset,np.float64),
        caller__follow_axes=program.follow_axes.cpu().numpy().copy())
    return values


def from_values(scene,values):
    def tensor(name,dtype):
        value=torch.as_tensor(values['caller__'+name],device=scene.device)
        if value.dtype!=dtype:raise ValueError('Exact caller field dtype required: '+name)
        return value
    ids={'align_ids','next_ids','follower_ids','parent_ids'}
    pose=RigidCallerPose(scene.body_com,*(tensor(name,torch.long if name in ids else torch.float64) for name in POSE_FIELDS))
    primary=tensor('primary',torch.long);period=tensor('steps_per_sample',torch.long);offset=tensor('damping_mass_offset',torch.float64)
    if any(v.ndim for v in (primary,period,offset)):raise ValueError('Scalar caller ownership/period/mass offset required')
    feedback=RigidTargetFeedback(scene.body_com,scene.mass,int(primary),tensor('feedback_followers',torch.long),
        tensor('feedback_gravity_compensated',torch.long),float(offset))
    return RigidCallerProgram(pose,feedback,int(period),tensor('follow_axes',torch.bool))


def time_channels(scene,values,device):
    """Validate full caller memory/control sequences, excluding label inputs."""
    from . import math3d as so3
    if any(k in values for k in ('input_force','input_torque','input_pin_x','input_pin_q')):
        raise ValueError('Recorded feedback forces and prescribed pose labels are not caller inputs')
    steps=scene.metadata['teacher_substeps'];program=scene.caller_program
    specs={'history__caller_target':((steps+1,3),torch.float64),
        'history__caller_phase':((steps+1,),torch.long),'history__caller_align':((steps+1,),torch.bool),
        'input_caller_stiffness':((steps,),torch.float64),'input_caller_damping':((steps,),torch.float64),
        'input_caller_picked_body':((steps,),torch.long)}
    channels={}
    for name,(shape,dtype) in specs.items():
        value=torch.as_tensor(values[name],device=device)
        if value.shape!=shape or value.dtype!=dtype or not bool(torch.isfinite(value).all()):raise ValueError('Complete typed caller time channel required: '+name)
        channels[name]=value
    counter=torch.arange(steps+1,device=device)
    if not torch.equal(channels['history__caller_phase'],counter%program.steps_per_sample) or not torch.equal(channels['history__caller_align'],counter>0):
        raise ValueError('Caller substep phases must cover the complete accepted reference sequence')
    if bool((channels['input_caller_picked_body']!=-1).any()):raise ValueError('This teacher contract contains no interactive pick/gizmo actions')
    if any(bool((channels[name]<0).any()) for name in ('input_caller_stiffness','input_caller_damping')):
        raise ValueError('Nonnegative exogenous caller gains required')
    x=torch.as_tensor(values['x'],device=device);q=torch.as_tensor(values['q'],device=device)
    primary=program.feedback.primary;com=scene.body_com.to(device)
    origin=x[:,primary]-so3.rotate(q[:,primary],com[primary])
    target=channels['history__caller_target'];axes=program.follow_axes.to(device)
    expected=torch.where(((counter[1:]%program.steps_per_sample)==0)[:,None]&axes[None],origin[1:],target[:-1])
    if not torch.equal(target[0],origin[0]) or not torch.equal(target[1:],expected):
        raise ValueError('Recorded target memory must follow only past accepted frame outputs')
    return channels
