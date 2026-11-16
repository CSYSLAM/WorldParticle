"""Explicit removal of prescribed geometry at a physical-step boundary."""
import torch

from . import math3d as so3
from .data import State


def validate_removal_bodies(scene, bodies):
    ids=torch.as_tensor(bodies,device=scene.device,dtype=torch.long).flatten()
    if len(ids)!=len(torch.unique(ids)) or bool(((ids<0)|(ids>=scene.world)).any()):
        raise ValueError('Distinct real prescribed geometry bodies required')
    if bool(scene.movable[ids].any()) or bool(torch.isin(ids,scene.rod.joints.flatten()).any()):
        raise ValueError('Geometry removal cannot rebaseline dynamic or rod-connected bodies')
    if bool((scene.body_radius[ids]<=0).any()):
        raise ValueError('Removal event requires prescribed geometry')
    return ids


def rebase_removed_geometry(scene, state, bodies):
    """Rebaseline only a declared non-material boundary, never rod memory."""
    ids=validate_removal_bodies(scene,bodies)
    if not len(ids):return state
    history=dict(state.history)
    history['rod_previous_x']=history['rod_previous_x'].index_copy(1,ids,state.x[:,ids])
    history['rod_previous_q']=history['rod_previous_q'].index_copy(1,ids,state.q[:,ids])
    zero=state.v.new_zeros(len(state.x),len(ids),3)
    return State(state.x,state.q,state.v.index_copy(1,ids,zero),state.omega.index_copy(1,ids,zero),history)


class BundleReleaseEvents:
    """Map the authored bundle's explicit release phase to removal inputs.

    The source advances its device clock after commanding poses, before step.
    A release is recognized only at the declared phase and exact target pose.
    No arbitrary large movement is automatically reclassified as removal.
    """
    def __init__(self,example,scene,wp):
        self.example=example;self.scene=scene;self.wp=wp
        self.ids=validate_removal_bodies(scene,example.obstacle_bodies)
        self.expected=wp.to_torch(example.model.body_q)[self.ids].clone()
        stop=float(example.obstacle_stop_time);period=float(example.obstacle_period)
        cycles=stop/period;frac=cycles-int(cycles);frac=(frac+.5)%1
        tri=2*(1-abs(2*frac-1))-1
        self.expected[:,1]=wp.to_torch(example.obstacle_init_y)*float(example.obstacle_amp_scale)*tri
        self.expected[:,2]=wp.to_torch(example.obstacle_init_z)+10.

    def __call__(self,native,state):
        if float(self.wp.to_torch(self.example.sim_time_array)[0])<float(self.example.obstacle_release_time):
            return self.ids[:0]
        current=self.wp.to_torch(native.body_q)[self.ids]
        h=state.history
        q=h['rod_previous_q'][0,self.ids]
        old_origin=h['rod_previous_x'][0,self.ids]-so3.rotate(q,self.scene.body_com[self.ids])
        previous=torch.cat((old_origin,q),-1).to(current.dtype)
        declared=(current==self.expected).all(-1)
        changed=(current!=previous).any(-1)
        return self.ids[declared&changed]
