"""Accepted COM-pose history for prescribed rigid motion.

Prescribed bodies have no response DOFs even when the native asset inverse
mass is positive. Their actual previous accepted poses enter damping/friction;
the current commanded pose stays fixed throughout the physical step.
"""
import torch
from . import math3d as so3
from .data import State

HISTORY_KEYS={'prescribed_x_prev','prescribed_q_prev','prescribed_initialized'}


def enabled(scene):return scene.metadata.get('prescribed_pose_history')=='accepted_com_pose_v30'


def configure(scene, *, prescribed_ids=None):
    if prescribed_ids is None:
        flags=scene.values['body_flags'].long()
        prescribed_ids=torch.where(flags==2)[0]
    if prescribed_ids.ndim!=1 or prescribed_ids.dtype!=torch.long or prescribed_ids.device!=scene.device:
        raise ValueError('Matched explicit rigid owner indices required')
    if bool(((prescribed_ids<0)|(prescribed_ids>=scene.n)).any()) or len(torch.unique(prescribed_ids))!=len(prescribed_ids):
        raise ValueError('Invalid or duplicate prescribed rigid owner')
    if hasattr(scene,'body') and not bool(scene.body[prescribed_ids].all()):
        raise ValueError('A particle cannot own a prescribed rigid pose')
    scene.prescribed_ids=prescribed_ids.clone()
    scene.movable=scene.movable.clone();scene.movable[scene.prescribed_ids]=False
    # Mixed particles retain translation-only DOFs. Ownership changes must
    # never introduce their three nonexistent angular degrees of freedom.
    scene.dof=scene.dof & scene.movable[:,None]
    # Asset inverse inertia remains in scene.values. These fields are the
    # effective response used by Delassus supports and whitening.
    scene.inverse_inertia=scene.inverse_inertia.clone();scene.inverse_inertia[scene.prescribed_ids]=0.
    scene.inverse_inertia_sqrt=scene.inverse_inertia_sqrt.clone();scene.inverse_inertia_sqrt[scene.prescribed_ids]=0.
    for name in ('elastic_scene','objective_scene'):
        child=getattr(scene,name,None)
        if child is not None:
            for key in ('movable','dof','inverse_inertia','inverse_inertia_sqrt'):
                if hasattr(child,key):setattr(child,key,getattr(scene,key))
    scene._graph_signature=None
    scene.update_graph(torch.empty(0,2,device=scene.device,dtype=torch.long),torch.empty(0,device=scene.device,dtype=torch.bool))


def initialize(scene,state):
    ids=scene.prescribed_ids
    return {'prescribed_x_prev':state.x[:,ids].clone(),'prescribed_q_prev':state.q[:,ids].clone(),'prescribed_initialized':torch.zeros(len(state.x),device=state.x.device,dtype=torch.bool)}


def previous(scene,start):
    ids=scene.prescribed_ids;h=start.history;b=len(start.x)
    if not isinstance(h,dict) or not HISTORY_KEYS.issubset(h):raise ValueError('Complete prescribed pose history must be restored')
    x,q,initialized=(h[k] for k in ('prescribed_x_prev','prescribed_q_prev','prescribed_initialized'))
    if x.shape!=(b,len(ids),3) or q.shape!=(b,len(ids),4) or initialized.shape!=(b,) or initialized.dtype!=torch.bool:raise ValueError('Invalid prescribed pose history shapes/types')
    if not bool(torch.isfinite(x).all()&torch.isfinite(q).all()) or not bool(((q.square().sum(-1)-1).abs()<1e-8).all()):raise ValueError('Invalid prescribed COM pose history')
    old_x=torch.where(initialized[:,None,None],x,start.x[:,ids]);old_q=torch.where(initialized[:,None,None],q,start.q[:,ids])
    return State(start.x.index_copy(1,ids,old_x),start.q.index_copy(1,ids,old_q),start.v,start.omega,start.history)


def domain(scene,start,dt):
    old=previous(scene,start);ids=scene.prescribed_ids
    angle=so3.log(so3.multiply(start.q[:,ids],so3.conjugate(old.q[:,ids]))).norm(dim=-1)
    travel=(start.x[:,ids]-old.x[:,ids]).norm(dim=-1)+scene.body_radius[ids]*angle
    valid=(travel<=scene.geometry_oracle.motion_budget).all(-1)&(angle<torch.pi-1e-6).all(-1)
    valid&=(start.omega[:,ids].norm(dim=-1)*dt<torch.pi-1e-6).all(-1)
    return old,valid,angle


def commit(scene,start,output,context,accepted,history,velocity,omega):
    ids=scene.prescribed_ids;old=context.material_start
    measured_v=(output.x[:,ids]-old.x[:,ids])/context.dt
    measured_w=so3.log(so3.multiply(output.q[:,ids],so3.conjugate(old.q[:,ids])))/context.dt
    velocity=velocity.index_copy(1,ids,torch.where(accepted[:,None,None],measured_v,start.v[:,ids]))
    omega=omega.index_copy(1,ids,torch.where(accepted[:,None,None],measured_w,start.omega[:,ids]))
    history.update(prescribed_x_prev=torch.where(accepted[:,None,None],output.x[:,ids],start.history['prescribed_x_prev']),prescribed_q_prev=torch.where(accepted[:,None,None],output.q[:,ids],start.history['prescribed_q_prev']),prescribed_initialized=torch.where(accepted,True,start.history['prescribed_initialized']))
    return history,velocity,omega
