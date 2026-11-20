"""Export non-rod finite joint rows from authored Newton model fields.

No solver constructor or private solver state is read. Export is a static
snapshot; a future live adapter must refresh model properties and controls
at their authored update times. Unsupported features fail before admission.
"""
from types import SimpleNamespace
import numpy as np
import torch
from . import math3d as so3


def export_spec(model,configuration,newton,device='cpu',*,allow_mimic=False,allow_rods=False):
    if not configuration['rigid_compliant_alm']:raise ValueError('Only finite compliant joint material is implemented')
    alpha=configuration.get('rigid_avbd_joint_alpha')
    if alpha is None:alpha=configuration.get('rigid_avbd_alpha') or 0.
    if alpha!=0:raise ValueError('Nonzero joint stabilization requires an explicit C0 lifecycle')
    mimic=getattr(model,'joint_mimic_joint',None)
    if not allow_mimic and mimic is not None and np.any(mimic.numpy()>=0):raise ValueError('Mimic projection has a separate admission gate')
    jt=model.joint_type.numpy();free=int(newton.JointType.FREE)
    admitted={int(getattr(newton.JointType,name)) for name in ('BALL','FIXED','REVOLUTE','PRISMATIC','D6','FREE')}
    if allow_rods:admitted.add(int(newton.JointType.ROD))
    if any(int(value) not in admitted for value in jt):raise ValueError('ROD and unsupported joint types require their dedicated constitutive contract')
    selected=jt!=free
    if allow_rods:selected&=jt!=int(newton.JointType.ROD)
    ids=np.where(selected)[0];count=len(ids);world=model.body_count
    parent=model.joint_parent.numpy()[ids];child=model.joint_child.numpy()[ids]
    if np.any(parent < -1) or np.any(parent>=world) or np.any(child<0) or np.any(child>=world):raise ValueError('Joint body ownership is invalid')
    owners=np.stack((np.where(parent<0,world,parent),child),-1)
    frames=np.stack((model.joint_X_p.numpy()[ids],model.joint_X_c.numpy()[ids]),1)
    axis=model.joint_axis.numpy();dims=model.joint_dof_dim.numpy();qd_start=model.joint_qd_start.numpy();q_start=model.joint_q_start.numpy();target_start=model.joint_target_q_start.numpy();joint_q=model.joint_q.numpy()
    linear_axes=np.zeros((count,3,3));angular_axes=np.zeros_like(linear_axes)
    mode=np.zeros(count,np.int64);offset=np.zeros((count,6));dof_index=np.full((count,6),-1,np.int64);target_index=np.full_like(dof_index,-1)
    lp=np.tile(np.eye(3),(count,1,1));ap=lp.copy()
    for index,joint in enumerate(ids):
        kind=int(jt[joint]);lin,ang=map(int,dims[joint]);start=int(qd_start[joint])
        if kind==int(newton.JointType.BALL):ap[index]=0;continue
        if kind==int(newton.JointType.FIXED):continue
        if max(lin,ang)>3:raise ValueError('At most three linear and angular axes are supported')
        for begin,n,output,slot in ((start,lin,linear_axes,0),(start+lin,ang,angular_axes,3)):
            selected=axis[begin:begin+n].astype(np.float64)
            if n:
                norm=np.linalg.norm(selected,axis=-1)
                if np.any(norm<1e-10):raise ValueError('Zero free joint axis')
                selected/=norm[:,None]
                if not np.allclose(selected@selected.T,np.eye(n),atol=1e-5,rtol=0):raise ValueError('Source joint projector requires orthonormal axes')
                output[index,:n]=selected
                dof_index[index,slot:slot+n]=np.arange(begin,begin+n)
                target_index[index,slot:slot+n]=int(target_start[joint])+np.arange(begin-start,begin-start+n)
        lp[index]-=linear_axes[index].T@linear_axes[index];ap[index]-=angular_axes[index].T@angular_axes[index]
        if ang==1:
            mode[index]=1;offset[index,3]=joint_q[int(q_start[joint])+lin]
        elif ang>=2:mode[index]=ang
    valid=dof_index>=0;safe=np.maximum(dof_index,0)
    def per_dof(name,default=0.):
        data=getattr(model,name).numpy()
        value=np.full((count,6),default,np.float64)
        value[valid]=data[safe[valid]]
        return value
    values={'owners':owners,'source_joint_ids':ids,'frames':frames,'linear_axes':linear_axes,'angular_axes':angular_axes,'linear_projector':lp,'angular_projector':ap,'angular_mode':mode,'coordinate_offset':offset,'dof_index':dof_index,'target_index':target_index,'enabled':model.joint_enabled.numpy()[ids],
            'stiffness':np.tile([configuration['rigid_joint_linear_ke']]*3+[configuration['rigid_joint_angular_ke']]*3,(count,1)),
            'damping':np.tile([configuration['rigid_joint_linear_kd']]*3+[configuration['rigid_joint_angular_kd']]*3,(count,1)),
            'drive_k':per_dof('joint_target_ke'),'drive_d':per_dof('joint_target_kd'),'lower':per_dof('joint_limit_lower',-1e10),'upper':per_dof('joint_limit_upper',1e10),'limit_k':per_dof('joint_limit_ke'),'limit_d':per_dof('joint_limit_kd')}
    result=SimpleNamespace(**{key:torch.as_tensor(value,device=device) for key,value in values.items()})
    result.frames=result.frames.double()
    for name in ('stiffness','damping','drive_k','drive_d','limit_k','limit_d'):
        value=getattr(result,name)
        if not bool(torch.isfinite(value).all()&(value>=0).all()):raise ValueError(f'Invalid finite material {name}')
    if bool(torch.isnan(result.lower).any()|torch.isnan(result.upper).any()|(result.lower>result.upper).any()):raise ValueError('Invalid joint limits')
    pose=torch.as_tensor(model.body_q.numpy(),device=device,dtype=torch.float64)
    pose=torch.cat((pose,pose.new_tensor([[0.,0.,0.,0.,0.,0.,1.]])),0)
    result.rest_parent=so3.multiply(pose[result.owners[:,0],3:],result.frames[:,0,3:])
    result.rest_child=so3.multiply(pose[result.owners[:,1],3:],result.frames[:,1,3:])
    return result


def read_controls(spec,control):
    """Read absolute position targets, velocity targets and scalar efforts."""
    device=spec.frames.device
    import warp as wp
    def read(value):
        if value is None or value.size==0:return torch.empty(0,device=device,dtype=torch.float64)
        # Native CUDA arrays expose their storage directly. Controls never
        # pass through numpy/host memory in the resident dynamics path.
        return wp.to_torch(value).to(device=device,dtype=torch.float64)
    target=read(control.joint_target_q)
    velocity=read(control.joint_target_qd)
    effort=read(control.joint_f)
    valid=spec.dof_index>=0;safe=spec.dof_index.clamp_min(0);qindex=spec.target_index.clamp_min(0)
    def gather(value,index):
        if not value.numel():
            if bool(valid.any()):raise ValueError('Missing authored joint control array')
            return spec.coordinate_offset.new_zeros(spec.dof_index.shape)
        return torch.where(valid,value[index],0.)
    return gather(target,qindex)[None],gather(velocity,safe)[None],gather(effort,safe)[None]
