"""Ideal scalar mimic constraints with a mass-orthogonal tangent response.

No penalty stiffness is invented. Repeated body roles are summed before
forming the <=6-row mass Gram. The shared positive inverse is surrounded
by P and P^T; applying only a left projection need not preserve descent.
This first gate admits one-axis PRISMATIC/REVOLUTE joints, not general D6.
"""
from types import SimpleNamespace
import numpy as np
import torch
from . import math3d as so3
from .data import State
from .joint_coordinates import anchor_coordinates
from .static_topology import tensor_versions


def export_spec(model,joints,newton,device):
    reference=model.joint_mimic_joint.numpy();followers=np.where(reference>=0)[0]
    source={int(value):i for i,value in enumerate(joints.source_joint_ids.tolist())}
    types=model.joint_type.numpy();dims=model.joint_dof_dim.numpy()
    admitted={int(newton.JointType.PRISMATIC),int(newton.JointType.REVOLUTE)}
    pairs=[];linear=[]
    for follower in followers:
        other=int(reference[follower])
        if other>=model.joint_count or other<0:raise ValueError('Invalid mimic reference joint')
        if follower==other:raise ValueError('Self-referencing mimic is not admitted')
        if any(int(types[j]) not in admitted or int(dims[j].sum())!=1 for j in (follower,other)):raise ValueError('Initial ideal mimic gate requires scalar PRISMATIC/REVOLUTE joints')
        pairs.append((source[int(follower)],source[other]))
        linear.append(tuple(int(types[j])==int(newton.JointType.PRISMATIC) for j in (follower,other)))
    coefficients=torch.as_tensor(model.joint_mimic_coeffs.numpy()[followers],device=device,dtype=torch.float64)
    if not bool(torch.isfinite(coefficients).all()):raise ValueError('Nonfinite mimic coefficients')
    return SimpleNamespace(pairs=torch.tensor(pairs,device=device,dtype=torch.long).reshape(-1,2),linear=torch.tensor(linear,device=device,dtype=torch.bool).reshape(-1,2),coefficients=coefficients,source_joint_ids=torch.as_tensor(followers,device=device))


def relation(scene):
    s=scene.mimic_spec;j=scene.joint_spec
    enabled=j.enabled[s.pairs].all(-1)
    owners=j.owners[s.pairs].flatten(1,2).clone()
    # A zero multiplier has no physical or neural incidence on the reference.
    owners[:,2:]=torch.where((s.coefficients[:,1]!=0)[:,None],owners[:,2:],-1)
    return owners,enabled


def rows(scene,state):
    s=scene.mimic_spec;j=scene.joint_spec;pairs=s.pairs
    ids=j.owners
    position=state.x[:,ids];rotation=state.q[:,ids]
    linear,jlinear,_,parent=anchor_coordinates(position,rotation,scene.body_com[ids],j.frames)
    child=so3.multiply(rotation[...,1,:],j.frames[...,1,3:])
    relative=so3.normalize(so3.multiply(so3.conjugate(parent),child))
    axis=j.angular_axes[:,0];v=relative[...,:3];w=relative[...,3];a=(v*axis).sum(-1)
    denominator=w.square()+a.square()
    angle=2*torch.atan2(a,w)
    angle=torch.atan2(angle.sin(),angle.cos())
    covector=(w.square()[...,None]*axis+w[...,None]*torch.linalg.cross(v,axis.expand_as(v))+a[...,None]*v)/denominator.clamp_min(1e-20)[...,None]
    covector=(covector[...,None,:]@so3.matrix(parent).transpose(-1,-2))[...,0,:]
    zero=torch.zeros_like(covector)
    angular_j=torch.stack((torch.cat((zero,-covector),-1),torch.cat((zero,covector),-1)),-2)
    linear_value=(linear*j.linear_axes[:,0]).sum(-1)
    linear_j=torch.einsum('ji,bjisk->bjsk',j.linear_axes[:,0],jlinear)
    values=torch.where(s.linear[None],linear_value[:,pairs],angle[:,pairs])
    jac=torch.where(s.linear[None,:,:,None,None],linear_j[:,pairs],angular_j[:,pairs])
    value=values[...,0]-s.coefficients[:,0]-s.coefficients[:,1]*values[...,1]
    value=torch.where(s.linear[None,:,0],value,torch.atan2(value.sin(),value.cos()))
    jac=torch.cat((jac[:,:,0],-s.coefficients[None,:,1,None,None]*jac[:,:,1]),2)
    scale=torch.where(s.linear[:,0],1/scene.length,1.)
    owners,enabled=relation(scene)
    # Principal twist has no chart at a 180-degree swing about a transverse axis.
    good=(denominator[:,pairs]>1e-10)|s.linear[None]
    valid=(good[...,0]&(good[...,1]|(s.coefficients[:,1]==0)))|~enabled
    jac=jac*enabled[None,:,None,None]*scale[None,:,None,None]
    return SimpleNamespace(value=value*scale*enabled,jacobian=jac,owners=owners,enabled=enabled,valid=valid,coordinates=values)


def group_topology(scene,evaluated,device,*,cache=True):
    sources=()
    if cache:
        spec=scene.joint_spec;mimic=scene.mimic_spec
        sources=(spec.owners,spec.enabled,mimic.pairs,mimic.coefficients,scene.movable)
    versions=tensor_versions(sources) if cache else None
    signature=(versions,str(device)) if versions is not None else None
    cached=getattr(scene,'_mimic_group_topology',None)
    if signature is not None and cached is not None and cached[0]==signature:
        return cached[1]
    active=torch.where(evaluated.enabled)[0].tolist();parents={i:i for i in active}
    def root(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    members={i:{k for k in evaluated.owners[i].tolist() if k>=0 and bool(scene.movable[k])} for i in active}
    seen={}
    for i in active:
        for k in members[i]:
            if k in seen:parents[root(i)]=root(seen[k])
            seen[k]=i
    clusters={}
    for i in active:clusters.setdefault(root(i),[]).append(i)
    groups=[]
    for selected in clusters.values():
        if len(selected)>6:raise ValueError('Coupled ideal mimic group exceeds the declared six-row factor budget')
        owners=sorted(set().union(*(members[i] for i in selected)))
        ids=torch.tensor(owners,device=device,dtype=torch.long)
        lookup={k:i for i,k in enumerate(owners)}
        roles=[]
        for row,index in enumerate(selected):
            for side,k in enumerate(evaluated.owners[index].tolist()):
                if k in lookup:roles.append((row,lookup[k],index,side))
        groups.append((torch.tensor(selected,device=device,dtype=torch.long),ids,tuple(roles)))
    if signature is not None:
        scene._mimic_group_topology=(signature,groups,sources)
    return groups

class Tangent:
    """Sparse P=I-B J^T (J B J^T)^+ J, with <=6 rows/group.

    B defaults to the physical inverse mass. A frozen local response
    inverse can be supplied to preserve its conditioning in primal/dual
    response sandwiches; geometry retraction retains the mass default.
    """
    def __init__(self,scene,state,evaluated=None,*,inverse_blocks=None):
        self.scene=scene;self.rows=rows(scene,state) if evaluated is None else evaluated
        self.groups=[];self.valid=self.rows.valid.all(-1)
        topology=group_topology(scene,self.rows,state.x.device,cache=evaluated is None)
        with torch.no_grad():
            for selected,ids,roles in topology:
                jac=state.x.new_zeros(len(state.x),len(selected),len(ids),6)
                # Sum repeated owners BEFORE computing J M^-1 J^T.
                for row,owner,index,side in roles:
                    jac[:,row,owner]+=self.rows.jacobian[:,index,side]
                if inverse_blocks is None:
                    inverse=state.x.new_zeros(len(state.x),len(ids),6,6)
                    inverse[...,:3,:3]=torch.eye(3,device=state.x.device,dtype=state.x.dtype)/scene.mass[ids,None,None]
                    rotation=so3.matrix(state.q[:,ids]);inverse[...,3:,3:]=rotation@scene.inverse_inertia[ids]@rotation.transpose(-1,-2)
                else:
                    # The primal AND dual projector must use the same
                    # inverse as the response. Mass projection can amplify
                    # the frozen elastic/contact curvature by orders of
                    # magnitude when the local inverse is H_diag^-1.
                    if inverse_blocks.shape!=(*state.x.shape[:2],6,6):
                        raise ValueError('Complete local inverse blocks required for a metric tangent')
                    inverse=inverse_blocks[:,ids].detach()
                    scale=inverse.abs().amax((-1,-2)).clamp_min(1e-30)
                    self.valid&=torch.isfinite(inverse).all((-1,-2,-3))
                    self.valid&=((inverse-inverse.transpose(-1,-2)).abs().amax((-1,-2))<=1e-10*scale).all(-1)
                    _,info=torch.linalg.cholesky_ex((inverse+inverse.transpose(-1,-2))*.5,check_errors=False)
                    self.valid&=(info==0).all(-1)
                lift=torch.einsum('bnij,brnj->brni',inverse,jac)
                gram=torch.einsum('brni,bsni->brs',jac,lift)
                if len(selected)==1:
                    # A scalar Gram has its entry as the sole eigenvalue and
                    # [1] as its eigenvector, including the zero-rank case.
                    eigen=gram[...,0];frame=torch.ones_like(gram)
                else:eigen,frame=torch.linalg.eigh(gram)
                keep=eigen>eigen.abs().amax(-1,keepdim=True).clamp_min(1e-30)*1e-12
                inverse_gram=(frame*torch.where(keep,eigen.clamp_min(1e-300).reciprocal(),0.)[:,None])@frame.transpose(-1,-2)
                self.valid&=torch.isfinite(inverse_gram).flatten(1).all(-1)&(eigen>=-1e-12*eigen.abs().amax(-1,keepdim=True).clamp_min(1.)).all(-1)
                self.groups.append((selected,ids,jac,lift,inverse_gram,gram))

    def primal(self,value):
        result=value
        for _,ids,jac,lift,inverse,gram in self.groups:
            coordinate=torch.einsum('brni,bni->br',jac,value[:,ids])
            delta=torch.einsum('brni,br->bni',lift,(inverse@coordinate[...,None])[...,0])
            result=result.index_add(1,ids,-delta)
        return result

    def dual(self,value):
        result=value
        for _,ids,jac,lift,inverse,gram in self.groups:
            coordinate=torch.einsum('brni,bni->br',lift,value[:,ids])
            delta=torch.einsum('brni,br->bni',jac,(inverse@coordinate[...,None])[...,0])
            result=result.index_add(1,ids,-delta)
        return result

    def correction(self):
        result=self.rows.value.new_zeros(len(self.rows.value),self.scene.n,6)
        valid=self.valid.clone()
        for selected,ids,jac,lift,inverse,gram in self.groups:
            value=self.rows.value[:,selected]
            multiplier=(inverse@value[...,None])[...,0]
            unresolved=value-(gram@multiplier[...,None])[...,0]
            valid&=(unresolved.abs()<=1e-9).all(-1)
            result=result.index_add(1,ids,-torch.einsum('brni,br->bni',lift,multiplier))
        return result,valid


def retract(scene,state,*,passes=2):
    """Exactly two bounded Newton retractions; never iterate to convergence."""
    current=state;valid=torch.ones(len(state.x),device=state.x.device,dtype=torch.bool)
    travel=state.x.new_zeros(len(state.x),scene.n)
    for _ in range(passes):
        operator=Tangent(scene,current);delta,ok=operator.correction()
        extent=torch.cat((delta[...,:3]/scene.length,delta[...,3:]),-1).norm(dim=-1)
        valid&=ok&(extent<=.1).all(-1);travel=travel+delta[...,3:].norm(dim=-1)
        current=State(current.x+delta[...,:3],so3.normalize(so3.multiply(so3.exp(delta[...,3:]),current.q)),state.v,state.omega,state.history)
    final=rows(scene,current)
    valid&=final.valid.all(-1)&(final.value.abs()<=1e-9).all(-1)&(travel<=.2).all(-1)
    return current,valid
