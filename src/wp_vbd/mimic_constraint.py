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
    from .inference_kinematics import memoize,spec_tensors
    return memoize('mimic_coordinates',scene,
        (state.x,state.q,scene.body_com,*spec_tensors(scene.joint_spec),*spec_tensors(scene.mimic_spec)),
        (scene.length,),lambda:_rows_uncached(scene,state))


def _rows_uncached(scene,state):
    s=scene.mimic_spec;j=scene.joint_spec;pairs=s.pairs
    ids=j.owners
    position=state.x[:,ids];rotation=state.q[:,ids]
    function=_rows_tensor
    if getattr(scene,'compiled_kinematics',False) and not torch.is_grad_enabled() and state.x.is_cuda:
        from .compiled_contact import evaluate as compiled
        function=lambda *args:compiled(_rows_tensor,*args)
    owners,enabled=relation(scene)
    values=function(position,rotation,scene.body_com[ids],j.frames,j.angular_axes,j.linear_axes,
                    pairs,s.linear,s.coefficients,scene.length,owners,enabled)
    return SimpleNamespace(**dict(zip(('value','jacobian','owners','enabled','valid','coordinates'),values)))


def _rows_tensor(position,rotation,body_com,frames,angular_axes,linear_axes,pairs,linear_flags,coefficients,length,owners,enabled):
    linear,jlinear,_,parent=anchor_coordinates(position,rotation,body_com,frames)
    child=so3.multiply(rotation[...,1,:],frames[...,1,3:])
    relative=so3.normalize(so3.multiply(so3.conjugate(parent),child))
    axis=angular_axes[:,0];v=relative[...,:3];w=relative[...,3];a=(v*axis).sum(-1)
    denominator=w.square()+a.square()
    angle=2*torch.atan2(a,w)
    angle=torch.atan2(angle.sin(),angle.cos())
    covector=(w.square()[...,None]*axis+w[...,None]*torch.linalg.cross(v,axis.expand_as(v))+a[...,None]*v)/denominator.clamp_min(1e-20)[...,None]
    covector=(covector[...,None,:]@so3.matrix(parent).transpose(-1,-2))[...,0,:]
    zero=torch.zeros_like(covector)
    angular_j=torch.stack((torch.cat((zero,-covector),-1),torch.cat((zero,covector),-1)),-2)
    linear_value=(linear*linear_axes[:,0]).sum(-1)
    linear_j=torch.einsum('ji,bjisk->bjsk',linear_axes[:,0],jlinear)
    values=torch.where(linear_flags[None],linear_value[:,pairs],angle[:,pairs])
    jac=torch.where(linear_flags[None,:,:,None,None],linear_j[:,pairs],angular_j[:,pairs])
    value=values[...,0]-coefficients[:,0]-coefficients[:,1]*values[...,1]
    value=torch.where(linear_flags[None,:,0],value,torch.atan2(value.sin(),value.cos()))
    jac=torch.cat((jac[:,:,0],-coefficients[None,:,1,None,None]*jac[:,:,1]),2)
    scale=torch.where(linear_flags[:,0],1/length,1.)
    # Principal twist has no chart at a 180-degree swing about a transverse axis.
    good=(denominator[:,pairs]>1e-10)|linear_flags[None]
    valid=(good[...,0]&(good[...,1]|(coefficients[:,1]==0)))|~enabled
    jac=jac*enabled[None,:,None,None]*scale[None,:,None,None]
    return value*scale*enabled,jac,owners,enabled,valid,values


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
    if evaluated is None:
        owners,enabled=relation(scene)
        evaluated=SimpleNamespace(owners=owners,enabled=enabled)
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


def _group_jacobian(x,selected,ids,roles,row_jacobian):
    jac=x.new_zeros(len(x),len(selected),len(ids),6)
    # Preserve summation order when several roles name the same real owner.
    for row,owner,index,side in roles:jac[:,row,owner]+=row_jacobian[:,index,side]
    return jac


def _mass_inverse(scene,x,q,ids):
    inverse=x.new_zeros(len(x),len(ids),6,6)
    inverse[...,:3,:3]=torch.eye(3,device=x.device,dtype=x.dtype)/scene.mass[ids,None,None]
    rotation=so3.matrix(q[:,ids])
    inverse[...,3:,3:]=rotation@scene.inverse_inertia[ids]@rotation.transpose(-1,-2)
    return inverse


def _factor_group(jac,inverse):
    lift=torch.einsum('bnij,brnj->brni',inverse,jac)
    gram=torch.einsum('brni,bsni->brs',jac,lift)
    if jac.shape[1]==1:
        eigen=gram[...,0];frame=torch.ones_like(gram)
    else:eigen,frame=torch.linalg.eigh(gram)
    keep=eigen>eigen.abs().amax(-1,keepdim=True).clamp_min(1e-30)*1e-12
    inverse_gram=(frame*torch.where(keep,eigen.clamp_min(1e-300).reciprocal(),0.)[:,None])@frame.transpose(-1,-2)
    valid=torch.isfinite(inverse_gram).flatten(1).all(-1)&(eigen>=-1e-12*eigen.abs().amax(-1,keepdim=True).clamp_min(1.)).all(-1)
    return lift,inverse_gram,gram,valid


def _tangent_group_tensor(scene,x,q,selected,ids,roles,row_jacobian,inverse_blocks):
    jac=_group_jacobian(x,selected,ids,roles,row_jacobian)
    valid=torch.ones(len(x),device=x.device,dtype=torch.bool)
    if inverse_blocks is None:inverse=_mass_inverse(scene,x,q,ids)
    else:
        inverse=inverse_blocks[:,ids].detach()
        scale=inverse.abs().amax((-1,-2)).clamp_min(1e-30)
        valid=valid&torch.isfinite(inverse).all((-1,-2,-3))
        valid=valid&((inverse-inverse.transpose(-1,-2)).abs().amax((-1,-2))<=1e-10*scale).all(-1)
        _,info=torch.linalg.cholesky_ex((inverse+inverse.transpose(-1,-2))*.5,check_errors=False)
        valid=valid&(info==0).all(-1)
    lift,inverse_gram,gram,good=_factor_group(jac,inverse)
    return jac,lift,inverse_gram,gram,valid&good


def _primal_tensor(value,groups):
    result=value
    for _,ids,jac,lift,inverse,gram in groups:
        coordinate=torch.einsum('brni,bni->br',jac,value[:,ids])
        delta=torch.einsum('brni,br->bni',lift,(inverse@coordinate[...,None])[...,0])
        result=result.index_add(1,ids,-delta)
    return result


def _dual_tensor(value,groups):
    result=value
    for _,ids,jac,lift,inverse,gram in groups:
        coordinate=torch.einsum('brni,bni->br',lift,value[:,ids])
        delta=torch.einsum('brni,br->bni',jac,(inverse@coordinate[...,None])[...,0])
        result=result.index_add(1,ids,-delta)
    return result


def _correction_tensor(value,valid,groups,n):
    result=value.new_zeros(len(value),n,6);valid=valid.clone()
    for selected,ids,jac,lift,inverse,gram in groups:
        v=value[:,selected]
        multiplier=(inverse@v[...,None])[...,0]
        unresolved=v-(gram@multiplier[...,None])[...,0]
        valid=valid&(unresolved.abs()<=1e-9).all(-1)
        result=result.index_add(1,ids,-torch.einsum('brni,br->bni',lift,multiplier))
    return result,valid

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
        if topology and inverse_blocks is not None and inverse_blocks.shape!=(*state.x.shape[:2],6,6):
            raise ValueError('Complete local inverse blocks required for a metric tangent')
        self.compiled=bool(getattr(scene,'compiled_candidate_kernels',False)) and not torch.is_grad_enabled() and state.x.is_cuda
        with torch.no_grad():
            for selected,ids,roles in topology:
                if self.compiled:
                    from .compiled_contact import evaluate as compiled
                    jac,lift,inverse_gram,gram,valid=compiled(_tangent_group_tensor,scene,state.x,state.q,
                        selected,ids,roles,self.rows.jacobian,inverse_blocks)
                else:
                    jac,lift,inverse_gram,gram,valid=_tangent_group_tensor(scene,state.x,state.q,
                        selected,ids,roles,self.rows.jacobian,inverse_blocks)
                self.valid&=valid
                self.groups.append((selected,ids,jac,lift,inverse_gram,gram))

    def primal(self,value):
        if self.compiled and not torch.is_grad_enabled():
            from .compiled_contact import evaluate as compiled
            return compiled(_primal_tensor,value,self.groups)
        return _primal_tensor(value,self.groups)

    def dual(self,value):
        if self.compiled and not torch.is_grad_enabled():
            from .compiled_contact import evaluate as compiled
            return compiled(_dual_tensor,value,self.groups)
        return _dual_tensor(value,self.groups)

    def correction(self):
        if self.compiled and not torch.is_grad_enabled():
            from .compiled_contact import evaluate as compiled
            return compiled(_correction_tensor,self.rows.value,self.valid,self.groups,self.scene.n)
        return _correction_tensor(self.rows.value,self.valid,self.groups,self.scene.n)


def retract(scene,state,*,passes=2):
    """Exactly two bounded Newton retractions; never iterate to convergence."""
    if (getattr(scene,'compiled_candidate_kernels',False)
            and not torch.is_grad_enabled() and state.x.is_cuda):
        from .compiled_contact import evaluate as compiled
        topology=group_topology(scene,None,state.x.device)
        x,q,valid=compiled(_retract_tensor,scene,state.x,state.q,topology,passes)
        return State(x,q,state.v,state.omega,state.history),valid
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


def _retract_tensor(scene,x,q,topology,passes):
    """The same two mass projections, including rank and travel checks."""
    valid=torch.ones(len(x),device=x.device,dtype=torch.bool)
    travel=x.new_zeros(len(x),scene.n)
    j=scene.joint_spec;s=scene.mimic_spec;owners,enabled=relation(scene)
    def evaluate_rows(x,q):
        ids=j.owners
        return _rows_tensor(x[:,ids],q[:,ids],scene.body_com[ids],j.frames,
            j.angular_axes,j.linear_axes,s.pairs,s.linear,s.coefficients,scene.length,owners,enabled)
    for _ in range(passes):
        value,row_jacobian,_,_,row_valid,_=evaluate_rows(x,q)
        delta=x.new_zeros(len(x),scene.n,6);ok=row_valid.all(-1)
        for selected,ids,roles in topology:
            jac=_group_jacobian(x,selected,ids,roles,row_jacobian)
            lift,inverse,gram,good=_factor_group(jac,_mass_inverse(scene,x,q,ids))
            v=value[:,selected];multiplier=(inverse@v[...,None])[...,0]
            unresolved=v-(gram@multiplier[...,None])[...,0]
            ok=ok&good&(unresolved.abs()<=1e-9).all(-1)
            delta=delta.index_add(1,ids,-torch.einsum('brni,br->bni',lift,multiplier))
        extent=torch.cat((delta[...,:3]/scene.length,delta[...,3:]),-1).norm(dim=-1)
        valid=valid&ok&(extent<=.1).all(-1)
        travel=travel+delta[...,3:].norm(dim=-1)
        x=x+delta[...,:3]
        q=so3.normalize(so3.multiply(so3.exp(delta[...,3:]),q))
    value,_,_,_,row_valid,_=evaluate_rows(x,q)
    valid=valid&row_valid.all(-1)&(value.abs()<=1e-9).all(-1)&(travel<=.2).all(-1)
    return x,q,valid
