"""F64 closest particle features and whole-feature linear-path certificates.

This module does not generate a complete candidate set. Its path certificate
applies only to the explicitly supplied primitive pairs.
"""
from dataclasses import dataclass

import torch

from .self_contact_rows import safe_norm


@dataclass
class ClosestGeometry:
    kind: int
    reference: torch.Tensor
    weights: torch.Tensor
    distance: torch.Tensor
    normal: torch.Tensor
    separating_axis: torch.Tensor
    separation_lower_bound: torch.Tensor


def validate(points):
    if points.dtype!=torch.float64 or points.shape[-2:]!=(4,3) or not bool(torch.isfinite(points).all()):
        raise ValueError('Four finite F64 particle positions required')


def choose(values,index):
    return torch.gather(values,-2,index[...,None,None].expand(*index.shape,1,values.shape[-1])).squeeze(-2)


def splitting(points,kind):
    return (points[...,:3,:],points[...,3:,:]) if kind==9 else (points[...,:2,:],points[...,2:,:])


def checked_result(kind,points,weights,distance,normal,axis,lower):
    if any(not bool(torch.isfinite(value).all()) for value in (weights,distance,normal,axis,lower)):
        raise ValueError('Self-contact geometry exceeded its finite arithmetic domain')
    return ClosestGeometry(kind,points.detach().clone(),weights,distance,normal,axis,lower)


def certificate(points,axes,kind):
    length=safe_norm(axes)
    unit=axes/length.clamp_min(1e-300)[...,None]
    first,second=splitting(points,kind)
    pa=torch.einsum('...ai,...vi->...av',unit,first)
    pb=torch.einsum('...ai,...vi->...av',unit,second)
    forward=pb.amin(-1)-pa.amax(-1);reverse=pa.amin(-1)-pb.amax(-1)
    sign=torch.where(forward>=reverse,1.,-1.)
    gap=torch.maximum(forward,reverse);index=gap.argmax(-1)
    axis=choose(unit*sign[...,None],index)
    best=torch.gather(gap,-1,index[...,None])[...,0]
    # Reserve a floating-point slack for dot products and normalization.
    # A projection support gap is a lower bound even when a closest-feature
    # calculation is ill-conditioned; approximate distance is never certified.
    slack=64*torch.finfo(points.dtype).eps*points.abs().amax((-1,-2)).clamp_min(1.)
    return axis,(best-slack).clamp_min(0.)


def checked_distance(distance):
    if not bool(torch.isfinite(distance).all()):
        raise ValueError('Self-contact distance exceeded its finite arithmetic domain')
    return distance


def vertex_face(points,*,distance_only=False):
    validate(points);triangle=points[...,:3,:];p=points[...,3,:]
    ab=triangle[...,1,:]-triangle[...,0,:];ac=triangle[...,2,:]-triangle[...,0,:];ap=p-triangle[...,0,:]
    dot=lambda x,y:(x*y).sum(-1)
    area=torch.linalg.cross(ab,ac,dim=-1);det=dot(area,area)
    safe=torch.where(det>0,det,torch.ones_like(det))
    # Cross products avoid subtracting nearly equal Gram products for thin
    # triangles. The point's orthogonal projection has these barycentrics.
    v=dot(torch.linalg.cross(ap,ac,dim=-1),area)/safe
    w=dot(torch.linalg.cross(ab,ap,dim=-1),area)/safe
    inside=(det>0)&(v>=0)&(w>=0)&(v+w<=1)&torch.isfinite(v)&torch.isfinite(w)
    face=torch.stack((1-v-w,v,w),-1)
    fallback=torch.zeros_like(face);fallback[...,0]=1.
    face=torch.where(inside[...,None],face,fallback)
    face=face.clamp_min(0.);face=face/face.sum(-1,keepdim=True)
    starts=torch.tensor([0,1,2],device=points.device);ends=torch.tensor([1,2,0],device=points.device)
    edge=triangle[...,ends,:]-triangle[...,starts,:]
    parameter=(((p[...,None,:]-triangle[...,starts,:])*edge).sum(-1)/edge.square().sum(-1).clamp_min(1e-300)).clamp(0,1)
    eye=torch.eye(3,device=points.device,dtype=points.dtype)
    boundary=(1-parameter[...,None])*eye[starts]+parameter[...,None]*eye[ends]
    candidates=torch.cat((face[...,None,:],boundary),-2)
    closest=torch.einsum('...ki,...ij->...kj',candidates,triangle)
    delta=p[...,None,:]-closest;square=delta.square().sum(-1)
    square=torch.cat((torch.where(inside,square[...,0],torch.inf)[...,None],square[...,1:]),-1)
    index=square.argmin(-1);bary=choose(candidates,index);difference=choose(delta,index)
    distance=safe_norm(difference)
    if distance_only:return checked_distance(distance)
    normal=difference/distance.clamp_min(1e-300)[...,None]
    zero=torch.zeros_like(bary[...,:1]);first=torch.cat((bary,zero),-1)
    second=torch.cat((torch.zeros_like(bary),torch.ones_like(zero)),-1)
    axes=torch.cat((delta,torch.linalg.cross(ab,ac,dim=-1)[...,None,:]),-2)
    axis,lower=certificate(points,axes,9)
    return checked_result(9,points,torch.stack((first,second),-2),distance,normal,axis,lower)


def edge_edge(points,*,distance_only=False):
    validate(points);a=points[...,0,:];b=points[...,2,:]
    u=points[...,1,:]-a;v=points[...,3,:]-b;offset=a-b
    dot=lambda x,y:(x*y).sum(-1)
    aa=dot(u,u);bb=dot(u,v);cc=dot(v,v);d=dot(u,offset);e=dot(v,offset)
    cross=torch.linalg.cross(u,v,dim=-1);det=dot(cross,cross)
    safe=torch.where(det>0,det,torch.ones_like(det))
    s=dot(torch.linalg.cross(-offset,v,dim=-1),cross)/safe
    t=dot(torch.linalg.cross(-offset,u,dim=-1),cross)/safe
    inside=(det>0)&(s>=0)&(s<=1)&(t>=0)&(t<=1)&torch.isfinite(s)&torch.isfinite(t)
    s=torch.where(inside,s,0.);t=torch.where(inside,t,0.)
    zero=torch.zeros_like(s);one=torch.ones_like(s)
    sc=torch.stack((s,zero,one,(-d/aa.clamp_min(1e-300)).clamp(0,1),((bb-d)/aa.clamp_min(1e-300)).clamp(0,1)),-1)
    tc=torch.stack((t,(e/cc.clamp_min(1e-300)).clamp(0,1),((e+bb)/cc.clamp_min(1e-300)).clamp(0,1),zero,one),-1)
    first=a[...,None,:]+sc[...,None]*u[...,None,:];second=b[...,None,:]+tc[...,None]*v[...,None,:]
    delta=second-first;square=delta.square().sum(-1)
    square=torch.cat((torch.where(inside,square[...,0],torch.inf)[...,None],square[...,1:]),-1)
    index=square.argmin(-1)
    difference=choose(delta,index);distance=safe_norm(difference)
    if distance_only:return checked_distance(distance)
    s=torch.gather(sc,-1,index[...,None])[...,0];t=torch.gather(tc,-1,index[...,None])[...,0]
    normal=difference/distance.clamp_min(1e-300)[...,None]
    first=torch.stack((1-s,s,zero,zero),-1);second=torch.stack((zero,zero,1-t,t),-1)
    axes=torch.cat((delta,torch.linalg.cross(u,v,dim=-1)[...,None,:]),-2)
    axis,lower=certificate(points,axes,10)
    return checked_result(10,points,torch.stack((first,second),-2),distance,normal,axis,lower)


def path_feasible(geometry,candidate,*,relaxation=.9):
    validate(candidate)
    if candidate.shape!=geometry.reference.shape or not 0<relaxation<1:raise ValueError('Matched primitive path and strict relaxation required')
    first,second=splitting(candidate,geometry.kind);axis=geometry.separating_axis
    a=torch.einsum('...vi,...i->...v',first,axis);b=torch.einsum('...vi,...i->...v',second,axis)
    magnitude=torch.maximum(candidate.abs().amax((-1,-2)),geometry.reference.abs().amax((-1,-2))).clamp_min(1.)
    slack=64*torch.finfo(candidate.dtype).eps*magnitude
    # All vertices must stay on opposite sides in this direction. A positive
    # gap at both endpoints proves separation throughout the linear path.
    plane=b.amin(-1)-a.amax(-1)>slack
    travel=safe_norm(candidate-geometry.reference)
    total=travel[...,:3].amax(-1)+travel[...,3] if geometry.kind==9 else travel[...,:2].amax(-1)+travel[...,2:].amax(-1)
    bounded=total<=relaxation*geometry.separation_lower_bound
    return (geometry.separation_lower_bound>slack)&(plane|bounded)


def refined_path_feasible(geometry,candidate,depth=0):
    """Prove one linear path by a finite union of certified subintervals.

    This only refines the supplied pairs; the caller still owns complete
    candidate coverage and the motion envelope. Unproved intervals reject.
    Existing F64 support-gap slack also dominates convex-midpoint roundoff.
    """
    if type(depth) is not int or not 0<=depth<=8:raise ValueError('Path subdivision depth in [0,8] required')
    if depth and geometry.reference.ndim!=3:raise ValueError('One primitive-row batch required for path refinement')
    good=path_feasible(geometry,candidate)
    if not depth:return good
    pending=torch.where(~good)[0]
    if not len(pending):return good
    reference=geometry.reference[pending];end=candidate[pending];middle=(reference+end)/2
    compute=vertex_face if geometry.kind==9 else edge_edge
    end_geometry=compute(end);middle_geometry=compute(middle)
    # A zero lower bound remains uncertified, even if an approximate closest
    # distance happens to be positive.
    valid=(end_geometry.separation_lower_bound>0)&(middle_geometry.separation_lower_bound>0)
    selected=torch.where(valid)[0]
    if not len(selected):return good
    reference=reference[selected];end=end[selected];middle=middle[selected]
    left=refined_path_feasible(compute(reference),middle,depth-1)
    indices=torch.where(left)[0]
    if len(indices):
        right=refined_path_feasible(compute(middle[indices]),end[indices],depth-1)
        good=good.index_copy(0,pending[selected[indices]],right)
    return good
