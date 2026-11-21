"""Owned collision snapshots and COM geometry; no dynamics-solver imports."""
from dataclasses import dataclass, field
import torch
from . import math3d as so3
from .native_array import native_tensor


@dataclass
class RigidContacts:
    owners:torch.Tensor
    shapes:torch.Tensor
    points:torch.Tensor
    offsets:torch.Tensor
    normal:torch.Tensor
    margins:torch.Tensor
    surface_velocity:torch.Tensor
    material:torch.Tensor
    damping:torch.Tensor
    mu:torch.Tensor
    physical_margin:torch.Tensor
    source_slots:torch.Tensor
    point_ids:torch.Tensor
    generation:int
    source_count:int
    capacity:int
    # Optional history metadata must not occupy positional arguments before
    # required row fields added by derived packets such as MixedContacts.
    matching_mode:str=field(default='disabled',kw_only=True)
    match_index:torch.Tensor|None=field(default=None,kw_only=True)
    match_generation:int=field(default=-1,kw_only=True)
    stream_id:int=field(default=0,kw_only=True)

    @classmethod
    def snapshot(cls,model,contacts,wp,*,dtype=torch.float64):
        count=int(wp.to_torch(contacts.rigid_contact_count)[0]);capacity=len(contacts.rigid_contact_shape0)
        if count<0 or count>capacity:raise ValueError(f'Rigid collision overflow: {count}/{capacity}')
        for name in ('_reduction_overflow','_soft_contact_source_overflow'):
            value=getattr(contacts,name,None)
            if value is not None and bool(wp.to_torch(value).any()):raise ValueError('Collision reduction/source overflow')
        pair=torch.stack((native_tensor(wp,contacts.rigid_contact_shape0)[:count],native_tensor(wp,contacts.rigid_contact_shape1)[:count]),-1).long().clone()
        if bool(((pair>=model.shape_count)|(pair < -1)).any()):raise ValueError('Invalid collision shape index')
        slots=torch.where((pair>=0).all(-1))[0];pair=pair[slots]
        def take(name):return native_tensor(wp,getattr(contacts,name))[:count][slots].clone().to(dtype)
        owner=native_tensor(wp,model.shape_body).long()[pair].clone()
        if bool((owner < -1).any()):raise ValueError('Invalid collision body owner')
        owner=torch.where(owner<0,model.body_count,owner)
        if bool((owner>model.body_count).any()):raise ValueError('Invalid collision body owner')
        def material(name):return native_tensor(wp,getattr(model,name))[pair].clone().to(dtype)
        normal=take('rigid_contact_normal')
        values=[take('rigid_contact_point0'),take('rigid_contact_point1'),take('rigid_contact_offset0'),take('rigid_contact_offset1'),normal]
        if not all(bool(torch.isfinite(value).all()) for value in values):raise ValueError('Nonfinite collision geometry')
        if not bool(((normal.square().sum(-1)-1).abs()<1e-5).all()):raise ValueError('Invalid collision normal')
        # Source normals are float32. Re-establish unit length after promotion
        # so a parallel displacement cannot create spurious tangential C0.
        normal=normal/normal.norm(dim=-1,keepdim=True)
        values[-1]=normal
        velocity=take('rigid_contact_surface_velocity') if contacts.rigid_contact_surface_velocity is not None else torch.zeros_like(normal)
        ke=material('shape_material_ke').mean(-1);kd=material('shape_material_kd').mean(-1);mu=material('shape_material_mu').prod(-1).sqrt()
        if any(not bool(torch.isfinite(value).all()) or bool((value<0).any()) for value in (ke,kd,mu)):raise ValueError('Invalid contact materials')
        matching=getattr(contacts,'contact_matching_mode','disabled')
        match_index=(native_tensor(wp,contacts.rigid_contact_match_index)[:count][slots].long().clone()
                     if matching!='disabled' else None)
        match_generation=int(native_tensor(wp,contacts.rigid_contact_match_generation)[0]) if matching!='disabled' else -1
        return cls(owner,pair,torch.stack(values[:2],1),torch.stack(values[2:4],1),normal,
            torch.stack((take('rigid_contact_margin0'),take('rigid_contact_margin1')),-1),velocity,ke,kd,mu,
            material('shape_margin').sum(-1),slots,native_tensor(wp,contacts.rigid_contact_point_id)[:count][slots].long().clone(),
            int(wp.to_torch(contacts.contact_generation)[0]),count,capacity,
            matching_mode=matching,match_index=match_index,match_generation=match_generation,stream_id=id(contacts))

    def geometry(self,scene,state):
        from .inference_kinematics import memoize
        return memoize('rigid_geometry',self,
            (state.x,state.q,scene.body_com,self.owners,self.points,self.offsets,self.normal,self.margins),(),
            lambda:self._geometry_uncached(scene,state))

    def _geometry_uncached(self,scene,state):
        origin=state.x-so3.rotate(state.q,scene.body_com)
        x=origin[:,self.owners];q=state.q[:,self.owners]
        skeleton=x+so3.rotate(q,self.points)
        surface=x+so3.rotate(q,self.points+self.offsets)
        depth=self.margins.sum(-1)[None]-((skeleton[:,:,1]-skeleton[:,:,0])*self.normal).sum(-1)
        return x,q,skeleton,surface,depth


def supports(scene,start,contacts,dt):
    """Inertial supports, with explicit optional structural conditioning."""
    ids=contacts.owners;q=start.q[:,ids];n=contacts.normal[None,:,None].expand(len(start.x),-1,2,-1)
    inverse=scene.inverse_inertia[ids];mobile=scene.movable[ids]
    structural=bool(getattr(scene,'contact_structural_support',False))
    if structural:inverse=torch.where(mobile[...,None,None],inverse,0.)
    inv_mass=torch.where(mobile,scene.mass[ids].clamp_min(1e-30).reciprocal(),0.)
    normal_local=so3.rotate(so3.conjugate(q),n)
    lever=contacts.points-scene.body_com[ids]
    cross=torch.linalg.cross(lever[None],normal_local)
    w=inv_mass[None]+(cross*(inverse[None]@cross[...,None])[...,0]).sum(-1)
    total=w.sum(-1)
    rho_n=torch.where((contacts.material[None]>0)&(total>0),1/(dt**2*total.clamp_min(1e-30)),0.)
    rotation=so3.matrix(q);world_inverse=rotation@inverse[None]@rotation.transpose(-1,-2)
    r=so3.rotate(q,contacts.points+contacts.offsets-scene.body_com[ids])
    from .response_operator import skew
    cross_r=skew(r)
    eye=torch.eye(3,device=r.device,dtype=r.dtype)
    delassus=(inv_mass[None,:,:,None,None]*eye-cross_r@world_inverse@cross_r).sum(2)
    nn=contacts.normal;projector=eye-nn[:,:,None]*nn[:,None,:]
    tangent=projector[None]@delassus@projector[None]
    maximum=torch.linalg.eigvalsh(tangent)[...,-1]
    rho_t=torch.where(maximum>0,1/(dt**2*maximum.clamp_min(1e-30)),0.)
    if structural:
        from .contact_conditioning import conditioned_supports
        return conditioned_supports(scene,contacts,w,rho_t,dt)
    return rho_n,rho_t
