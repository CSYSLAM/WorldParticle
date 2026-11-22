"""Learned low-cost material response directions in a rigid-free basis."""
import math

import torch
from torch import nn

from .coarse_response import rigid_component_modes
from .protected_response import metric_action


def _pseudoinverse_solve(matrix,rhs):
    eigen,frame=torch.linalg.eigh((matrix+matrix.transpose(-1,-2))*.5)
    threshold=eigen.abs().amax(-1,keepdim=True).clamp_min(1e-30)*1e-10
    if not bool(torch.isfinite(eigen).all()) or bool((eigen < -100*threshold).any()):
        raise ValueError('Elastic shape response requires finite positive semidefinite curvature')
    good=eigen>threshold
    coefficients=frame.transpose(-1,-2)@rhs[...,None]
    return (frame@(coefficients*torch.where(good,eigen.clamp_min(1e-300).reciprocal(),0.)[...,None]))[...,0]


def polynomial_basis(points,mass):
    """Six affine strains plus eighteen quadratic displacements, rigid-free."""
    center=(points*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
    radius=((points-center).square().sum(-1)*mass).sum(-1).div(mass.sum()).sqrt().clamp_min(1e-8)
    coordinate=(points-center)/radius[:,None,None]
    identity=torch.eye(3,dtype=points.dtype,device=points.device)
    columns=[]
    for a,b in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
        value=coordinate[...,a,None]*identity[b]
        if a!=b:value=value+coordinate[...,b,None]*identity[a]
        columns.append(value)
    for a,b in ((0,0),(1,1),(2,2),(0,1),(0,2),(1,2)):
        scalar=coordinate[...,a]*coordinate[...,b]
        scalar=scalar-(scalar*mass).sum(-1,keepdim=True)/mass.sum()
        for axis in range(3):columns.append(scalar[...,None]*identity[axis])
    basis=torch.stack(columns,-1)
    dof=torch.ones(len(mass),6,dtype=torch.bool,device=mass.device)
    rigid=rigid_component_modes(points,center,dof)[...,:3,:]
    gram=torch.einsum('bnip,n,bniq->bpq',rigid,mass,rigid)
    cross=torch.einsum('bnip,n,bniq->bpq',rigid,mass,basis)
    eigen,frame=torch.linalg.eigh((gram+gram.transpose(-1,-2))*.5)
    threshold=eigen.abs().amax(-1,keepdim=True).clamp_min(1e-30)*1e-10
    if not bool(torch.isfinite(eigen).all()) or bool((eigen < -100*threshold).any()):
        raise ValueError('Elastic shape response requires finite positive semidefinite rigid metric')
    projected=frame.transpose(-1,-2)@cross
    solution=frame@(projected*torch.where(eigen>threshold,eigen.clamp_min(1e-300).reciprocal(),0.)[...,None])
    basis=basis-torch.einsum('bnip,bpq->bniq',rigid,solution)
    norm=torch.einsum('bnip,n,bnip->bp',basis,mass,basis).div(mass.sum()).sqrt()
    return basis/norm[:,None,None].clamp_min(1e-12)


def material_basis(points,rest_points,mass):
    """Express coefficients in a material frame carried by two rest-selected rays."""
    rest=rest_points-(rest_points*mass[:,None]).sum(0)/mass.sum()
    first=rest.square().sum(-1).argmax().reshape(1)
    second=torch.linalg.cross(rest.index_select(0,first).expand_as(rest),rest).square().sum(-1).argmax().reshape(1)
    centered=points-(points*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
    u=centered.index_select(1,first)[:,0]
    u=u/u.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    second_ray=centered.index_select(1,second)[:,0]
    v=second_ray-(second_ray*u).sum(-1,keepdim=True)*u
    v=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
    frame=torch.stack((u,v,torch.linalg.cross(u,v)),-1)
    local=centered@frame
    basis=polynomial_basis(local,mass)
    return torch.einsum('bij,bnjp->bnip',frame,basis)


class ElasticShapeDirectionHead(nn.Module):
    """Predict a bounded combination of material modes before curvature work."""

    mode_count=24
    feature_count_v1=38
    feature_count_v2=86

    def __init__(self,width=64,maximum_fraction=.08,analytic_prior_fraction=.001,
                 learned_coefficient_scale=1.,feature_version=1,
                 initialization='zero',local_heads=0,local_fraction=.02):
        super().__init__()
        if (width<8 or not 0<maximum_fraction<=.25 or not 0<=analytic_prior_fraction<.5
                or not math.isfinite(learned_coefficient_scale) or learned_coefficient_scale<=0):
            raise ValueError('Invalid elastic direction head budget')
        if feature_version not in (1,2):raise ValueError('Unsupported elastic direction feature version')
        if type(local_heads) is not int or local_heads<0 or local_heads>4:
            raise ValueError('Elastic local head count must be an integer in [0, 4]')
        if not math.isfinite(local_fraction) or not 0<local_fraction<=maximum_fraction:
            raise ValueError('Local elastic response fraction must be finite and bounded by the main response cap')
        if initialization not in ('zero','normal'):raise ValueError('Unsupported elastic head initialization')
        self.feature_version=feature_version
        self.feature_count=self.feature_count_v1 if feature_version==1 else self.feature_count_v2
        self.maximum_fraction=maximum_fraction
        self.analytic_prior_fraction=analytic_prior_fraction
        self.learned_coefficient_scale=learned_coefficient_scale
        self.network=nn.Sequential(nn.Linear(self.feature_count,width),nn.SiLU(),
                                   nn.Linear(width,width),nn.SiLU(),nn.Linear(width,self.mode_count))
        if initialization=='zero':nn.init.zeros_(self.network[-1].weight)
        else:nn.init.normal_(self.network[-1].weight,std=.01)
        nn.init.zeros_(self.network[-1].bias)
        self.initialization=initialization
        self.local_heads=local_heads
        self.local_fraction=local_fraction
        self.local_network=None
        if local_heads:
            # Kept separate so old global-direction checkpoints load unchanged.
            self.local_network=nn.Sequential(nn.Linear(self.feature_count,width),nn.SiLU(),
                nn.Linear(width,width),nn.SiLU(),nn.Linear(width,local_heads*self.mode_count))
            nn.init.normal_(self.local_network[-1].weight,std=.01)
            nn.init.zeros_(self.local_network[-1].bias)

    @staticmethod
    def _localized_basis(points,rest_points,mass,basis,head):
        """Smooth material patches, projected off object translation/rotation."""
        rest=rest_points-(rest_points*mass[:,None]).sum(0)/mass.sum()
        first=int(rest.square().sum(-1).argmax())
        second=int(torch.linalg.cross(rest[first].expand_as(rest),rest).square().sum(-1).argmax())
        centered=points-(points*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
        u=centered[:,first]/centered[:,first].norm(dim=-1,keepdim=True).clamp_min(1e-12)
        v=centered[:,second]-(centered[:,second]*u).sum(-1,keepdim=True)*u
        v=v/v.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        local=centered@torch.stack((u,v,torch.linalg.cross(u,v)),-1)
        scale=(local.square().sum(-1).mean(-1,keepdim=True).sqrt().clamp_min(1e-8))
        coordinate=local[...,0]/scale
        center=(-.55 if head%2==0 else .55)
        envelope=torch.exp(-.5*((coordinate-center)/.55).square())
        candidate=basis*envelope[:, :, None, None]
        dof=torch.ones(len(mass),6,dtype=torch.bool,device=mass.device)
        center=(points*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
        rigid=rigid_component_modes(points,center,dof)[...,:3,:]
        gram=torch.einsum('bnip,n,bniq->bpq',rigid,mass,rigid)
        cross=torch.einsum('bnip,n,bniq->bpq',rigid,mass,candidate)
        eigen,frame=torch.linalg.eigh((gram+gram.transpose(-1,-2))*.5)
        threshold=eigen.abs().amax(-1,keepdim=True).clamp_min(1e-30)*1e-10
        projected=frame.transpose(-1,-2)@cross
        solution=frame@(projected*torch.where(eigen>threshold,eigen.clamp_min(1e-300).reciprocal(),0.)[...,None])
        candidate=candidate-torch.einsum('bnip,bpq->bniq',rigid,solution)
        norm=torch.einsum('bnip,n,bnip->bp',candidate,mass,candidate).div(mass.sum()).sqrt()
        return candidate/norm[:,None,None].clamp_min(1e-12)

    @staticmethod
    def _material_features(scene,ids):
        values=[]
        object_ids=torch.unique(scene.object_id[ids])
        mask=torch.isin(scene.object_id,object_ids)
        for name in ('tet_material','tri_material','bend_material'):
            material=getattr(scene,name,None)
            if material is None or not material.numel():
                values.extend((0.,0.))
                continue
            relation_name={'tet_material':'tets','tri_material':'triangles','bend_material':'bend_edges'}[name]
            rows=getattr(scene,relation_name,None)
            if rows is None or not len(rows):
                values.extend((0.,0.))
                continue
            selected=mask[rows].any(-1)
            active=material[selected]
            if not len(active):
                values.extend((0.,0.))
            else:
                values.append(float(torch.log1p(active[:,0].clamp_min(0)).mean()))
                second=active[:,1] if active.shape[1]>1 else active[:,0].new_zeros(len(active))
                values.append(float(torch.log1p(second.clamp_min(0)).mean()))
        return values

    def propose(self,scene,state,residual,dt,*,response_hints=(),return_details=False):
        if not math.isfinite(dt) or dt<=0:
            raise ValueError('A positive finite dt is required')
        result=residual.new_zeros(len(state.x),scene.n,6)
        column_parts=[]
        details=[]
        for object_id in torch.unique(scene.object_id[(scene.kind<2)&scene.movable]).tolist():
            ids=torch.where((scene.object_id==object_id)&(scene.kind<2)&scene.movable)[0]
            if len(ids)<8:
                continue
            with torch.no_grad():
                points=state.x[:,ids].detach().to(residual.dtype)
                basis=material_basis(points,scene.rest_x[ids].to(residual.dtype),scene.mass[ids].to(residual.dtype))
            work=torch.einsum('bncp,bnc->bp',basis,residual[:,ids,:3])
            hint_features=[]
            hints=tuple(response_hints)+(None,)*(2-len(response_hints))
            if len(hints)>2:raise ValueError('Elastic direction head accepts at most two local response hints')
            for hint in hints:
                if hint is None:
                    hint_features.append(work.new_zeros(work.shape))
                    continue
                modal=torch.einsum('bncp,bnc->bp',basis,hint[:,ids,:3].to(basis))
                modal_rms=modal.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
                hint_features.append(modal/modal_rms)
            mass=scene.mass[ids].sum().clamp_min(1e-30)
            radius=((points-points.mean(1,keepdim=True)).square().sum(-1).mean(-1).sqrt()
                    .clamp_min(scene.length*.01))
            dimensionless=work*dt**2/(mass*scene.length)
            rms=dimensionless.square().mean(-1,keepdim=True).sqrt().clamp_min(1e-8)
            material=self._material_features(scene,ids)
            material=torch.as_tensor(material,device=state.x.device,dtype=state.x.dtype).expand(len(state.x),-1)
            kind=torch.nn.functional.one_hot(scene.kind[ids[0]].long(),4).to(state.x.dtype).expand(len(state.x),-1)
            dt_feature=state.x.new_full((len(state.x),1),math.log(dt*60))
            radius_feature=(radius/scene.length).log()[:,None]
            mass_feature=torch.log1p(mass).expand(len(state.x),1)
            hints=hint_features if self.feature_version>=2 else []
            features=torch.cat((dimensionless/rms,*hints,dt_feature,radius_feature,mass_feature,
                                torch.log(rms),material,kind),-1)
            if features.shape[-1]!=self.feature_count:
                raise AssertionError('Elastic direction feature contract changed')
            learned=(self.network(features.to(self.network[0].weight.dtype)).tanh()
                     *self.learned_coefficient_scale).to(state.x.dtype)
            analytic_prior=dimensionless/dimensionless.norm(dim=-1,keepdim=True).clamp_min(1e-12)
            # The residual is a small initialization/regularizer; learned
            # coefficients are the principal direction once trained.
            coefficients=torch.nn.functional.normalize(self.analytic_prior_fraction*analytic_prior+learned,dim=-1,eps=1e-12)
            # Bound the predicted RMS displacement in object-relative units.
            displacement=torch.einsum('bncp,bp->bnc',basis,coefficients)
            cap=scene.length*self.maximum_fraction
            rms_gram=torch.einsum('bncp,bncq->bpq',basis,basis)/len(ids)
            norm=displacement.square().sum(-1).mean(-1).clamp_min(1e-30).sqrt()
            proposal_scale=(cap/norm.clamp_min(cap)).clamp_max(1)
            displacement=displacement*proposal_scale[:,None,None]
            mass_gram=torch.einsum('bncp,n,bncq->bpq',basis,scene.mass[ids],basis)
            local=residual.new_zeros(len(state.x),len(ids),6)
            local=local.index_copy(2,torch.arange(3,device=state.x.device),displacement)
            full=result.new_zeros(len(state.x),scene.n,6).index_copy(1,ids,local)
            result=result+full
            object_columns=[displacement]
            object_bases=[basis]
            object_coefficients=[coefficients]
            if self.local_heads:
                local_coefficients=(self.local_network(features.to(self.local_network[0].weight.dtype))
                    .tanh().reshape(len(state.x),self.local_heads,self.mode_count).to(state.x.dtype))
                local_coefficients=torch.nn.functional.normalize(local_coefficients,dim=-1,eps=1e-12)
                for head in range(self.local_heads):
                    local_basis=self._localized_basis(points,scene.rest_x[ids].to(residual.dtype),
                                                       scene.mass[ids].to(residual.dtype),basis,head)
                    local_disp=torch.einsum('bncp,bp->bnc',local_basis,local_coefficients[:,head])
                    local_norm=local_disp.square().sum(-1).mean(-1).sqrt().clamp_min(1e-30)
                    local_disp=local_disp*(cap/local_norm.clamp_min(cap)).clamp_max(1)[:,None,None]
                    object_columns.append(local_disp)
                    object_bases.append(local_basis)
                    object_coefficients.append(local_coefficients[:,head])
            object_columns=torch.stack(object_columns,-1)
            local_columns=residual.new_zeros(len(state.x),len(ids),6,object_columns.shape[-1])
            local_columns[:,:,:3]=object_columns
            full_columns=result.new_zeros(len(state.x),scene.n,6,object_columns.shape[-1]).index_copy(1,ids,local_columns)
            column_parts.append(full_columns)
            details.append({'object_id':int(object_id),'nodes':len(ids),'ids':ids,
                            'basis':torch.cat(object_bases,-1),
                            'predicted_coefficients':torch.cat(object_coefficients,-1),
                            'features':features.detach(),
                            'mass_gram':mass_gram.detach(),
                            'rms_gram':rms_gram.detach(),
                            'mass_sum':mass.detach(),
                            'cap':float(cap),
                            'proposal_scale':proposal_scale.detach(),
                            'analytic_prior_coefficients':analytic_prior,
                            'proposal_rms_m':displacement.square().sum(-1).mean(-1).sqrt().detach(),
            'columns':full_columns})
        all_columns=torch.cat(column_parts,-1) if column_parts else residual.new_zeros(len(state.x),scene.n,6,0)
        # Diagnostics must not keep a rollout autograd graph alive between
        # calls. The explicitly returned details still carry training grads.
        self.last_details=[{key:(value.detach() if torch.is_tensor(value) else value)
                            for key,value in item.items()} for item in details]
        if return_details:
            return (result,details,all_columns) if self.local_heads else (result,details)
        return result


@torch.no_grad()
def response_columns(scene,state,metric,residual,ids,*,return_details=False):
    """One nonrigid physical response column per actual material object."""
    material=ids[(scene.kind[ids]<2)&scene.movable[ids]]
    columns=[];details=[]
    for object_id in torch.unique(scene.object_id[material]).tolist():
        selected=material[scene.object_id[material]==object_id]
        if len(selected)<8:continue
        # Bound this prototype's preprocessing; large objects need a sparse
        # basis implementation rather than an implicit unbounded allocation.
        if len(selected)>4096:raise ValueError('Elastic shape mode object exceeds 4096-node budget')
        basis=polynomial_basis(state.x[:,selected],scene.mass[selected])
        local=residual.new_zeros(len(residual),len(selected),6,basis.shape[-1]);local[...,:3,:]=basis
        full=residual.new_zeros(len(residual),scene.n,6,basis.shape[-1]).index_copy(1,selected,local)
        image=metric_action(metric,full)
        gram=torch.einsum('bnip,bniq->bpq',full,image)
        rhs=torch.einsum('bnip,bni->bp',full,residual)
        coefficients=_pseudoinverse_solve(gram,rhs)
        value=torch.einsum('bnip,bp->bni',local,coefficients)
        rms=((value[...,:3].square().sum(-1)*scene.mass[selected]).sum(-1)/scene.mass[selected].sum()).sqrt()
        active=torch.isfinite(value).flatten(1).all(-1)&(rms>1e-14)
        if not bool(active.any()):continue
        value=torch.where(active[:,None,None],value/rms[:,None,None].clamp_min(1e-30),0.)
        columns.append(residual.new_zeros(len(residual),scene.n,6).index_copy(1,selected,value))
        details.append({'object_id':object_id,'nodes':len(selected),'candidate_basis_columns':basis.shape[-1],
                        'quadratic_response_rms_m':rms.cpu().tolist()})
    previous=getattr(scene,'elastic_shape_mode_diagnostic',{})
    previous.update({row['object_id']:row for row in details});scene.elastic_shape_mode_diagnostic=previous
    result=torch.stack(columns,-1) if columns else residual.new_zeros(len(residual),scene.n,6,0)
    return (result,details) if return_details else result
