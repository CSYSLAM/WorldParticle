"""Shared directional inverse response with dimensionless neural inputs."""
import torch
from .inverse_response import InversePortResponse,energy_gram,inverse_replacement
from .protected_response import metric_action
from .vector_ports import SharedVectorPorts
from .complete_response import coarse_gain


def dimensionless_relations(scene,relation,dt):
    """Normalize physical energy coefficients, without adding ALM penalties.

    Legacy relation parameters encode log1p(energy)/10. Contact slots 2:5
    already contain dimensionless friction/geometry and must stay unchanged.
    Numerical inverse conditioning is already dimensionless: W=h M^-1/2.
    """
    layout=scene.layout;batch=len(relation)
    mass=relation.new_zeros(len(layout.incidence));count=torch.zeros_like(mass)
    for rows,ids in layout.slots:
        active=scene.movable[ids]
        mass=mass.index_add(0,rows,scene.mass[ids]*active)
        count=count.index_add(0,rows,active.to(count.dtype))
    reference=mass/count.clamp_min(1)
    scale=dt**2/(reference.clamp_min(1e-30)*scene.length**2)
    old=relation[...,9:18].double()
    contact=(layout.relation_kind>=6)&(layout.relation_kind<=10)
    keep=contact[:,None]&(torch.arange(9,device=relation.device)[None]>=2)
    energy=torch.expm1(torch.where(keep[None],0.,10*old))
    normalized=torch.log1p(energy*scale.double()[None,:,None])/10
    parameter=torch.where(keep[None],old,normalized).to(relation.dtype)
    return torch.cat((relation[...,:9],parameter,relation[...,18:]),-1)


class VectorInverseResponse(InversePortResponse):
    """Six direction-preserving columns, shared across every entity type."""
    def __init__(self,width=32,*,ridge=1e-8,rank_relative=1e-12):
        super().__init__(width,ridge=ridge)
        del self.shape
        self.ports=SharedVectorPorts(width)
        self.rank_relative=rank_relative
        self.mode='learned'
        self.width=width

    def encode(self,scene,state,physical,baseline,relation,dt):
        normalized=dimensionless_relations(scene,relation,dt)
        active=scene.dof[None,:,:,None]&scene.dof[None,:,None,:]
        # mass_whitening includes h, so W H W and its inverse already have
        # no time units. Ghost blocks are only numerical padding.
        condition=torch.where(active,baseline,0.)
        return super().encode(scene,state,physical,condition,normalized,dt)

    def make_ports(self,scene,state,proposal,latent,relation,dt,*,unit_shapes=False):
        return self.ports(scene,state,proposal,latent,relation,dt,unit_shapes=unit_shapes)

    def forward(self,scene,state,residual,whitening,baseline,relation,physical,coarse,metric,dt,*,return_coarse_gain=False):
        if self.mode not in ('learned','physical_vector'):raise ValueError(self.mode)
        w=whitening.double();source=residual.double()
        if getattr(coarse,'packed',False) and not coarse.all_mobile_singletons:
            from .component_batch import vector_response
            return vector_response(self,scene,state,residual,whitening,baseline,relation,physical,coarse,metric,dt,return_coarse_gain)
        if getattr(coarse,'all_mobile_singletons',False):
            # A complete six-DOF coarse inverse has no complementary ports.
            # Keep the shared latent/coarse head, rather than constructing
            # and subsequently annihilating six learned columns per body.
            valid=torch.ones(len(source),device=source.device,dtype=torch.bool)
            if return_coarse_gain:
                h,_=self.encode(scene,state,physical,baseline,relation,dt) if self.mode=='learned' else (physical.new_zeros(len(physical),scene.n,self.width),None)
                gain=coarse_gain(self,scene,h,relation,coarse) if self.mode=='learned' else h.new_ones(*h.shape[:2])
                return torch.zeros_like(source),valid&torch.isfinite(gain).all(-1),gain.double()
            return torch.zeros_like(source),valid
        local=-(w@baseline.double()@(w@source[...,None]))[...,0]
        if self.mode=='learned':h,_=self.encode(scene,state,physical,baseline,relation,dt)
        else:h=physical.new_zeros(len(physical),scene.n,self.width)
        # These directions contain the complementary residual's phase/sign.
        raw=self.make_ports(scene,state,-local,h,relation,dt,unit_shapes=self.mode=='physical_vector')
        safe_w=w+torch.diag_embed((~scene.dof).double())[None]
        root,info=torch.linalg.cholesky_ex(baseline.double(),check_errors=False);factor=safe_w@root
        valid=(info==0).all(-1)&torch.isfinite(root).flatten(1).all(-1)
        if not hasattr(scene,'_vector_active_components'):
            scene._vector_active_components=tuple(i for i,ids in enumerate(scene.interaction_entities) if bool(scene.movable[ids].any()))
        result=local
        for index in scene._vector_active_components:
            if getattr(coarse,'singleton_ids',None) is not None and coarse.singleton_group_mask[index]:continue
            ids=scene.interaction_entities[index]
            full=raw.new_zeros(len(raw),scene.n,6,6).index_copy(1,ids,raw[:,ids])
            projected=full-coarse(metric_action(metric,full))
            before,ib=torch.linalg.solve_ex(factor[:,ids],full[:,ids],check_errors=False)
            after,ia=torch.linalg.solve_ex(factor[:,ids],projected[:,ids],check_errors=False)
            original=before.square().sum((1,2));norm=after.square().sum((1,2))
            keep=(norm>self.rank_relative*original)&(original>0)&torch.isfinite(norm)
            complete=coarse.ranks[index]>=scene.dof[ids].sum()
            keep=keep&(~complete[:,None])
            # Per-column metric scaling keeps velocity from suppressing small
            # but genuine force directions. Numerical rank choices are frozen.
            scale=torch.where(keep,norm.clamp_min(1e-300).rsqrt(),0.).detach()
            full=projected*scale[:,None,None,:]
            gh=energy_gram(scene,metric,full)
            mobile=scene.movable[ids]
            if self.mode=='learned':
                pooled=(h[:,ids]*mobile[None,:,None]).sum(1)/mobile.sum().clamp_min(1)
                logit=self.factor(pooled)[...,0].double()
            else:logit=source.new_full((len(source),),30.)
            delta,ok,_=inverse_replacement(source[:,ids],full[:,ids],factor[:,ids],gh,logit,ridge=self.ridge)
            valid=valid&ok&(ib==0).all(-1)&(ia==0).all(-1)
            result=result.index_add(1,ids,-delta)
        if return_coarse_gain:
            gain=coarse_gain(self,scene,h,relation,coarse) if self.mode=='learned' else h.new_ones(*h.shape[:2])
            return result*scene.dof,valid,gain.double()
        return result*scene.dof,valid
