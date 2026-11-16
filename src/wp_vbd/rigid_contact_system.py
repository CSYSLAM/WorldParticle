"""Complete rigid inertia/contact residual and positive sparse metric."""
from types import SimpleNamespace
import math
import torch
from . import math3d as so3
from .data import State
from .mixed_metric import FrozenMixedMetric,frozen_contact_term
from .rigid_contact_rows import ContactContext,evaluate
from .coarse_response import rigid_component_modes


class ContactMetric(FrozenMixedMetric):
    def __init__(self,local,terms,dof):self.local=local.detach();self.terms=terms;self.dof=dof


def begin(scene,start,contacts,dt):
    from .rigid_collision import supports
    _,_,_,surface,depth=contacts.geometry(scene,start)
    delta=surface[:,:,1]-surface[:,:,0];normal=contacts.normal
    c0_t=-(delta-(delta*normal).sum(-1,keepdim=True)*normal)
    rho_n,rho_t=supports(scene,start,contacts,dt)
    return SimpleNamespace(contacts=contacts,depth0=depth,c0_t=c0_t,rho_n=rho_n,rho_t=rho_t,dual_t=torch.zeros_like(delta),dt=dt)


def _row_values(scene,c,x,q,previous_x,previous_q,depth0,c0_t,rho_n,rho_t,dual_t,dt):
    """Pure physical row evaluation; no query, cache or history publication."""
    state=State(x,q,None,None);material_start=State(previous_x,previous_q,None,None)
    origin,q,skeleton,surface,raw=c.geometry(scene,state)
    if not len(c.owners):
        empty=raw;return {'residual':state.x.new_zeros(len(state.x),scene.n,6),'terms':[],'active':empty.bool(),'raw_depth':raw,'effective_depth':raw,'normal_load':empty,'tangent_residual':surface[:,:,0],'tangent_force':surface[:,:,0],'valid':empty.new_ones(empty.shape,dtype=torch.bool),'damping_active':empty}
    previous_origin,previous_q,old_skeleton,old_surface,_=c.geometry(scene,material_start)
    alpha=scene.metadata['contact_alpha'];effective=torch.where(raw<0,raw,raw-alpha*depth0)
    normal_load=c.material[None]*effective.clamp_min(0)
    # At the finite normal ALM fixed point lambda_n=K*C. Inserting it into
    # k_eff*C+s*lambda gives exactly K*C for every positive numerical rho.
    # Tangential Coulomb duals remain explicit and are not eliminated.
    dual=dual_t+normal_load[...,None]*c.normal
    ctx=ContactContext(c.material[None],rho_n,rho_t,c.damping[None],c.mu[None],dual,(1-alpha)*c0_t,c.mu.new_full((1,len(c.mu)),scene.metadata['friction_epsilon']),dt)
    result=evaluate(origin,q,previous_origin,previous_q,scene.body_com[c.owners],c.points,c.offsets,c.normal,effective,c.surface_velocity,ctx)
    active=(normal_load>0)&(c.material[None]>0)&scene.movable[c.owners].any(-1)[None]
    approach=((skeleton[:,:,1]-old_skeleton[:,:,1]-skeleton[:,:,0]+old_skeleton[:,:,0])*c.normal).sum(-1)/dt
    damping=c.damping[None]*(approach<0)*active
    normal_weight=(c.material[None]+damping/dt)*active
    normal_j=torch.einsum('ri,brsij->brsj',c.normal,result['relative_skeleton_jacobian'])[:,:,None]
    eigen,frame=so3.eigh_zero_safe(result['tangent_solve_metric'].detach())
    tangent_j=torch.einsum('bric,brsij->brcsj',frame,result['relative_surface_jacobian'])
    residual=state.x.new_zeros(len(state.x),scene.n,6)
    wrench=torch.cat((result['force'],result['torque']),-1)
    residual=residual.index_add(1,c.owners.flatten(),-wrench.flatten(1,2))
    slip=surface[:,:,1]-old_surface[:,:,1]-surface[:,:,0]+old_surface[:,:,0]+c.surface_velocity*dt
    tangent=-(slip-(slip*c.normal).sum(-1,keepdim=True)*c.normal)+(1-alpha)*c0_t
    result.update(residual=residual*scene.dof,active=active,raw_depth=raw,effective_depth=effective,normal_load=normal_load,tangent_residual=tangent,surface=surface,skeleton=skeleton,damping_active=damping,
                  _normal_j=normal_j.detach(),_normal_weight=normal_weight[...,None].detach(),
                  _tangent_j=tangent_j.detach(),_tangent_weight=eigen.clamp_min(0).detach())
    return result


def rows(scene,state,start,context):
    c=context.contacts;previous=getattr(context,'material_start',start)
    arguments=(scene,c,state.x,state.q,previous.x,previous.q,context.depth0,context.c0_t,
               context.rho_n,context.rho_t,context.dual_t,context.dt)
    if getattr(scene,'contact_row_checkpointing',False) and torch.is_grad_enabled() and len(c.owners):
        from torch.utils.checkpoint import checkpoint
        # Tensor inputs bind this evaluation's numerical duals and poses.
        # Recomputing the pure rows never reads the subsequently updated
        # context/scene candidates and never repeats an accepted transaction.
        result=checkpoint(_row_values,*arguments,use_reentrant=False,preserve_rng_state=False)
    else:result=_row_values(*arguments)
    if len(c.owners):
        compact=bool(getattr(scene,'compact_frozen_contact_metric',False))
        result['terms']=[frozen_contact_term(c,result.pop('_normal_j'),result.pop('_normal_weight'),compact=compact,cache_name='_normal_metric_selection'),
                         frozen_contact_term(c,result.pop('_tangent_j'),result.pop('_tangent_weight'),compact=compact,cache_name='_tangent_metric_selection')]
    return result


def system(scene,state,start,tx,tq,context):
    dt=context.dt;rotation=so3.matrix(state.q)
    angular=rotation@scene.inertia@rotation.transpose(-1,-2)/dt**2
    angle=so3.log(so3.multiply(state.q,so3.conjugate(tq)))
    residual=torch.cat((scene.mass[None,:,None]/dt**2*(state.x-tx),(angular@angle[...,None])[...,0]),-1)
    eye=torch.eye(3,device=state.x.device,dtype=state.x.dtype)
    local=state.x.new_zeros(len(state.x),scene.n,6,6)
    local[...,:3,:3]=scene.mass[None,:,None,None]/dt**2*eye;local[...,3:,3:]=angular
    contact=rows(scene,state,start,context)
    metric=ContactMetric(local,contact['terms'],scene.dof)
    return (residual+contact['residual'])*scene.dof,metric,contact


def coarse(scene,state):
    blocks=[]
    batched=bool(getattr(scene,'component_batch',False))
    if batched:
        from .component_batch import layout
        groups=layout(scene);center=groups.centers(state.x)
        packed_modes=rigid_component_modes(state.x,center[:,groups.labels],scene.dof).double()
    for ids in scene.interaction_entities:
        ids=ids[scene.movable[ids]]
        if not len(ids):continue
        if batched:modes=packed_modes[:,ids]
        else:
            center=(state.x[:,ids]*scene.mass[ids][None,:,None]).sum(1,keepdim=True)/scene.mass[ids].sum()
            modes=rigid_component_modes(state.x[:,ids],center,scene.dof[ids]).double()
        blocks.append((SimpleNamespace(ids=ids),modes,None,None,None))
    return SimpleNamespace(blocks=blocks)


def relation_features(scene,state,start,context,contact,feedback):
    c=context.contacts;b=len(state.x);r=len(c.owners);length=scene.length;dt=context.dt
    value=state.x.new_zeros(b,r,9);parameter=torch.zeros_like(value);rate=torch.zeros_like(value)
    value[...,0]=contact['effective_depth']/length;value[...,1:4]=contact['tangent_residual']/length
    value[...,4]=contact['normal_load']*dt**2/(scene.mass[c.owners].sum(-1)[None].clamp_min(1e-30)*length)
    if getattr(context,'matched_contact_history',False):
        value[...,5:8]=context.dual_t*dt**2/(scene.mass[c.owners].sum(-1)[None,:,None].clamp_min(1e-30)*length)
        value[...,8]=(c.match_index>=0).to(value.dtype)[None]
        parameter[...,5]=float(scene.metadata['contact_matching_policy']['mode']=='sticky')
    parameter[...,0]=torch.log1p(c.material*length**2)/10;parameter[...,1]=torch.log1p(c.damping*length**2/dt)/10
    parameter[...,2]=c.mu;parameter[...,3]=c.margins.sum(-1)/length;parameter[...,4]=scene.metadata['friction_epsilon']*dt/length
    rate[...,0]=((state.v[:,c.owners[:,1]]-state.v[:,c.owners[:,0]])*c.normal).sum(-1)*dt/length
    scalar=torch.stack((contact['active'].to(value.dtype),value.new_full((b,r),math.log(dt)),torch.log1p(scene.mass[c.owners].sum(-1))[None].expand(b,-1),feedback[:,c.owners[:,0]],torch.ones_like(value[...,0])),-1)
    return torch.cat((value,parameter,rate,scalar),-1)


def merit(scene,state,start,tx,tq,context,frozen,*,include_inertia=True):
    dt=context.dt;c=context.contacts
    angle=so3.log(so3.multiply(so3.conjugate(state.q),tq))
    energy=.5*scene.mass[None]/dt**2*(state.x-tx).square().sum(-1)+.5*(angle*(scene.inertia@angle[...,None])[...,0]).sum(-1)/dt**2
    if not include_inertia:energy=torch.zeros_like(energy)
    if not len(c.owners) and not torch.is_grad_enabled():return energy
    _,_,skeleton,surface,raw=c.geometry(scene,state);_,_,old_skeleton,old_surface,_=c.geometry(scene,getattr(context,'material_start',start))
    alpha=scene.metadata['contact_alpha'];effective=torch.where(raw<0,raw,raw-alpha*context.depth0)
    pair=.5*c.material[None]*effective.clamp_min(0).square()
    normal_displacement=((skeleton[:,:,1]-old_skeleton[:,:,1]-skeleton[:,:,0]+old_skeleton[:,:,0])*c.normal).sum(-1)
    pair=pair+.5*frozen['damping_active']/dt*normal_displacement.clamp_max(0).square()
    displacement=surface[:,:,1]-old_surface[:,:,1]-surface[:,:,0]+old_surface[:,:,0]+c.surface_velocity*dt
    tangent=-(displacement-(displacement*c.normal).sum(-1,keepdim=True)*c.normal)+(1-alpha)*context.c0_t
    trial=context.dual_t+context.rho_t[...,None]*tangent;limit=c.mu[None]*frozen['normal_load']
    def potential(vector):
        norm=vector.norm(dim=-1);return torch.where(norm<=limit,.5*norm.square(),limit*norm-.5*limit.square())
    pair=pair+(potential(trial)-potential(context.dual_t))/context.rho_t.clamp_min(1e-30)
    mobile=scene.movable[c.owners]
    count=mobile.sum(-1).clamp_min(1)
    for side in (0,1):energy=energy.index_add(1,c.owners[:,side],pair*(mobile[:,side]/count)[None])
    return energy
