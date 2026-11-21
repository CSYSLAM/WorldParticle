"""Shape an existing correction using normal activation on prospective groups.

The bound describes the unprojected Exp/affine path. Existing retractions,
whole-path guards and model-specific energy acceptance remain authoritative;
this is not a new nonlinear solve or an acceptance certificate for every law.
"""
import torch

from .contact_activation_bound import normal_activation_curvature
from .protected_response import metric_action


def _rigid_rows(scene,state,update,contacts,context,frozen):
    count=len(contacts.owners)
    if not count:return contacts.owners,contacts.owners>=0,update.new_zeros(len(update),0)
    jacobian=torch.einsum('ri,brsij->brsj',contacts.normal,frozen['relative_skeleton_jacobian'])
    linear=-(jacobian*update[:,contacts.owners]).sum((-1,-2))
    arm=frozen['skeleton']-state.x[:,contacts.owners]
    remainder=.5*(arm.norm(dim=-1)*update[:,contacts.owners,3:].square().sum(-1)).sum(-1)
    shift=scene.metadata['contact_alpha']*context.depth0
    value=normal_activation_curvature(frozen['raw_depth'],linear,remainder,shift,contacts.material[None],
        compiled=bool(getattr(scene,'fused_physical_checks',False)))
    return contacts.owners,contacts.owners>=0,value


def _port_motion(scene,state,update,ports,jacobian,normal):
    velocity=ports.point_velocity(jacobian,update)
    linear=-((velocity[:,:,1]-velocity[:,:,0])*normal).sum(-1)
    remainder=torch.zeros_like(linear)
    if ports.has_rigid:
        # A rigid port has exactly one nonzero owner weight. No [R,2,S,6,3]
        # Jacobian is needed to bound its angular second derivative.
        slot=ports.weights.argmax(-1)
        owners=ports.ids[:,None].expand(-1,2,-1).gather(2,slot[...,None])[...,0]
        radius=(ports.local_points-scene.body_com[owners]).norm(dim=-1)
        remainder=.5*(radius[None]*ports.rigid[None]*update[:,owners,3:].square().sum(-1)).sum(-1)
    return linear,remainder


def _soft_rows(scene,state,update,snapshot,context,frozen):
    ports=snapshot.ports;participating=(ports.owners>=0)&(ports.weights>0).any(1)
    if not len(ports.owners):return ports.owners,participating,update.new_zeros(len(update),0)
    linear,remainder=_port_motion(scene,state,update,ports,frozen['jacobian'],snapshot.normal)
    value=normal_activation_curvature(frozen['raw_depth'],linear,remainder,
        torch.zeros_like(linear),context.material.stiffness[None],
        compiled=bool(getattr(scene,'fused_physical_checks',False)))
    return ports.owners,participating,value


def _self_rows(scene,state,update,snapshot,context,frozen):
    from .self_contact_rows import normal_potential
    ports=snapshot.ports;participating=(ports.owners>=0)&(ports.weights>0).any(1)
    if not len(ports.owners):return ports.owners,participating,update.new_zeros(len(update),0)
    linear,remainder=_port_motion(scene,state,update,ports,frozen['jacobian'],snapshot.normal)
    maximum=(frozen['raw_depth']+linear+remainder).clamp_min(0)
    # The pinned C2 self potential has nondecreasing curvature with depth.
    # Thus E(t*M)<=t²*E(M) for inactive rows. This uses its actual barrier /
    # finite continuation, never the insufficient quadratic K*M²/2 alone.
    worst=normal_potential(snapshot.margin[None]-maximum,snapshot.margin[None],context.stiffness[None])[0]
    value=torch.where(frozen['raw_depth']<=0,2*worst,0.)
    return ports.owners,participating,value


@torch.no_grad()
def activation_rows(scene,state,update,context,frozen):
    contacts=context.contacts
    if hasattr(contacts,'rigid'):
        rows=[_rigid_rows(scene,state,update,contacts.rigid,context.rigid,frozen['rigid_rows']),
              _soft_rows(scene,state,update,contacts.soft,context.soft,frozen['soft_rows'])]
        if hasattr(contacts,'self_contacts'):
            rows.append(_self_rows(scene,state,update,contacts.self_contacts,context.self_contact,frozen['self_rows']))
        return rows
    return [_rigid_rows(scene,state,update,contacts,context,frozen)]


def scale_proposal(scene,state,context,frozen,metric,residual,update,labels):
    """One frozen scalar on each complete prospective interaction component.

    labels must contain every material/metric coupling and every contact
    reachable by this full correction. Zero-weight/padded ports do not own
    row energy. Each physical row is allocated once over responding slots.
    """
    if residual.shape!=update.shape or update.shape!=(*state.x.shape[:2],6):
        raise ValueError('Matched physical residual and six-slot correction required')
    if labels.shape!=(scene.n,) or labels.dtype!=torch.long or labels.device!=update.device or bool(((labels<0)|(labels>=scene.n)).any()):
        raise ValueError('Complete prospective physical component labels required')
    with torch.no_grad():
        valid=torch.isfinite(residual).all((1,2))&torch.isfinite(update).all((1,2))
        safe=torch.where(valid[:,None,None],update.detach()*scene.dof,0.)
        curvature=safe.new_zeros(len(safe),scene.n)
        for owners,participating,value in activation_rows(scene,state,safe,context,frozen):
            ids=owners.clamp_min(0);mobile=scene.movable[ids]&participating
            if not len(ids):continue
            if not bool(torch.isfinite(value).all()) or bool((value<0).any()):raise ValueError('Finite nonnegative activation curvature required')
            row_labels=labels[ids]
            lower=torch.where(mobile,row_labels,scene.n).amin(-1)
            upper=torch.where(mobile,row_labels,-1).amax(-1)
            if bool(((value>0)&(mobile.sum(-1)>1)[None]&(lower!=upper)[None]).any()):
                raise ValueError('An activating contact spans different prospective components')
            share=mobile.to(value.dtype)/mobile.sum(-1).clamp_min(1)[:,None]
            curvature=curvature.index_add(1,ids.flatten(),(value[...,None]*share[None]).flatten(1,2))
        extra=torch.zeros_like(curvature).index_add(1,labels,curvature)
        affected=extra>0
        scales=torch.ones_like(extra)
        details=[]
        if bool(affected.any()):
            image=metric_action(metric,safe)
            linear=-(residual.detach()*safe).sum(-1)
            metric_energy=(safe*image).sum(-1)
            numerator=torch.zeros_like(linear).index_add(1,labels,linear)
            old_curvature=torch.zeros_like(metric_energy).index_add(1,labels,metric_energy)
            denominator=old_curvature+extra
            good=torch.isfinite(numerator)&torch.isfinite(denominator)&(numerator>=0)&(old_curvature>=0)
            ratio=(numerator/denominator.clamp_min(torch.finfo(update.dtype).tiny)).clamp(0.,1.)
            scales=torch.where(affected,torch.where(good,ratio,0.),scales)
            valid&=(good|~affected).all(-1)
            if (getattr(scene,'fused_physical_checks',False)
                    and scene.metadata.get('compact_numerical_diagnostics',False)):
                details.append({'values_omitted':True})
            else:
                for label in torch.where(affected.any(0))[0].tolist():
                    details.append({'component':label,'mobile_entities':int(((labels==label)&scene.movable).sum()),
                        'linear_descent':numerator[:,label].cpu().tolist(),'metric_curvature':old_curvature[:,label].cpu().tolist(),
                        'new_contact_curvature':extra[:,label].cpu().tolist(),'scale':scales[:,label].cpu().tolist()})
        selected=scales[:,labels,None]
    return torch.where(valid[:,None,None],update*scene.dof*selected,0.),valid,details
