"""Finite joint rows combined with the independent rigid contact residual."""
import math
from types import SimpleNamespace
import torch
from . import math3d as so3
from .data import State
from .unified_graph import build_topology
from .mixed_metric import MetricTerm
from . import rigid_contact_system as contact
from .general_joint_rows import evaluate


def begin(scene,start,candidates,dt):
    context=contact.begin(scene,start,candidates,dt)
    from . import prescribed_motion as prescribed
    expected={'joint_angle_prev'}|(prescribed.HISTORY_KEYS if prescribed.enabled(scene) else set())
    if start.history is None or set(start.history)!=expected:raise ValueError('Complete joint/prescribed pose history is required and must be restored on restart')
    context.previous_angle=start.history['joint_angle_prev']
    if context.previous_angle.shape!=(len(start.x),len(scene.joint_spec.owners)) or not bool(torch.isfinite(context.previous_angle).all()):raise ValueError('Invalid committed joint winding history')
    context.material_start=prescribed.previous(scene,start) if prescribed.enabled(scene) else start
    context.joint_start=scene.joint_coordinates(context.material_start,context.previous_angle)
    context.targets,context.target_rates,context.efforts=scene.joint_controls
    return context


def joint_rows(scene,state,context,*,frozen_bound=None):
    current=scene.joint_coordinates(state,context.previous_angle)
    spec=scene.joint_spec
    if scene.metadata.get('prescribed_pose_history'):spec=SimpleNamespace(**{**vars(spec),'enabled':spec.enabled&scene.movable[spec.owners].any(-1)})
    result=evaluate(spec,current,context.joint_start,context.dt,context.targets,context.target_rates,frozen_bound=frozen_bound)
    # Scalar joint efforts perform virtual work in the same objective axes.
    enabled=spec.enabled[None,:,None]
    effort=context.efforts*enabled
    result['pair_residual']=result['pair_residual']-torch.einsum('bjcsi,bjc->bjsi',current.axis_jacobian,effort)
    result['energy']=result['energy']-(effort*current.axis).sum(-1)
    return result


def joint_energy(scene,state,context,*,frozen_bound=None):
    current=scene.joint_coordinates(state,context.previous_angle,jacobian=False)
    spec=scene.joint_spec
    if scene.metadata.get('prescribed_pose_history'):spec=SimpleNamespace(**{**vars(spec),'enabled':spec.enabled&scene.movable[spec.owners].any(-1)})
    result=evaluate(spec,current,context.joint_start,context.dt,context.targets,context.target_rates,frozen_bound=frozen_bound,derivatives=False)
    effort=context.efforts*spec.enabled[None,:,None]
    return result['energy']-(effort*current.axis).sum(-1)


def merge(scene,state,result,joint):
    ids=scene.joint_spec.owners
    result['residual']=result['residual'].index_add(1,ids.flatten(),joint['pair_residual'].flatten(1,2))*scene.dof
    result['terms']=[*result['terms'],MetricTerm(ids,joint['jacobian'].detach(),joint['weight'].detach())]
    result['valid']=torch.cat((result['valid'],joint['valid']),-1);result['joint_rows']=joint
    return result


def rows(scene,state,start,context):return merge(scene,state,contact.rows(scene,state,start,context),joint_rows(scene,state,context))


def system(scene,state,start,tx,tq,context):
    residual,metric,result=contact.system(scene,state,start,tx,tq,context)
    merged=merge(scene,state,result,joint_rows(scene,state,context))
    joint=merged['joint_rows'];ids=scene.joint_spec.owners
    residual=residual+state.x.new_zeros(residual.shape).index_add(1,ids.flatten(),joint['pair_residual'].flatten(1,2))*scene.dof
    return residual,contact.ContactMetric(metric.local,merged['terms'],scene.dof),merged


def merit(scene,state,start,tx,tq,context,frozen):
    energy=contact.merit(scene,state,start,tx,tq,context,frozen)
    if not len(scene.joint_spec.owners):return energy
    joint=joint_energy(scene,state,context,frozen_bound=frozen['joint_rows']['bound_lower']);ids=scene.joint_spec.owners
    mobile=scene.movable[ids];count=mobile.sum(-1).clamp_min(1)
    for side in (0,1):energy=energy.index_add(1,ids[:,side],joint*(mobile[:,side]/count)[None])
    return energy


def relation_features(scene,state,start,context,result,feedback):
    base=contact.relation_features(scene,state,start,context,result,feedback)
    s=scene.joint_spec;j=result['joint_rows'];current=j['coordinates'];dt=context.dt;length=scene.length;b=len(state.x);n=len(s.owners)
    scale=state.x.new_tensor([1/length]*3+[1.]*3)
    target=torch.where((s.limit_k>0)&((s.lower>-1e10)|(s.upper<1e10)),torch.minimum(torch.maximum(context.targets,s.lower),s.upper),context.targets)
    bound=torch.where(j['bound_lower'],s.lower,s.upper);bound=torch.where(torch.isfinite(bound)&(bound.abs()<1e10),bound,current.axis.detach())
    errors=[current.structural,current.axis-target,current.axis-bound]
    rates=[current.structural-context.joint_start.structural,current.axis-context.joint_start.axis,current.axis-context.joint_start.axis]
    stiffness=[s.stiffness,s.drive_k,s.limit_k];damping=[s.damping,s.drive_d,s.limit_d]
    features=[];active=scene.joint_potential().reshape(3,n)
    mass=scene.mass[s.owners].sum(-1);scalar_feedback=feedback[:,s.owners[:,0]]
    for group in range(3):
        value=state.x.new_zeros(b,n,9);value[...,:6]=errors[group]*scale
        parameter=torch.zeros_like(value);parameter[...,:6]=torch.log1p(stiffness[group]/scale.square())/10
        parameter[...,6:8]=torch.stack((torch.log1p((damping[group][:,:3]*length**2/dt).sum(-1))/10,torch.log1p((damping[group][:,3:]/dt).sum(-1))/10),-1)
        rate=torch.zeros_like(value);rate[...,:6]=rates[group]*scale
        scalar=torch.stack((active[group][None].expand(b,-1).to(value.dtype),value.new_full((b,n),math.log(dt)),torch.log1p(mass)[None].expand(b,-1),scalar_feedback,torch.ones_like(scalar_feedback)),-1)
        features.append(torch.cat((value,parameter,rate,scalar),-1))
    if scene.mimic_spec is not None:
        from .mimic_constraint import rows as mimic_rows
        m=mimic_rows(scene,state);s=scene.mimic_spec;r=len(s.pairs)
        value=state.x.new_zeros(b,r,9);value[...,0]=m.value
        value[...,1]=s.coefficients[:,0]*torch.where(s.linear[:,0],1/scene.length,1.)
        coordinate_scale=torch.where(s.linear,1/scene.length,1.)
        value[...,2]=s.coefficients[:,1]*coordinate_scale[:,0]/coordinate_scale[:,1]
        value[...,3]=s.linear[:,0];value[...,4]=s.linear[:,1]&(s.coefficients[:,1]!=0);value[...,8]=1.
        parameter=torch.zeros_like(value);rate=torch.zeros_like(value)
        scalar=value.new_zeros(b,r,5);scalar[...,0]=m.enabled;scalar[...,1]=math.log(dt);scalar[...,4]=1.
        features.append(torch.cat((value,parameter,rate,scalar),-1))
    return torch.cat((base,*features),1)


coarse=contact.coarse


def acceptance_groups(scene,c,depth,travel,*,labels=False):
    possible=(depth[0]+travel[0,c.owners].sum(-1)>=0)&(c.material>0)
    owners,kinds,active=scene.all_relations(c.owners,possible)
    graph=build_topology(scene.kind,scene.movable,scene.object_id,owners,kinds,potential_active=active)
    if labels:return graph.interaction_id
    return [torch.where(graph.interaction_id==i)[0] for i in torch.unique(graph.interaction_id).tolist() if bool(scene.movable[graph.interaction_id==i].any())]


def predictor_domain(scene,start,forecast,config,*,dt=None):
    """Reserve enough single-axis angular travel for every bounded correction.

    Each body correction has angle <= trust_radius * relaxation. Summing
    endpoint lengths bounds relative joint travel along the actual warm
    geodesic plus all K corrections, so nearest-turn unwrapping is unique.
    """
    ids=scene.joint_spec.owners;single=scene.joint_spec.angular_mode==1
    angle=so3.log(so3.multiply(forecast.q,so3.conjugate(start.q))).norm(dim=-1)
    bound=angle[:,ids].sum(-1)+scene.movable[ids].sum(-1)*config.get('unroll',4)*config.get('trust_radius',.1)*config.get('relaxation',.8)
    prescribed_valid=torch.ones(len(start.x),device=start.x.device,dtype=torch.bool)
    if scene.metadata.get('prescribed_pose_history'):
        from .prescribed_motion import domain
        _,prescribed_valid,turn=domain(scene,start,scene.metadata['dt'] if dt is None else dt)
        full=angle.new_zeros(angle.shape).index_copy(1,scene.prescribed_ids,turn)
        bound=bound+full[:,ids].sum(-1)
    if scene.mimic_spec is not None:
        # Reserve both bounded retractions for warm start and all K steps.
        bound=bound+scene.movable[ids].sum(-1)*.2*(config.get('unroll',4)+1)
    return ((bound<torch.pi-1e-6)|~single).all(-1)&torch.isfinite(bound).all(-1)&prescribed_valid


def tangent(scene,state):
    if scene.mimic_spec is None:return None
    from .mimic_constraint import Tangent
    return Tangent(scene,state)


def retract(scene,state):
    if scene.mimic_spec is None:return state,torch.ones(len(state.x),device=state.x.device,dtype=torch.bool)
    from .mimic_constraint import retract as project
    return project(scene,state)


def valid_state(scene,state):
    if scene.mimic_spec is None:return torch.ones(len(state.x),device=state.x.device,dtype=torch.bool)
    from .mimic_constraint import rows as mimic_rows
    current=mimic_rows(scene,state)
    return current.valid.all(-1)&torch.isfinite(current.value).all(-1)&(current.value.abs()<=1e-9).all(-1)


def finish(scene,start,output,context,accepted):
    value=scene.joint_coordinates(output,context.previous_angle).axis[...,3]
    committed=torch.where(scene.joint_spec.angular_mode==1,value,0.)
    history={'joint_angle_prev':torch.where(accepted[:,None],committed,context.previous_angle)}
    velocity=torch.where(scene.movable[None,:,None],output.v,start.v)
    omega=torch.where(scene.movable[None,:,None],output.omega,start.omega)
    if scene.metadata.get('prescribed_pose_history'):
        from .prescribed_motion import commit
        history,velocity,omega=commit(scene,start,output,context,accepted,history,velocity,omega)
    return State(output.x,output.q,velocity,omega,history)
