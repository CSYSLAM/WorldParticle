"""One inertia, elastic/rod material, finite joints and complete contact rows."""
from types import SimpleNamespace

import torch

from . import self_contact_system as elastic, surface_joint_system as surface
from . import rod_contact_system as rod
from .data import State
from .mixed_metric import FrozenMixedMetric, MetricTerm
from .mixed_physics import residual_system
from .mixed_solver import relation_inputs
from .rigid_contact_system import ContactMetric

supports_rigid_component_batch=False
supports_component_batch=False
HISTORY_KEYS=surface.HISTORY_KEYS|rod.HISTORY_KEYS
contact_safety=elastic.contact_safety+'; NONANTIPARALLEL_NATIVE_ROD_MATERIAL_PATH'


def selected_history(state,keys):
    return State(state.x,state.q,state.v,state.omega,{key:state.history[key] for key in keys})


def initial_history(scene,state):
    spec=scene.joint_spec
    angle=scene.joint_coordinates(state,spec.coordinate_offset[None,:,3],jacobian=False).axis[...,3]
    return {**rod.initial_history(scene,state),
        'pin_x_prev':state.x[:,scene.fixed_particle_ids].clone(),
        'pin_initialized':torch.ones(len(state.x),device=scene.device,dtype=torch.bool),
        'joint_angle_prev':torch.where(spec.angular_mode==1,angle,0.)}


def validate_history(scene,state):
    if not isinstance(state.history,dict) or set(state.history)!=HISTORY_KEYS:
        raise ValueError('Complete shared elastic/rod/pin/joint history required')
    surface.validate_history(scene,selected_history(state,surface.HISTORY_KEYS))
    rod.validate_history(scene,selected_history(state,rod.HISTORY_KEYS))
    h=state.history;ids=scene.fixed_particle_ids
    same=(h['pin_x_prev']==h['rod_previous_x'][:,ids]).flatten(1).all(-1)
    if bool((h['rod_initialized']&(~h['pin_initialized']|~same)).any()):
        raise ValueError('Pin and rod histories must describe the same accepted physical step')


def begin(scene,start,candidates,dt):
    validate_history(scene,start)
    h=start.history;ids=scene.fixed_particle_ids
    pin_previous=torch.where(h['pin_initialized'][:,None,None],h['pin_x_prev'],start.x[:,ids])
    first_x=start.x.index_copy(1,ids,pin_previous)
    initialized=h['rod_initialized'][:,None,None]
    previous=State(torch.where(initialized,h['rod_previous_x'],first_x),
                   torch.where(initialized,h['rod_previous_q'],start.q),start.v,start.omega)
    context=elastic.begin(scene,selected_history(start,surface.HISTORY_KEYS),candidates,dt,
                          material_start_override=previous)
    context.rod,context.rod_memory=rod.begin_material(scene,context.material_start,start.history,dt)
    return context


def append_rod(scene,state,context,result):
    material=rod.material_rows(scene,state,context)
    result['residual']=(result['residual'].index_add(1,scene.rod.joints.flatten(),
                       material['residual'].flatten(1,2)))*scene.dof
    term=MetricTerm(scene.rod.joints,material['jacobian'].detach(),
                    context.rod.terms(material['coordinates'])[1].detach())
    result['terms']=[*result['terms'],term]
    result['valid']=torch.cat((result['valid'],material['in_domain']),-1)
    result['rod_rows']=material
    return result


def rows(scene,state,start,context):
    return append_rod(scene,state,context,elastic.rows(scene,state,start,context))


def system(scene,state,start,tx,tq,context):
    residual,_,extras,local=residual_system(scene,state,context.material_start,tx,tq,context.dt,
                                          return_metric_base=True)
    jacobian,_=relation_inputs(scene.elastic_scene,state,context.material_start,extras,context.dt,
                               torch.ones_like(state.x[...,0]))
    material=FrozenMixedMetric(scene,state,context.material_start,tx,tq,context.dt,extras,jacobian,
                               base=local,elastic_mode='objective')
    result=rows(scene,state,start,context)
    result['elastic_extras']=extras
    result['topology_active']=result['active']
    from .rod_material_ports import fields
    profile=None
    if getattr(scene,'rod_force_ports',False):
        stress,weight,_=context.rod.terms(result['rod_rows']['coordinates'])
        profile=torch.where(weight>0,stress/weight.clamp_min(1e-30),0.).detach()
    scene._rod_material_fields,valid=fields(scene,result['rod_rows']['jacobian'].detach(),profile)
    if getattr(scene,'rod_gravity_ports',False):
        from .rod_gravity_ports import profile as gravity_profile
        _,weight,_=context.rod.terms(result['rod_rows']['coordinates'])
        load,load_valid=gravity_profile(scene,state,result['rod_rows']['jacobian'].detach(),weight.detach())
        scene._rod_gravity_fields,field_valid=fields(scene,result['rod_rows']['jacobian'].detach(),load)
        valid=valid&load_valid&field_valid
        result['rod_gravity_material_profile']=load
    result['rod_material_profile']=profile
    result['valid']=torch.cat((result['valid'],valid[:,None]),-1)
    anticipated=[]
    if getattr(scene,'predictive_contact_metric',False):
        from .surface_contact_anticipation import metric as prepare
        # self rows end in [finite-joint, self-contact, rod] terms. Include
        # every material term, but no contact term, in the compliance baseline.
        joint_term=result['terms'][-3];rod_term=result['terms'][-1]
        baseline=ContactMetric(material.local,[*material.terms,joint_term,rod_term],scene.dof)
        joint_residual=state.x.new_zeros(residual.shape).index_add(1,
            scene.joint_spec.owners.flatten(),result['joint_rows']['pair_residual'].flatten(1,2))
        rod_residual=state.x.new_zeros(residual.shape).index_add(1,
            scene.rod.joints.flatten(),result['rod_rows']['residual'].flatten(1,2))
        anticipated,topology,valid=prepare(scene,state,State(tx,tq,start.v,start.omega),baseline,
            context.contacts,result,(residual+joint_residual+rod_residual)*scene.dof)
        result['topology_active']=topology
        result['valid']=torch.cat((result['valid'],valid[:,None]),-1)
        context.anticipated_contacts=int((topology&~result['active']).sum())
    metric=ContactMetric(material.local,[*material.terms,*result['terms'],*anticipated],scene.dof)
    return (residual+result['residual'])*scene.dof,metric,result


def merit(scene,state,start,tx,tq,context,frozen):
    energy=elastic.merit(scene,state,start,tx,tq,context,frozen)
    value=context.rod.terms(rod.strain(scene,state))[2]
    ids=scene.rod.joints;mobile=scene.movable[ids];count=mobile.sum(-1).clamp_min(1)
    for side in (0,1):
        energy=energy.index_add(1,ids[:,side],value*(mobile[:,side].to(value.dtype)/count)[None])
    return energy


def relation_features(scene,state,start,context,result,feedback):
    return torch.cat((elastic.relation_features(scene,state,start,context,result,feedback),
        rod.material_relation_features(scene,state,start,context,result,feedback)),1)


def predictor_domain(scene,start,forecast,config,*,dt=None):
    validate_history(scene,start)
    return (elastic.predictor_domain(scene,selected_history(start,surface.HISTORY_KEYS),forecast,config,dt=dt)
            &rod.predictor_domain(scene,selected_history(start,rod.HISTORY_KEYS),forecast,config,dt=dt))


def path_feasible(scene,start,end):
    return elastic.path_feasible(scene,start,end)&rod.path_feasible(scene,start,end)


def valid_state(scene,state):return elastic.valid_state(scene,state)&rod.valid_state(scene,state)
def retract(scene,state):return state,valid_state(scene,state)
def tangent(scene,state):return None


def finish(scene,start,output,context,accepted):
    material=elastic.finish(scene,selected_history(start,surface.HISTORY_KEYS),output,context,accepted)
    cable=rod.finish(scene,selected_history(start,rod.HISTORY_KEYS),output,context,accepted)
    mask=scene.rod_node_mask[None,:,None]
    return State(output.x,output.q,torch.where(mask,cable.v,material.v),
        torch.where(mask,cable.omega,material.omega),{**material.history,**cable.history})


def diagnostics(scene,result):
    return {**elastic.diagnostics(scene,result),
        'rod_rows_with_nonzero_wrench':int((result['rod_rows']['residual'].abs().amax((-1,-2))>1e-12).sum()),
        'finite_joint_rows_with_nonzero_wrench':int((result['joint_rows']['pair_residual'].abs().amax((-1,-2))>1e-12).sum())}


coarse=rod.coarse
acceptance_groups=elastic.acceptance_groups
internal_dual_small=rod.internal_dual_small
update_internal=rod.update_internal
proposal_scale=elastic.proposal_scale
