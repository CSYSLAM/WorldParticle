"""Compose rod and finite joint relations around exactly one COM inertia."""
import torch

from . import math3d as so3, rod_contact_system as rod, general_joint_system as joint
from .rigid_contact_system import ContactMetric

supports_component_batch=False
contact_safety=rod.contact_safety+'; FINITE_GENERAL_JOINTS; ACCEPTED_SINGLE_AXIS_WINDING'


def begin(scene,start,candidates,dt):
    context=rod.begin(scene,start,candidates,dt)
    context.previous_angle=start.history['joint_angle_prev']
    context.joint_start=scene.joint_coordinates(context.material_start,context.previous_angle)
    context.targets,context.target_rates,context.efforts=scene.joint_controls
    return context


def rows(scene,state,start,context):
    return joint.merge(scene,state,rod.rows(scene,state,start,context),joint.joint_rows(scene,state,context))


def system(scene,state,start,tx,tq,context):
    residual,metric,result=rod.system(scene,state,start,tx,tq,context)
    # The merged dictionary is the complete constitutive residual too.
    result['residual']=result['residual'].index_add(1,scene.rod.joints.flatten(),
        result['rod_rows']['residual'].flatten(1,2))*scene.dof
    finite=joint.joint_rows(scene,state,context)
    result=joint.merge(scene,state,result,finite)
    residual=residual+state.x.new_zeros(residual.shape).index_add(1,
        scene.joint_spec.owners.flatten(),finite['pair_residual'].flatten(1,2))*scene.dof
    return residual,ContactMetric(metric.local,result['terms'],scene.dof),result


def merit(scene,state,start,tx,tq,context,frozen):
    energy=rod.merit(scene,state,start,tx,tq,context,frozen)
    value=joint.joint_energy(scene,state,context,frozen_bound=frozen['joint_rows']['bound_lower'])
    ids=scene.joint_spec.owners;mobile=scene.movable[ids];count=mobile.sum(-1).clamp_min(1)
    for side in (0,1):energy=energy.index_add(1,ids[:,side],value*(mobile[:,side]/count)[None])
    return energy


def relation_features(scene,state,start,context,result,feedback):
    base=rod.relation_features(scene,state,start,context,result,feedback)
    finite=joint.relation_features(scene,state,start,context,result,feedback)
    return torch.cat((base,finite[:,len(context.contacts.owners):]),1)


def predictor_domain(scene,start,forecast,config,*,dt=None):
    # Rod history already contains the full previous accepted COM pose. Use
    # it for finite joint damping and commanded angular branch bounds too;
    # there is no second independent pose history to drift out of sync.
    valid=rod.predictor_domain(scene,start,forecast,config,dt=dt)
    ids=scene.joint_spec.owners;single=scene.joint_spec.angular_mode==1
    command=so3.log(so3.multiply(start.q,so3.conjugate(start.history['rod_previous_q']))).norm(dim=-1)
    command=command*(~scene.movable)[None]
    angle=so3.log(so3.multiply(forecast.q,so3.conjugate(start.q))).norm(dim=-1)
    bound=(angle+command)[:,ids].sum(-1)
    bound=bound+scene.movable[ids].sum(-1)*config.get('unroll',4)*config.get('trust_radius',.1)*config.get('relaxation',.8)
    return valid & (((bound<torch.pi-1e-6)|~single).all(-1)) & torch.isfinite(bound).all(-1)


def diagnostics(scene,result):
    return {**rod.diagnostics(scene,result),
        'finite_joint_rows_with_nonzero_wrench':int((result['joint_rows']['pair_residual'].abs().amax((-1,-2))>1e-12).sum())}


coarse=rod.coarse
acceptance_groups=rod.acceptance_groups
internal_dual_small=rod.internal_dual_small
update_internal=rod.update_internal
valid_state=rod.valid_state
path_feasible=rod.path_feasible
tangent=rod.tangent
retract=rod.retract
finish=rod.finish
