"""Fixed-budget shared primal/dual rigid-contact step, without VBD calls."""
import torch
from . import math3d as so3
from .data import State
from .mixed_physics import predictor,mass_whitening
from .protected_response import protected_update
from .rigid_contact_system import begin,rows,system,coarse,relation_features,merit
from .unified_graph import build_topology


def move(scene,state,update,alpha):
    delta=update*alpha[...,None]*scene.dof
    return State(state.x+delta[...,:3],so3.normalize(so3.multiply(so3.exp(delta[...,3:]),state.q)),state.v,state.omega)


def acceptance_groups(scene,c,depth,travel,*,labels=False):
    # Only a real body pair reachable within this proposed motion shares an
    # acceptance decision. Static world never unites independent bodies.
    possible=(depth[0]+travel[0,c.owners].sum(-1)>=0)&(c.material>0)
    graph=build_topology(scene.kind,scene.movable,scene.object_id,c.owners,torch.full((len(c.owners),),7,device=scene.device,dtype=torch.long),potential_active=possible)
    if labels:return graph.interaction_id
    return [torch.where(graph.interaction_id==i)[0] for i in torch.unique(graph.interaction_id).tolist() if bool(scene.movable[graph.interaction_id==i].any())]


def choose(scene,current,start,update,tx,tq,context,frozen,alphas,*,physics=None,acceptance_labels=None,prepared_geometric=None,geometry_prepared=False,remaining_motion_scale=False):
    c=context.contacts;oracle=scene.geometry_oracle
    energy=merit if physics is None else physics.merit
    grouping=acceptance_groups if physics is None else physics.acceptance_groups
    with torch.no_grad():before=energy(scene,current,start,tx,tq,context,frozen)
    full=move(scene,current,update,torch.ones_like(current.x[...,0]))
    _,travel=oracle.motion(current,full)
    fast=bool(getattr(scene,'fast_rigid_contact',False))
    group_options={}
    if 'topology_active' in frozen:group_options['metric_active']=frozen['topology_active']
    if acceptance_labels is None:
        groups=grouping(scene,c,frozen['raw_depth'],travel,labels=fast,**group_options)
    else:
        groups=acceptance_labels if fast else [torch.where(acceptance_labels==label)[0]
            for label in torch.unique(acceptance_labels[scene.movable]).tolist()]
    if fast:old_group=before.new_zeros(before.shape).index_add(1,groups,before)
    selected=torch.zeros_like(current.x[...,0],dtype=torch.bool);chosen=current
    # Surface/joint merit, retraction and path checks have no persistent
    # side effects. Repair supervision is different: it records every trial,
    # so retain its full schedule even after the selected states are final.
    completed_surface_search=(scene.metadata.get('contract')=='elastic_surface_joint_contact_v38'
        and not (getattr(scene,'learned_contact_repair',False) and hasattr(c,'self_snapshot'))
        and not scene.metadata.get('evaluate_all_line_search_rays',False))
    used=torch.zeros_like(current.x[...,0])
    scaled=prepared_geometric
    if not geometry_prepared and physics is not None and hasattr(physics,'proposal_scale'):
        scaled=physics.proposal_scale(scene,current,update)
    if scaled is not None:
        cap,cap_valid,detail=scaled
        context.geometric_step_caps=getattr(context,'geometric_step_caps',[])+[detail]
        context.geometric_step_valid=getattr(context,'geometric_step_valid',torch.ones_like(cap_valid))&cap_valid
    if remaining_motion_scale:
        from .contact_motion_step import proposal_scale
        effective=update if scaled is None or geometry_prepared else update*scaled[0][...,None]
        motion_cap,motion_cap_valid,motion_detail=proposal_scale(scene,start,current,effective)
        context.motion_step_caps=getattr(context,'motion_step_caps',[])+[motion_detail]
        context.motion_step_valid=getattr(context,'motion_step_valid',torch.ones_like(motion_cap_valid))&motion_cap_valid
    for alpha in alphas:
        trial_scale=torch.full_like(used,alpha)
        if scaled is not None and not geometry_prepared:trial_scale=trial_scale*cap
        if remaining_motion_scale:trial_scale=trial_scale*motion_cap[:,None]
        candidate=move(scene,current,update,trial_scale)
        if getattr(scene,'learned_contact_repair',False):
            from .self_contact_repair import candidate as repair_candidate
            candidate=repair_candidate(scene,candidate,context)
        feasible=torch.ones(len(candidate.x),device=candidate.x.device,dtype=torch.bool)
        if physics is not None:candidate,feasible=physics.retract(scene,candidate)
        if physics is not None and hasattr(physics,'path_feasible'):
            feasible=feasible&physics.path_feasible(scene,current,candidate)
        motion_ok,_=oracle.motion(start,candidate)
        motion_ok=motion_ok&feasible
        if scaled is not None:motion_ok=motion_ok&cap_valid
        if remaining_motion_scale:motion_ok=motion_ok&motion_cap_valid
        with torch.no_grad():value=energy(scene,candidate,start,tx,tq,context,frozen)
        if fast:
            new_group=value.new_zeros(value.shape).index_add(1,groups,value)
            good=(new_group<=old_group+1e-12+old_group.abs()*1e-7)&motion_ok[:,None]&torch.isfinite(new_group)
            if scaled is not None or remaining_motion_scale:
                changed=((candidate.x!=current.x).any(-1)|(candidate.q!=current.q).any(-1))&scene.movable[None]
                changed_groups=torch.zeros_like(changed,dtype=torch.long).index_add(1,groups,changed.long())
                good=good&(changed_groups>0)
            take=good[:,groups]&~selected&scene.movable[None]
        else:
            take=torch.zeros_like(selected)
            for ids in groups:
                old=before[:,ids].sum(-1);new=value[:,ids].sum(-1)
                good=(new<=old+1e-12+old.abs()*1e-7)&motion_ok&torch.isfinite(new)
                if scaled is not None or remaining_motion_scale:
                    changed=(candidate.x[:,ids]!=current.x[:,ids]).any(-1)|(candidate.q[:,ids]!=current.q[:,ids]).any(-1)
                    good=good&changed.any(-1)
                take[:,ids]=good[:,None]&~selected[:,ids]
        chosen=State(torch.where(take[...,None],candidate.x,chosen.x),torch.where(take[...,None],candidate.q,chosen.q),start.v,start.omega)
        feedback_scale=alpha if scaled is None else (alpha*cap if geometry_prepared else trial_scale)
        if remaining_motion_scale:
            feedback_scale=alpha*motion_cap[:,None] if scaled is None else (alpha*cap*motion_cap[:,None] if geometry_prepared else trial_scale)
        selected|=take;used=torch.where(take,feedback_scale,used)
        # This contract's retraction, energy and path queries are pure. Once
        # every movable group has accepted a ray, later rays cannot replace
        # any chosen state or feedback. Keep their full budget for groups
        # which still need a smaller admissible step.
        if (scene.metadata.get('contract')=='elastic_rod_surface_joint_self_contact_v222' or completed_surface_search) and bool((selected|~scene.movable[None]).all()):
            break
    selected=selected|~scene.movable[None]
    return chosen,selected,used


def step(model,scene,start,dt,learned,force,torque,*,physics=None):
    if len(start.x)!=1 or physics is None and start.history is not None:raise ValueError('Initial rigid-contact contract requires batch1 and no cross-step contact warm start')
    begin_rows=begin if physics is None else physics.begin
    evaluate_rows=rows if physics is None else physics.rows
    evaluate_system=system if physics is None else physics.system
    make_coarse=coarse if physics is None else physics.coarse
    encode_relations=relation_features if physics is None else physics.relation_features
    if not hasattr(scene,'geometry_oracle') or not hasattr(scene,'candidates'):raise ValueError('Fresh collision snapshot and motion envelope are required')
    scene.use_exact_inertia=True
    scene.component_batch=bool(model.config.get('rigid_component_batch',False))
    scene.fast_rigid_contact=bool(model.config.get('rigid_contact_singleton_fast_path',False)) if physics is None else bool(model.config.get('general_joint_singleton_fast_path',False))
    scene.fast_rigid_contact=scene.fast_rigid_contact or scene.component_batch
    if physics is not None and not getattr(physics,'supports_rigid_component_batch',True):
        scene.component_batch=bool(model.config.get('mixed_component_batch',False));scene.fast_rigid_contact=scene.component_batch
    if physics is not None and not getattr(physics,'supports_component_batch',True):
        scene.component_batch=False;scene.fast_rigid_contact=False
    tx,tq=predictor(scene,start,dt,force,torque)
    oracle=scene.geometry_oracle;forecast=State(tx,tq,start.v,start.omega)
    predictor_motion_valid,forecast_travel=oracle.motion(start,forecast)
    predictor_physics_valid=torch.ones_like(predictor_motion_valid)
    bounded_target=bool(scene.metadata.get('bounded_motion_forecast_target_only',False))
    if bounded_target and (physics is None or not model.config.get('contact_motion_ray_step',False)):
        raise ValueError('A target-only forecast requires physics and a bounded motion ray on every proposal')
    # A free-flight target may cross the query envelope while its constrained
    # physical solution stays inside it. Every actual proposal still passes
    # the remaining-motion cap, retraction/path checks and the native oracle.
    valid=(torch.isfinite(tx).all((1,2))&torch.isfinite(tq).all((1,2))) if bounded_target else predictor_motion_valid
    repair_enabled=bool(getattr(scene,'learned_contact_repair',False)) and hasattr(scene.candidates,'self_snapshot')
    if physics is not None and not repair_enabled:
        predictor_physics_valid=physics.predictor_domain(scene,start,forecast,model.config,dt=dt)
        valid=valid&predictor_physics_valid
    context=begin_rows(scene,start,scene.candidates,dt)
    current=start;feedback=torch.ones_like(start.x[...,0]);roots=torch.zeros_like(feedback,dtype=torch.bool)
    if repair_enabled:
        from .self_contact_repair import candidate as repair_candidate
        from .mixed_contact_system import path_feasible as material_path
        snapshot=scene.candidates.self_snapshot;n=len(snapshot.reference)
        reference=State(torch.cat((snapshot.reference[None],start.x[:,n:]),1),start.q,start.v,start.omega,start.history)
        current=repair_candidate(scene,current,context)
        predictor_physics_valid=physics.predictor_domain(scene,current,forecast,model.config,dt=dt)&material_path(scene,reference,current)
        # Free-flight is an inertial target. Only accepted/repaired geometry
        # spends the query envelope; a distant target is not itself a step.
        valid=torch.isfinite(tx).all((1,2))&torch.isfinite(tq).all((1,2))&predictor_physics_valid
    frozen=evaluate_rows(scene,current,start,context)
    warm=torch.cat((tx-current.x,so3.log(so3.multiply(tq,so3.conjugate(current.q)))),-1)
    current,_,_=choose(scene,current,start,warm,tx,tq,context,frozen,model.config.get('line_search',[1.,.5,.25,.125,.0625]),physics=physics,remaining_motion_scale=bool(model.config.get('contact_motion_ray_step',False)))
    # Only the chosen state survives the predictor line search. Keeping its
    # initial row dictionary also keeps every unused dense contact Jacobian
    # and metric factor alive for the complete differentiable unroll.
    del frozen
    minimum=torch.ones(len(start.x),device=scene.device,dtype=start.x.dtype)
    dual_only=torch.zeros_like(roots);last_physical=torch.zeros_like(warm)
    rod_retention=scene.metadata.get('contract') in ('rod_shape_contact_com_v40','rod_finite_joint_contact_com_v143','elastic_rod_surface_joint_self_contact_v222') and bool(model.config.get('rod_retain_accepted_primal_iterate',False))
    self_retention=scene.metadata.get('contract') in ('elastic_surface_joint_self_contact_v58','elastic_rod_surface_joint_self_contact_v222') and bool(model.config.get('self_retain_accepted_primal_iterate',False))
    allow_stall=(scene.metadata.get('contract') in ('elastic_shape_contact_v33','elastic_surface_joint_contact_v38','elastic_surface_joint_dynamic_proxy_v300') and bool(model.config.get('allow_internal_stall',False))) or rod_retention or self_retention
    retain_accepted=(allow_stall and bool(model.config.get('retain_accepted_primal_iterate',False))) or rod_retention or self_retention
    progressed=torch.zeros_like(roots) if retain_accepted else None
    if repair_enabled and retain_accepted:progressed[:]=True
    w=mass_whitening(scene,start,dt).detach();eye=torch.eye(6,device=scene.device,dtype=start.x.dtype)
    iterations=[];metric_ray_scales=[];activation_ray_scales=[];separate_ray_scales=[]
    for _ in range(model.config.get('unroll',4)):
        # The previous evaluated row contributes its tangent dual through
        # context; its other readout/metric arrays have no later consumer.
        updated=None
        residual,metric,contact=evaluate_system(scene,current,start,tx,tq,context)
        tangent=None if physics is None else physics.tangent(scene,current)
        if tangent is not None:residual=tangent.dual(residual);valid=valid&tangent.valid
        scene.update_graph(context.contacts.owners,contact.get('topology_active',contact['active'])[0])
        blocks=metric.diagonal;white=w@blocks.detach()@w+torch.diag_embed((~scene.dof).to(w.dtype))[None]+1e-6*eye
        inverse=torch.linalg.inv_ex(white,check_errors=False);baseline=inverse.inverse.detach()
        physical=-(w@baseline@w@residual[...,None])[...,0];last_physical=physical
        if tangent is not None:physical=tangent.primal(physical);last_physical=physical
        relation=encode_relations(scene,current,start,context,contact,feedback)
        from .residual_backward import freeze as freeze_residual
        response_residual=residual.detach() if freeze_residual(model.config,scene.metadata.get('contract')) and torch.is_grad_enabled() else residual
        raw_coarse=make_coarse(scene,current)
        if scene.metadata.get('joint_tree_coarse_modes'):
            from .articulation_coarse import augment
            raw_coarse=augment(scene,current,raw_coarse,metric,response_residual)
        separate=bool(scene.metadata.get('separate_coarse_ray_step',False))
        if separate and not model.config.get('contact_metric_ray_step',False):
            raise ValueError('Separate subspace proposals require the final physical curvature ray check')
        response=protected_update(model,scene,current,metric,raw_coarse,response_residual,w,baseline,relation,physical,dt,learned,return_parts=separate)
        update,coarse_r,ok=response[:3]
        protected_part=response[3] if separate else None
        # The mass tangent can change the energy curvature of a proposal.
        # Scale the actual constrained direction, rather than projecting a
        # previously scaled unconstrained direction afterwards.
        if tangent is not None:update=tangent.primal(update);coarse_r=tangent.primal(coarse_r)
        if separate:
            from .metric_ray_step import scale_proposal
            if tangent is not None:protected_part=tangent.primal(protected_part)
            complement=update-protected_part
            protected_part,protected_valid,protected_detail=scale_proposal(scene,metric,response_residual,protected_part)
            complement,complement_valid,complement_detail=scale_proposal(scene,metric,response_residual,complement)
            update=protected_part+complement;ok=ok&protected_valid&complement_valid
            separate_ray_scales.append({'protected':protected_detail,'complement':complement_detail})
        if model.config.get('contact_metric_ray_step',False):
            from .metric_ray_step import scale_proposal
            update,ray_valid,ray_detail=scale_proposal(scene,metric,response_residual,update)
            ok=ok&ray_valid;metric_ray_scales.append(ray_detail)
        valid=valid&ok&(inverse.info==0).all(-1)&contact['valid'].all(-1)
        extent=torch.cat((update[...,:3]/scene.length,update[...,3:]),-1).norm(dim=-1)
        radius=model.config.get('trust_radius',.1)
        if scene.component_batch:
            from .component_batch import layout
            groups=layout(scene);maximum=extent.new_zeros(len(extent),groups.count).scatter_reduce(1,groups.labels[None].expand_as(extent),extent,reduce='amax',include_self=True)
            scale=(radius/maximum.clamp_min(radius)).clamp_max(1)
            update=update*scale[:,groups.labels,None]
        for ids in (() if scene.component_batch else scene.interaction_entities):
            scale=(radius/extent[:,ids].amax(-1,keepdim=True).clamp_min(radius)).clamp_max(1)
            update=update.index_copy(1,ids,update[:,ids]*scale[...,None])
        update=update*model.config.get('relaxation',.8)
        acceptance_labels=None
        prepared_geometric=None
        if model.config.get('contact_activation_ray_step',False):
            from .contact_activation_step import scale_proposal as activation_scale
            grouping=acceptance_groups if physics is None else physics.acceptance_groups
            if physics is not None and hasattr(physics,'proposal_scale'):
                prepared_geometric=physics.proposal_scale(scene,current,update)
                if prepared_geometric is not None:update=update*prepared_geometric[0][...,None]
            with torch.no_grad():
                full=move(scene,current,update,torch.ones_like(current.x[...,0]))
                _,travel=scene.geometry_oracle.motion(current,full)
                options={'metric_active':contact['topology_active']} if 'topology_active' in contact else {}
                acceptance_labels=grouping(scene,context.contacts,contact['raw_depth'],travel,labels=True,**options)
            update,activation_valid,activation_detail=activation_scale(scene,current,context,contact,metric,response_residual,update,acceptance_labels)
            valid=valid&activation_valid;activation_ray_scales.append(activation_detail)
        choices={'physics':physics,'remaining_motion_scale':bool(model.config.get('contact_motion_ray_step',False))}
        if acceptance_labels is not None:choices.update(acceptance_labels=acceptance_labels,prepared_geometric=prepared_geometric,geometry_prepared=True)
        chosen,selected,used=choose(scene,current,start,update,tx,tq,context,contact,model.config.get('line_search',[1.,.5,.25,.125,.0625]),**choices)
        small=(physical[...,:3].abs().amax(-1)<=scene.length*5e-7)&(physical[...,3:].abs().amax(-1)<=2e-6)
        small=small&(coarse_r[...,:3].abs().amax(-1)<=scene.length*5e-7)&(coarse_r[...,3:].abs().amax(-1)<=2e-6)
        dual_change=(contact['tangent_force']-context.dual_t).norm(dim=-1)/context.rho_t.clamp_min(1e-30)
        internal_small=physics.internal_dual_small(scene,current,context,contact) if physics is not None and hasattr(physics,'internal_dual_small') else torch.ones_like(small)
        if scene.fast_rigid_contact:
            bad=(dual_change>scene.length*5e-7).long()
            owners=context.contacts.owners
            values=bad[:,:,None].expand(-1,-1,owners.shape[1])*(owners>=0)[None]
            counts=(~internal_small).long().index_add(1,owners.clamp_min(0).flatten(),values.flatten(1,2))
            ids=scene.interaction_id
            primal_bad=torch.zeros_like(counts).index_add(1,ids,(~small).long())
            dual_bad=torch.zeros_like(counts).index_add(1,ids,(counts>0).long())
            stationary=primal_bad[:,ids]==0;dual_fixed=dual_bad[:,ids]==0
            dual_only|=stationary&~dual_fixed;selected|=stationary;roots|=stationary&dual_fixed
        else:
            from .contact_stationarity import incident_stationary
            dual_small=incident_stationary(context.contacts.owners,dual_change,internal_small,scene.length*5e-7)
            for ids in scene.interaction_entities:
                stationary=small[:,ids].all(-1,keepdim=True)
                dual_fixed=dual_small[:,ids].all(-1,keepdim=True)
                dual_only[:,ids]|=stationary&~dual_fixed
                selected[:,ids]|=stationary
                roots[:,ids]|=stationary&dual_fixed
        if not allow_stall:valid=valid&selected.all(-1)
        if retain_accepted:progressed|=selected
        iterations.append({'inverse_valid':bool((inverse.info==0).all()),'response_valid':bool(ok.all()),'contact_valid':bool(contact['valid'].all()),'selected':selected.detach().cpu().tolist(),'maximum_physical':float(physical.abs().max()),'maximum_update':float(update.abs().max())})
        current=chosen
        updated=evaluate_rows(scene,current,start,context)
        context.dual_t=updated['tangent_force']
        if physics is not None and hasattr(physics,'update_internal'):physics.update_internal(scene,current,context,updated)
        feedback=torch.where(selected,torch.where(used>0,used,1.),0.)
        minimum=torch.minimum(minimum,feedback[:,scene.movable].amin(-1))
    result=State(current.x,current.q,(current.x-start.x)/dt,so3.log(so3.multiply(current.q,so3.conjugate(start.q)))/dt)
    if allow_stall:valid=valid&(progressed if retain_accepted else selected).all(-1)
    if hasattr(context,'geometric_step_valid'):valid=valid&context.geometric_step_valid
    if hasattr(context,'motion_step_valid'):valid=valid&context.motion_step_valid
    finite=all(bool(torch.isfinite(value).all()) for value in (result.x,result.q,result.v,result.omega))
    valid=valid&finite
    if bounded_target:
        final_motion_valid,final_travel=oracle.motion(start,result)
        valid=valid&final_motion_valid
    final_physics_valid=torch.ones_like(valid)
    if physics is not None:
        final_physics_valid=physics.valid_state(scene,result)
        if repair_enabled:final_physics_valid=final_physics_valid&material_path(scene,reference,result)
        valid=valid&final_physics_valid
    output=State(*(torch.where(valid[:,None,None],getattr(result,key),getattr(start,key)) for key in ('x','q','v','omega')))
    if physics is not None:output=physics.finish(scene,start,output,context,valid)
    physical_gap=context.contacts.physical_margin[None]-updated['raw_depth']
    diagnostic={'accepted':valid,'iterations':iterations,'finite':finite,'min_alpha':minimum,'rejected_entities':~selected,'rejected_physical_step':last_physical,'numerical_root_entities':roots,'dual_only_iterations':dual_only,'minimum_contact_normal_physical_separation':float(physical_gap.min()) if physical_gap.numel() else None,'contacts':len(context.contacts.owners),'collision_generation':context.contacts.generation,'full_vbd_calls':0,'contact_safety':'BOUNDED_MOTION_WITH_EXPANDED_DETECTION_GAP; FINITE_MATERIAL_PENETRATION_ALLOWED','architecture':model.architecture_id}
    diagnostic.update(predictor_motion_valid=predictor_motion_valid.detach(),
        predictor_physics_domain_valid=predictor_physics_valid.detach(),
        predictor_maximum_surface_travel=forecast_travel.detach().amax(-1),
        final_physics_domain_valid=final_physics_valid.detach())
    if bounded_target:
        diagnostic.update(inertial_forecast_is_target_only=True,
            final_motion_domain_valid=final_motion_valid.detach(),
            final_maximum_surface_travel=final_travel[:,scene.movable].amax(-1).detach())
    if repair_enabled:
        repairs=getattr(context,'contact_repairs',[])
        auxiliary=getattr(context,'contact_repair_loss',start.x.new_zeros(()))
        diagnostic.update(contact_acceptance='finite_oriented_overlap_with_local_repair',
            contact_repair_calls=len(repairs),contact_repairs=repairs,
            contact_repair_supervision=1000*auxiliary/max(1,len(repairs)),
            inertial_forecast_is_target_only=True)
    if metric_ray_scales:
        diagnostic['contact_metric_ray_scales']=metric_ray_scales
    if separate_ray_scales:diagnostic['separate_subspace_ray_scales']=separate_ray_scales
    if activation_ray_scales:
        diagnostic['contact_activation_ray_scales']=activation_ray_scales
    if hasattr(context,'motion_step_caps'):
        diagnostic['contact_motion_ray_scales']=context.motion_step_caps
        diagnostic['contact_motion_domain_valid']=bool(context.motion_step_valid.all())
    if hasattr(context.contacts,'source_kind'):
        particle_shape=context.contacts.source_kind==1
        diagnostic['particle_shape_candidate_rows']=int(particle_shape.sum())
        diagnostic['active_particle_shape_rows']=int((updated['active']&particle_shape[None]).sum())
        # Count reciprocal mobile particle/body rows separately from ground.
        owners=context.contacts.owners
        eligible=owners>=0
        if hasattr(context.contacts,'soft'):
            soft=context.contacts.soft.ports
            soft_begin=context.contacts.rigid_count
            eligible=eligible.clone()
            eligible[soft_begin:soft_begin+len(soft.owners)]&=(soft.weights>0).any(1)
        mobile=scene.movable[owners.clamp_min(0)]&eligible
        particles=scene.metadata.get('particle_count',0)
        bodies=scene.metadata.get('body_count',0)
        mobile_particle=(mobile&(owners<particles)).any(-1)
        mobile_body=(mobile&(owners>=particles)&(owners<particles+bodies)).any(-1)
        coupled=particle_shape&mobile_particle&mobile_body
        diagnostic['active_dynamic_particle_body_rows']=int((updated['active']&coupled[None]).sum())
    if hasattr(context,'anticipated_contacts'):diagnostic['anticipated_contact_metric_rows']=context.anticipated_contacts
    if hasattr(context.contacts,'soft_feature_size'):
        diagnostic['active_soft_feature_rows']={str(size):int((updated['active']&(context.contacts.soft_feature_size==size)[None]).sum()) for size in (1,2,3)}
    if physics is not None and hasattr(physics,'contact_safety'):diagnostic['contact_safety']=physics.contact_safety
    if physics is not None and hasattr(physics,'diagnostics'):diagnostic.update(physics.diagnostics(scene,updated))
    if hasattr(context,'geometric_step_caps'):
        diagnostic['geometric_step_caps']=context.geometric_step_caps
        diagnostic['geometric_step_domain_valid']=bool(context.geometric_step_valid.all())
    if allow_stall:diagnostic['internal_stalls_allowed_with_final_selection_required']=True
    if retain_accepted:
        diagnostic['internal_stalls_allowed_with_final_selection_required']=False
        diagnostic['accepted_primal_iterate_required_for_each_entity']=True
        diagnostic['retained_accepted_iterate_entities']=(progressed&~selected&scene.movable[None]).detach()
    if getattr(scene,'request_physical_load_packets',False):
        from .physical_loads import packets_from_rows
        diagnostic['physical_load_packets']=packets_from_rows(scene,context.contacts,updated,start.x) if bool(valid.all()) else None
    if getattr(scene,'request_contact_wrench',False):
        # This is the already evaluated last physical row, before its
        # tangent multiplier is advanced. Re-evaluation would add a dual step.
        if len(context.contacts.owners) and ('force' not in updated or 'torque' not in updated):
            raise NotImplementedError('Weighted/mixed physical loads require real-owner packets')
        empty=current.x.new_zeros(len(start.x),0,2,3)
        diagnostic['contact_wrench']=torch.cat((updated.get('force',empty),updated.get('torque',empty)),-1).detach()
    return output,diagnostic
