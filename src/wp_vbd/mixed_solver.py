"""Fixed-budget shared response solver for the five-family ground fixture."""

import math

import torch
from torch import nn

from . import math3d as so3
from .data import State
from .feasibility import tetrahedron_prefix_coefficients, tetrahedron_polynomial_minimum
from .mixed_physics import incremental_objective, mass_whitening, predictor, residual_system, rod_domain
from .response_operator import RelationResponse


def relation_inputs(scene,state,start,extras,dt,feedback):
    coordinates,joint_j,point_j,slip,distance,active=extras[:6]
    b=state.x.shape[0];r=len(scene.layout.incidence)
    jac=state.x.new_zeros(b,r,9,4,6)
    value=state.x.new_zeros(b,r,9)
    parameter=state.x.new_zeros(b,r,9)
    flag=state.x.new_ones(b,r,1)
    for key in ('tets','triangles'):
        sl,selected=scene.structural_slices[key]
        if not len(selected):continue
        ids=getattr(scene,key)[selected]
        dm=(scene.tet_Dm if key=='tets' else scene.tri_Dm)[selected]
        points=state.x[:,ids]
        f=(points[:,:,1:]-points[:,:,:1]).transpose(-1,-2)@dm
        metric=f.transpose(-1,-2)@f
        count=f.shape[-1]
        m=torch.cat((-dm.sum(-2,keepdim=True),dm),-2)
        derivative=torch.einsum('btia,tsc->btacsi',f,m)
        derivative=derivative+derivative.transpose(2,3)
        jac[:,sl,:count**2,:count+1,:3]=derivative.reshape(b,len(ids),count**2,count+1,3)
        value[:,sl,:count**2]=(metric-torch.eye(count,device=state.x.device)).flatten(-2)
        material=(scene.tet_material if key=='tets' else scene.tri_material)[selected,:2]
        measure=(1/(6*torch.linalg.det(dm))).abs() if key=='tets' else scene.tri_area[selected]
        parameter[:,sl,:2]=torch.log1p(material*measure[:,None])/10
        damping=(scene.tet_material if key=='tets' else scene.tri_material)[selected,2]
        parameter[:,sl,2]=torch.log1p(damping*measure/dt)/10
    sl,selected=scene.structural_slices['bend_edges']
    if len(selected):
        with torch.enable_grad():
            points=state.x[:,scene.bend_edges[selected]].detach().requires_grad_(True)
            n1=so3.normalize(torch.linalg.cross(points[:,:,2]-points[:,:,0],points[:,:,3]-points[:,:,0]))
            n2=so3.normalize(torch.linalg.cross(points[:,:,3]-points[:,:,1],points[:,:,2]-points[:,:,1]))
            edge=so3.normalize(points[:,:,3]-points[:,:,2])
            theta=torch.atan2((torch.linalg.cross(n1,n2)*edge).sum(-1),(n1*n2).sum(-1))
            derivative=torch.autograd.grad(theta.sum(),points)[0]
        jac[:,sl,0,:,:3]=derivative
        from .bending_angle import difference
        value[:,sl,0]=difference(scene,theta.detach(),scene.bend_angle[selected])
        parameter[:,sl,0]=torch.log1p(scene.bend_material[selected,0]*scene.bend_length[selected])/10
        parameter[:,sl,1]=torch.log1p(scene.bend_material[selected,1]*scene.bend_length[selected]/dt)/10
    sl,selected=scene.structural_slices['joints']
    scale=scene.generalized_scale
    jac[:,sl,:6,:2]=joint_j[:,selected]*scale[None,None,:,None,None]
    value[:,sl,:6]=coordinates[:,selected]*scale
    parameter[:,sl,:6]=torch.log1p(scene.joint_stiffness[selected]/scale.square())/10
    sl,selected=scene.structural_slices['attachments']
    jac[:,sl,:3,:2]=extras[7][:,selected]/scene.length
    value[:,sl,:3]=extras[6][:,selected]/scene.length
    parameter[:,sl,0]=torch.log1p(scene.attachment_stiffness[selected]*scene.length**2)/10
    sl=slice(scene.contact_start,r)
    contact_rows=torch.stack((scene.site_mu>0,scene.site_mu>0,torch.ones_like(scene.site_mu,dtype=torch.bool)),-1)
    jac[:,sl,:3,0]=point_j/scene.length*active[:,:,None,None]*contact_rows[None,:,:,None]
    value[:,sl,0]=distance/scene.length
    value[:,sl,1:4]=slip/scene.length
    parameter[:,sl,0]=torch.log1p(scene.contact_ke*scene.length**2)/10
    parameter[:,sl,1]=torch.log1p(scene.contact_kd*scene.length**2/dt)/10
    parameter[:,sl,2]=scene.site_mu
    parameter[:,sl,3]=scene.site_margin/scene.length
    parameter[:,sl,4]=scene.site_radius/scene.length
    flag[:,sl,0]=active.to(flag.dtype)
    velocity=torch.cat((state.v,state.omega),-1)
    rate=RelationResponse.gather(scene.layout,jac,velocity)*dt
    ids=scene.layout.incidence.clamp_min(0);valid=scene.layout.incidence>=0
    mass=(scene.mass[ids]*valid).sum(-1)/valid.sum(-1)
    scalar=torch.cat((flag,state.x.new_full((b,r,1),math.log(dt)),torch.log1p(mass)[None,:,None].expand(b,-1,-1),feedback[:,scene.layout.incidence[:,0],None],state.x.new_ones(b,r,1)),-1)
    features=torch.cat((value,parameter,rate,scalar),-1)
    return jac.detach(),features


def apply_update(scene,state,update,alpha):
    delta=update*alpha[...,None]*scene.dof[None]
    q=so3.normalize(so3.multiply(so3.exp(delta[...,3:]),state.q))
    q=torch.where((scene.movable&scene.body)[None,:,None],q,state.q)
    x=state.x+delta[...,:3]
    if getattr(scene,'use_collective_retraction',False):
        from .collective_retraction import retract
        x=retract(scene,state,delta)
    return State(x,q,state.v,state.omega)


def geometry_scale(scene,state,update):
    owner=scene.site_owner
    arm=so3.rotate(state.q[:,owner],scene.site_local)
    distance=state.x[:,owner,2]+arm[...,2]-scene.site_radius-scene.plane_height
    angular=(update[:,owner,3:].square().sum(-1)+1e-30).sqrt()*scene.site_local.norm(dim=-1)
    bound=((-update[:,owner,2]).clamp_min(0)+angular).clamp_min(1e-20)
    if getattr(scene,'use_collective_retraction',False):
        from .collective_retraction import plane_motion_bound
        curve_bound=plane_motion_bound(scene,state,update)[:,owner]
        bound=torch.maximum(bound,curve_bound)
    allowed=torch.where(distance-bound<0,.99*distance.clamp_min(0)/bound,torch.ones_like(distance)).clamp(0,1)
    scales=state.x.new_ones(state.x.shape[0],scene.n)
    valid=torch.ones(state.x.shape[0],device=state.x.device,dtype=torch.bool)
    for ids,sites in zip(scene.interaction_entities,scene.interaction_sites):
        scale=allowed[:,sites].amin(-1) if len(sites) else state.x.new_ones(state.x.shape[0])
        if len(sites):valid=valid&(distance[:,sites]>=-1e-7).all(-1)
        scales[:,ids]=scale[:,None].expand(-1,len(ids))
    if len(scene.tets):
        coefficients=tetrahedron_prefix_coefficients(scene,state.x,state.x+update[...,:3])
        for ids,rows,first in scene.tet_components:
            alpha=scales[:,first]
            local=tuple(c[:,rows] for c in coefficients)
            for _ in range(6):
                minimum=tetrahedron_polynomial_minimum(local,alpha)
                alpha=torch.where(minimum>=scene.tet_floor,alpha,.5*alpha)
            valid=valid&(tetrahedron_polynomial_minimum(local,alpha)>=scene.tet_floor)
            scales[:,ids]=alpha[:,None].expand(-1,len(ids))
    return scales.detach(),valid


class MixedSolver(nn.Module):
    architecture_id='mixed_response_v1'

    def __init__(self,config):
        super().__init__()
        relation_rows=config.get('checkpoint_relation_ffn_rows',0)
        attention_checkpoint=config.get('checkpoint_attention_blocks',False)
        if type(relation_rows) is not int or relation_rows<0:
            raise ValueError('Nonnegative integer checkpoint_relation_ffn_rows required')
        if type(attention_checkpoint) is not bool:raise ValueError('Boolean checkpoint_attention_blocks required')
        if (relation_rows or attention_checkpoint) and not all(config.get(key,False) for key in ('protected_learning','protected_inverse_replacement','protected_vector_ports','shared_hierarchy')):
            raise ValueError('Nested neural checkpointing requires the shared hierarchical response')
        self.config=config
        self.response=RelationResponse(width=config.get('width',64),rank=config.get('rank',2),learned_baseline=True,metric_normalization=True,material_film=config.get('material_film',False))
        self.coarse_gain=None
        self.coarse_ablation='full'
        if config.get('learned_coarse',False):
            if not config.get('coarse_mode'):raise ValueError('learned_coarse requires a coarse layout')
            self.architecture_id='mixed_coarse_v2'
            width=config.get('width',64)
            self.coarse_gain=nn.Sequential(nn.LayerNorm(width),nn.Linear(width,6))
            nn.init.zeros_(self.coarse_gain[-1].weight);nn.init.constant_(self.coarse_gain[-1].bias,-4.)
        if config.get('coupled_attachments',False):self.architecture_id='mixed_coupled_v3'
        if config.get('material_film',False):self.architecture_id='mixed_conditioned_v4'
        if config.get('coarse_metric')=='objective':self.architecture_id='shared_objective_v5'
        if config.get('coarse_mode')=='bending':self.architecture_id='shared_bending_v7'
        if config.get('joint_history'):self.architecture_id='shared_history_v9'
        if config.get('rod_cycle_modes'):self.architecture_id='shared_cycle_v10'
        if config.get('rod_boundary_lift'):self.architecture_id='shared_boundary_v11'
        if config.get('rod_chain_shear_modes'):self.architecture_id='shared_chain_v12'
        if config.get('rod_curvature_ports'):self.architecture_id='shared_curvature_v13'
        if config.get('rod_geometry_fp64'):self.architecture_id='shared_precision_v14'
        if config.get('rod_curvature_flexure'):self.architecture_id='shared_flexure_v15'
        self.protected_response=None
        if config.get('protected_learning'):
            if not config.get('coarse_mode'):raise ValueError('Protected response requires physical coarse columns')
            from .protected_response import InvariantPortResponse
            self.protected_response=InvariantPortResponse(config.get('protected_width',32),config.get('protected_factor_bound',.4))
            self.architecture_id='shared_protected_v17'
            if config.get('protected_inverse_replacement'):
                from .inverse_response import InversePortResponse
                self.protected_response=InversePortResponse(config.get('protected_width',32),ridge=config.get('protected_inverse_ridge',1e-8))
                self.architecture_id='shared_inverse_v18'
                if config.get('protected_vector_ports'):
                    from .vector_response import VectorInverseResponse
                    self.protected_response=VectorInverseResponse(config.get('protected_width',32),ridge=config.get('protected_inverse_ridge',1e-8))
                    self.architecture_id='shared_vector_v19'
                    if config.get('shared_hierarchy'):
                        from .hierarchical_response import HierarchicalVectorResponse
                        self.protected_response=HierarchicalVectorResponse(config.get('protected_width',32),
                            maximum_super_tokens=config.get('maximum_super_tokens',128),
                            hierarchy_heads=config.get('hierarchy_heads',4),
                            hierarchy_encoder_layers=config.get('hierarchy_encoder_layers',3),
                            hierarchy_decoder_layers=config.get('hierarchy_decoder_layers',2),
                            ridge=config.get('protected_inverse_ridge',1e-8))
                        self.architecture_id='shared_hierarchy_v20'
        if config.get('shared_constitutive_coarse'):
            if not config.get('shared_hierarchy'):raise ValueError('Shared constitutive coarse response requires the typed hierarchy')
            from .constitutive_coarse import ConstitutiveCoarseGain
            self.protected_response.coarse_response=ConstitutiveCoarseGain(config.get('protected_width',32),scope=config.get('constitutive_coarse_scope','all'),protect_complete_response=config.get('protected_complete_response',False))
            self.architecture_id='shared_constitutive_v21'
        if self.protected_response is not None:
            self.protected_response.compact_relations=bool(config.get('compact_neural_relations',False))
            self.protected_response.encoder_checkpointing=bool(config.get('neural_encoder_checkpointing',False))
            if hasattr(self.protected_response,'hierarchy'):
                self.protected_response.hierarchy.compact_relations=self.protected_response.compact_relations
                self.protected_response.hierarchy.relation_condition.checkpoint_rows=relation_rows
                for layer in (*self.protected_response.hierarchy.encoder,*self.protected_response.hierarchy.decoder):
                    layer.checkpoint_block=attention_checkpoint
        self.register_buffer('trial_alphas',torch.tensor(config.get('line_search',[1.,.5,.25])),persistent=False)
        self.objective=incremental_objective
        self.terminal_velocity = None
        if config.get('terminal_velocity_correction', False):
            from .terminal_velocity import TerminalVelocity
            width=config.get('protected_width',32) if config.get('shared_terminal_velocity',False) else None
            self.terminal_velocity = TerminalVelocity(shared_width=width)
            if width is not None:self.protected_response.shared_velocity_features=True
        self.elastic_shape_head = None
        if config.get('learned_elastic_shape_direction', False):
            from .elastic_shape_modes import ElasticShapeDirectionHead
            self.elastic_shape_head = ElasticShapeDirectionHead(
                width=config.get('elastic_shape_width',64),
                maximum_fraction=config.get('elastic_shape_maximum_fraction',.08),
                analytic_prior_fraction=config.get('elastic_shape_analytic_prior_fraction',.001),
                learned_coefficient_scale=config.get('elastic_shape_learned_coefficient_scale',1.),
                feature_version=config.get('elastic_shape_feature_version',1),
                initialization=config.get('elastic_shape_head_initialization','zero'),
                local_heads=config.get('elastic_shape_local_heads',0))
            if config.get('elastic_shape_local_heads',0):self.architecture_id='shared_local_elastic_v22'
        self.contact_tangent_head=None
        if config.get('learned_contact_tangent_seed',False):
            width=config.get('contact_tangent_width',32)
            self.contact_tangent_head=nn.Sequential(nn.Linear(7,width),nn.SiLU(),
                nn.Linear(width,width),nn.SiLU(),nn.Linear(width,2))
            nn.init.normal_(self.contact_tangent_head[-1].weight,std=.01)
            nn.init.zeros_(self.contact_tangent_head[-1].bias)
            self.architecture_id='shared_contact_memory_v23'
        if config.get('analytic_predictor_rounds',0):
            if self.elastic_shape_head is not None:
                raise ValueError('Analytic predictor uses physical elastic directions; disable learned elastic heads')
            self.architecture_id='shared_analytic_predictor_v24'
        if config.get('compiled_objective',False):
            self.objective=torch.compile(incremental_objective,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})

    def forward(self,scene,start,dt,*,learned=True,force=None,torque=None):
        self._elastic_shape_supervision_terms=[]
        if (getattr(self,'step_saved_tensors_on_cpu',False) and self.training
                and torch.is_grad_enabled() and start.x.is_cuda):
            # Offload autograd's saved copies, preserving the actual state,
            # computation, dtype and complete contact graph. Nested encoder
            # and contact checkpoints retain their own execution policy.
            with torch.autograd.graph.save_on_cpu(pin_memory=False):
                return self._step_with_velocity(scene,start,dt,learned,force,torque)
        return self._step_with_velocity(scene,start,dt,learned,force,torque)

    def _step_with_velocity(self,scene,start,dt,learned,force,torque):
        shared_velocity=bool(self.config.get('shared_terminal_velocity',False))
        if shared_velocity:self.protected_response.last_latent=None
        result, diagnostic = self._forward_step(scene,start,dt,learned=learned,force=force,torque=torque)
        if self.training and self._elastic_shape_supervision_terms:
            diagnostic['elastic_shape_supervision']=torch.stack(self._elastic_shape_supervision_terms).mean()
        if learned and self.terminal_velocity is not None:
            features=self.protected_response.last_latent if shared_velocity else None
            result = self.terminal_velocity(scene,start,result,dt,diagnostic['accepted'],features)
        if shared_velocity:self.protected_response.last_latent=None
        return result, diagnostic

    def _forward_step(self,scene,start,dt,*,learned=True,force=None,torque=None):
        from .self_contact_repair import configure
        configure(scene,self.config)
        compact=bool(self.config.get('compact_frozen_contact_metric',False))
        if bool(getattr(scene,'compact_frozen_contact_metric',False))!=compact:
            if hasattr(scene,'_protected_topology'):delattr(scene,'_protected_topology')
        scene.compact_frozen_contact_metric=compact
        scene.contact_row_checkpointing=bool(self.config.get('contact_row_checkpointing',False))
        scene.contact_structural_support=bool(self.config.get('contact_structural_support',False))
        from .self_contact_refresh import enabled as refresh_self_geometry
        scene.self_contact_refresh_geometry=refresh_self_geometry(self.config,scene.metadata.get('contract'))
        scene.compact_neural_relations=bool(self.config.get('compact_neural_relations',False))
        prestress=bool(self.config.get('elastic_prestress_metric',False))
        for target in (scene,getattr(scene,'elastic_scene',None),getattr(scene,'objective_scene',None)):
            if target is None:continue
            if bool(getattr(target,'elastic_prestress_metric',False))!=prestress:
                for key in ('coarse_layout','_protected_topology','_hierarchy_plans','_transport_plans','_vector_active_components','_hierarchy_material_masks'):
                    if hasattr(target,key):delattr(target,key)
            target.elastic_prestress_metric=prestress
            target.periodic_bending=bool(self.config.get('periodic_bending',False))
        if scene.metadata.get('contract') in ('rod_shape_contact_com_v40','rod_finite_joint_contact_com_v143','elastic_rod_surface_joint_self_contact_v222'):
            scene.rod_force_ports=bool(self.config.get('rod_contact_force_ports',False))
            scene.rod_prescribed_inputs=bool(self.config.get('rod_prescribed_fresh_inputs',False))
            scene.rod_caller_pose_history=bool(self.config.get('rod_caller_pose_history',False))
            scene.rod_memory_features=bool(self.config.get('rod_dahl_history_features',False))
            scene.rod_gravity_ports=bool(self.config.get('rod_gravity_material_ports',False))
            if scene.metadata.get('prescribed_geometry_removal_training_schema') and not self.config.get('rod_prescribed_removal_events',False):
                raise ValueError('Prescribed geometry-removal data requires the explicit event contract')
            if (scene.metadata.get('prescribed_rod_training_schema') or scene.metadata.get('zero_Dahl_prescribed_fresh_training_schema')) and not scene.rod_prescribed_inputs:
                raise ValueError('Prescribed rod data requires its explicit fresh-input model configuration')
            from .rigid_contact_solver import step
            from . import rod_contact_system as physics
            if scene.metadata.get('contract')=='elastic_rod_surface_joint_self_contact_v222':
                if not self.config.get('elastic_rod_self_contact',False) or not self.config.get('particle_self_contact_v58',False):
                    raise ValueError('Elastic/native-ROD composition requires explicit shared model configuration')
                scene.predictive_contact_metric=bool(self.config.get('predictive_contact_metric',False))
                scene.self_geometric_step_cap=bool(self.config.get('self_geometric_step_cap',False))
                from . import elastic_rod_system as physics
            if scene.metadata.get('contract')=='rod_finite_joint_contact_com_v143':
                from . import rod_joint_system as physics
            return step(self,scene,start,dt,learned,force,torque,physics=physics)
        if scene.metadata.get('requires_self_contact_assembly') and scene.metadata.get('contract')!='elastic_surface_joint_self_contact_v58':
            raise ValueError('Static material export requires the complete self-contact assembly')
        if scene.metadata.get('requires_finite_joint_assembly') and scene.metadata.get('contract') not in ('elastic_surface_joint_contact_v38','elastic_surface_joint_self_contact_v58','elastic_surface_joint_dynamic_proxy_v300'):
            raise ValueError('Static material export requires the complete finite-joint assembly')
        if scene.metadata.get('contract')=='elastic_surface_joint_self_contact_v58':
            if not self.config.get('particle_self_contact_v58',False):
                raise ValueError('Self-contact requires its explicit experimental model configuration')
            scene.predictive_contact_metric=bool(self.config.get('predictive_contact_metric',False))
            scene.self_geometric_step_cap=bool(self.config.get('self_geometric_step_cap',False))
            from .rigid_contact_solver import step
            from . import self_contact_system as physics
            return step(self,scene,start,dt,learned,force,torque,physics=physics)
        if scene.metadata.get('contract') in ('elastic_surface_joint_contact_v38','elastic_surface_joint_dynamic_proxy_v300'):
            if scene.metadata.get('contract')=='elastic_surface_joint_dynamic_proxy_v300':
                if (not hasattr(scene,'proxy_mask') or scene.proxy_mask.shape!=(scene.n,)
                    or not hasattr(scene,'owned_response') or bool((scene.proxy_mask&scene.owned_response).any())):
                    raise ValueError('Explicit disjoint native proxy/primary ownership required')
            scene.predictive_contact_metric=bool(self.config.get('predictive_contact_metric',False))
            from .rigid_contact_solver import step
            from . import surface_joint_system as physics
            return step(self,scene,start,dt,learned,force,torque,physics=physics)
        if scene.metadata.get('contract')=='elastic_shape_contact_v33':
            scene.predictive_contact_metric=bool(self.config.get('predictive_contact_metric',False))
            from .rigid_contact_solver import step
            from . import mixed_contact_system as physics
            return step(self,scene,start,dt,learned,force,torque,physics=physics)
        if scene.metadata.get('contract')=='rigid_joint_contact_finite_v26':
            from .rigid_contact_solver import step
            from . import general_joint_system as physics
            return step(self,scene,start,dt,learned,force,torque,physics=physics)
        if scene.metadata.get('contract')=='rigid_contact_primal_dual_v21':
            from .rigid_contact_solver import step
            return step(self,scene,start,dt,learned,force,torque)
        if scene.metadata.get('history_contract') == 'rod_compliant_alm_v1':
            if not self.config.get('joint_history'):raise ValueError('Checkpoint does not support joint material history')
            from .rod_solver import step
            return step(self, scene, start, dt, learned, force, torque)
        scene.use_collective_retraction=self.config.get('collective_retraction',False)
        protected=self.config.get('protected_learning',False)
        scene.use_bending_retraction=self.config.get('coarse_mode')=='bending' and scene.use_collective_retraction
        scene.use_exact_inertia=self.config.get('exact_inertia',False)
        if self.config.get('coarse_mode'):
            from .mixed_coarse import MixedCoarseLayout
            if not hasattr(scene,'coarse_layout') or scene.coarse_layout.mode!=self.config['coarse_mode']:
                # Static topology preparation happens on the first eager/warm
                # call, before CUDA capture; it never reads teacher states.
                scene.coarse_layout=MixedCoarseLayout(scene,self.config['coarse_mode'])
        tx,tq=predictor(scene,start,dt,force,torque)
        w=mass_whitening(scene,start,dt).detach()
        # The inertial predictor is a target, not an unchecked starting state.
        # Commit its analytic horizontal translation modes first (plane/tet
        # invariants), then guard the remaining translational/rotation path.
        x=start.x
        for ids,_ in scene.floating_components:
            mass=scene.mass[ids]
            shift=((tx[:,ids]-start.x[:,ids])*mass[None,:,None]).sum(1)/mass.sum()
            shift=shift*scene.axis_xy
            x=x.index_copy(1,ids,x[:,ids]+shift[:,None])
        legal=State(x,start.q,start.v,start.omega)
        warm=torch.cat((tx-x,so3.log(so3.multiply(tq,so3.conjugate(start.q)))),-1)*scene.dof[None]
        warm_scale,warm_valid=geometry_scale(scene,legal,warm)
        current=apply_update(scene,legal,warm,warm_scale)
        if self.config.get('guard_predictor_merit',False):
            # A stiff shell's inertial predictor can inject enormous strain
            # energy before the bounded learned iterations even begin. Admit
            # the predictor only through the same incremental merit contract.
            # Failure to initialize from it leaves the legal starting state;
            # the subsequent K physical proposals must still pass normally.
            from .mixed_physics import contact_kinematics
            _,_,_,distance,selected=contact_kinematics(scene,start,start)
            load=scene.contact_ke*(scene.site_margin-distance).clamp_min(0)*selected
            with torch.no_grad():initial_merit=self.objective(scene,legal,start,tx,tq,dt,load)
            current=legal;chosen=torch.zeros_like(start.x[...,0],dtype=torch.bool)
            if self.config.get('coarse_velocity_predictor',False) and scene.bending_plans:
                # Carry admissible collective momentum before the K response
                # updates. A cold solve at every small h otherwise damps that
                # velocity by its finite iteration error on every timestep.
                velocity=torch.cat((start.v*dt,torch.zeros_like(start.omega)),-1)
                coast=torch.zeros_like(velocity)
                for plan in scene.bending_plans:
                    _,_,_,remainder=plan.decompose(scene,legal,velocity)
                    projected=velocity[:,plan.mobile,:3]-remainder
                    coast=coast.index_copy(1,plan.mobile,torch.cat((projected,torch.zeros_like(projected)),-1))
                coast_scale,coast_valid=geometry_scale(scene,legal,coast)
                candidate=apply_update(scene,legal,coast,coast_scale)
                with torch.no_grad():merit=self.objective(scene,candidate,start,tx,tq,dt,load)
                select=torch.zeros_like(chosen)
                for plan in scene.bending_plans:
                    ids=plan.entities;good=merit[:,ids].sum(-1)<=initial_merit[:,ids].sum(-1)
                    select[:,ids]=good[:,None].expand(-1,len(ids))
                current=State(torch.where(select[...,None],candidate.x,current.x),torch.where(select[...,None],candidate.q,current.q),start.v,start.omega)
                chosen=chosen|select;warm_valid=warm_valid&coast_valid
            for alpha in self.config.get('line_search',[1.,.5,.25,.125,.0625]):
                candidate=apply_update(scene,legal,warm,warm_scale*alpha)
                with torch.no_grad():merit=self.objective(scene,candidate,start,tx,tq,dt,load)
                select=torch.zeros_like(chosen)
                for ids in scene.interaction_entities:
                    good=merit[:,ids].sum(-1)<=initial_merit[:,ids].sum(-1)
                    select[:,ids]=good[:,None].expand(-1,len(ids))
                select=select&~chosen
                current=State(torch.where(select[...,None],candidate.x,current.x),torch.where(select[...,None],candidate.q,current.q),start.v,start.omega)
                chosen=chosen|select
        accepted=warm_valid&rod_domain(scene,start)&rod_domain(scene,current)
        minimum=start.x.new_ones(start.x.shape[0])
        rejected_entities=torch.zeros(start.x.shape[0],scene.n,device=start.x.device,dtype=torch.bool)
        numerical_roots=torch.zeros_like(rejected_entities)
        rejected_step=torch.zeros_like(start.x).repeat(1,1,2)
        rejected_cost=start.x.new_zeros(start.x.shape[0],scene.n)
        feedback=start.x.new_ones(start.x.shape[0],scene.n)
        eye=torch.eye(6,device=start.x.device)
        predictor_rounds=int(self.config.get('analytic_predictor_rounds',0))
        phases=[]
        for iteration in range(predictor_rounds+self.config.get('unroll',4)):
            iteration_learned=learned and iteration>=predictor_rounds
            phases.append({'phase':'neural_correction' if iteration_learned else 'analytic_predictor'})
            assembled=residual_system(scene,current,start,tx,tq,dt,batched=self.config.get('batched_jacobian',True),return_metric_base=bool(self.config.get('coarse_mode')))
            residual,blocks,extras=assembled[:3]
            if self.config.get('frozen_residual_backward',False) and torch.is_grad_enabled():
                # Explicit approximate backward for stiff material solves.
                # Forward values, geometry, forces and acceptance are exact
                # and unchanged. Short supervised rollout trains the response
                # coefficients without multiplying stiff residual Jacobians.
                residual=residual.detach()
            white=w.double()@blocks.detach().double()@w.double() if protected else w@blocks.detach()@w
            white=white+torch.diag_embed((~scene.dof).to(white.dtype))[None]+1e-6*eye.to(white.dtype)
            baseline=torch.linalg.inv_ex(white,check_errors=False).inverse.detach()
            r=(w@residual[...,None]).squeeze(-1)
            physical=(-(w.double()@baseline@w.double()@residual.double()[...,None]).squeeze(-1)).to(residual.dtype) if protected else -(w@baseline@r[...,None]).squeeze(-1)
            coarse=None
            jac=relation=None
            entity=encoded=None
            if iteration_learned and not protected:
                entity=torch.cat(((current.x-scene.origins)/scene.length,start.v*dt/scene.length,(tx-current.x)/scene.length,so3.log(so3.multiply(tq,so3.conjugate(current.q))),torch.asinh(r/.1),physical*scene.generalized_scale),-1)
            response_residual=residual
            if self.config.get('coarse_mode'):
                from .mixed_metric import FrozenMixedMetric
                from .mixed_coarse import PreparedMixedCoarse
                jac,relation=relation_inputs(scene,current,start,extras,dt,feedback)
                metric=FrozenMixedMetric(scene,current,start,tx,tq,dt,extras,jac,base=assembled[3],elastic_mode=self.config.get('coarse_metric','projected'))
                gains=None
                if iteration_learned and not protected and self.coarse_gain is not None:
                    encoded=self.response.encode_entities(scene.layout,entity,relation)
                    gains=[self.coarse_gain(encoded[:,plan.ids].mean(1)) for plan in scene.coarse_layout.plans]
                    gains=[1+.5*gain.tanh() for gain in gains] if self.config.get('coarse_gain_mode')=='centered' else [1.9*gain.sigmoid() for gain in gains]
                    if self.coarse_ablation=='disabled':gains=[gain*0 for gain in gains]
                coarse=PreparedMixedCoarse(scene,current,metric,scene.coarse_layout,gains,self.config.get('coarse_composition','balanced'))
                accepted=accepted&coarse.valid
                if not protected:coarse_r,response_residual=coarse.residual_complement(residual)
            if protected:
                from .protected_response import protected_update
                update,coarse_value,valid=protected_update(self,scene,current,metric,coarse,residual,w,baseline,relation,physical,dt,iteration_learned)
                coarse_r=coarse_value[...,None]
                accepted=accepted&valid
            elif iteration_learned:
                if jac is None:jac,relation=relation_inputs(scene,current,start,extras,dt,feedback)
                neural=self.response(scene.layout,response_residual,w,baseline,jac,entity,relation,encoded=encoded)
                gain=self.config.get('response_gain',.1)
                response_physical=-(w@baseline@w@response_residual[...,None]).squeeze(-1) if coarse is not None else physical
                update=response_physical+gain*(neural-response_physical)
            else:update=-(w@baseline@w@response_residual[...,None]).squeeze(-1) if coarse is not None else physical
            if coarse is not None and not protected:update=coarse.combine_negative_local(coarse_r,update)
            # Bounded trust radius is numerical, not a user material parameter.
            extent=torch.cat((update[...,:3]/scene.length,update[...,3:]),-1).norm(dim=-1,keepdim=True)
            trust=torch.ones_like(extent)
            for ids in scene.interaction_entities:
                radius=self.config.get('trust_radius',.1)
                # min(radius / extent, 1) has the same forward value with
                # the denominator clamped to radius. A tiny denominator can
                # overflow reciprocal's FP32 backward even in the saturated
                # branch, producing 0 * inf for an exactly solved component.
                scale=(radius/extent[:,ids].amax(1,keepdim=True).clamp_min(radius)).clamp_max(1)
                trust[:,ids]=scale.expand(-1,len(ids),-1)
            update=update*trust
            update=update*self.config.get('relaxation',.8)
            geometry,valid=geometry_scale(scene,current,update.detach())
            accepted=accepted&valid
            with torch.no_grad():
                lagged_load=scene.contact_ke*(scene.site_margin-extras[4]).clamp_min(0)*extras[5]
                previous=self.objective(scene,current,start,tx,tq,dt,lagged_load)
            chosen=current
            selected=torch.zeros(start.x.shape[0],scene.n,device=start.x.device,dtype=torch.bool)
            used=torch.zeros_like(feedback)
            best_delta=torch.full_like(feedback,float('inf'))
            if self.config.get('vectorized_trials',True):
                # Trial paths are independent. Evaluate their complete local
                # models in one batch instead of serially launching the same
                # small kernels five times; retain the exact first-pass rule.
                count=len(self.trial_alphas);batch=start.x.shape[0]
                steps=self.trial_alphas[:,None,None]*geometry[None]
                delta=update[None]*steps[...,None]*scene.dof[None,None]
                trial_x=current.x[None]+delta[...,:3]
                trial_q=so3.normalize(so3.multiply(so3.exp(delta[...,3:]),current.q[None]))
                trial_q=torch.where((scene.movable&scene.body)[None,None,:,None],trial_q,current.q[None])
                flat=lambda v:v.expand(count,*v.shape).reshape(count*batch,*v.shape[1:])
                if scene.use_collective_retraction:
                    from .collective_retraction import retract
                    trial_x=retract(scene,State(flat(current.x),flat(current.q),flat(start.v),flat(start.omega)),delta.flatten(0,1)).reshape(count,batch,scene.n,3)
                trials=State(trial_x.flatten(0,1),trial_q.flatten(0,1),flat(start.v),flat(start.omega))
                with torch.no_grad():
                    diagnostic=State(trials.x.detach(),trials.q.detach(),trials.v,trials.omega)
                    initial=State(flat(start.x),flat(start.q),flat(start.v),flat(start.omega))
                    all_merit=self.objective(scene,diagnostic,initial,flat(tx),flat(tq),dt,flat(lagged_load)).reshape(count,batch,scene.n)
            # Evaluate a lagged-friction incremental model, never mechanical
            # energy or an assumed universal residual-monotonicity rule.
            for trial,alpha in enumerate(self.config.get('line_search',[1.,.5,.25])):
                step=geometry*alpha
                if self.config.get('vectorized_trials',True):
                    candidate=State(trial_x[trial],trial_q[trial],start.v,start.omega)
                    merit=all_merit[trial]
                else:
                    candidate=apply_update(scene,current,update,step)
                    with torch.no_grad():
                        diagnostic_state=State(candidate.x.detach(),candidate.q.detach(),start.v,start.omega)
                        merit=self.objective(scene,diagnostic_state,start,tx,tq,dt,lagged_load)
                choose=torch.zeros_like(selected)
                for ids in scene.interaction_entities:
                    delta=(merit[:,ids].sum(-1)-previous[:,ids].sum(-1)).to(best_delta.dtype)
                    best_delta[:,ids]=torch.minimum(best_delta[:,ids],delta[:,None].expand(-1,len(ids)))
                    good=(merit[:,ids].sum(-1)<=previous[:,ids].sum(-1)*(1+1e-5)+1e-10)&(step[:,ids].amax(-1)>1e-9)
                    choose[:,ids]=good[:,None].expand(-1,len(ids))
                choose=choose&~selected
                chosen=State(torch.where(choose[...,None],candidate.x,chosen.x),torch.where(choose[...,None],candidate.q,chosen.q),start.v,start.omega)
                used=torch.where(choose,step,used)
                selected=selected|choose
            # A candidate already solved to the FP32 coordinate resolution can
            # have no representable descending neighbour. Check the FULL
            # preconditioned residual, per interaction component, before
            # accepting a no-op internal iteration. This is not a failed-step
            # fallback or a substitute for requested physical-time advancement.
            tolerance=self.config.get('root_tolerance',[5e-7,2e-6])
            small=(physical[...,:3].abs().amax(-1)<=scene.length*tolerance[0])&(physical[...,3:].abs().amax(-1)<=tolerance[1])
            if coarse is not None and self.config.get('require_coarse_root',False):
                response=coarse_r[...,0]
                small=small&(response[...,:3].abs().amax(-1)<=scene.length*tolerance[0])&(response[...,3:].abs().amax(-1)<=tolerance[1])
            for ids in scene.interaction_entities:
                root=small[:,ids].all(-1,keepdim=True).expand(-1,len(ids))&~selected[:,ids]
                numerical_roots[:,ids]=numerical_roots[:,ids]|root
                used[:,ids]=torch.where(root,torch.ones_like(used[:,ids]),used[:,ids])
                selected[:,ids]=selected[:,ids]|root
            # Retaining the candidate for a failed internal line search is
            # diagnostic; it is not counted as a successful physical timestep.
            accepted=accepted&selected.all(-1)
            rejected_entities=rejected_entities|~selected
            rejected_step=torch.where((~selected)[...,None],physical,rejected_step)
            rejected_cost=torch.where(~selected,best_delta,rejected_cost)
            current=chosen
            # With zero tangential contact coefficient, a floating object's
            # horizontal COM is an exact inertial mode. Enforce that physical
            # nullspace rather than manufacturing numerical sliding friction.
            # This common translation preserves every primitive-plane distance
            # and tet determinant in the declared uncoupled-ground scope.
            x=current.x
            for ids,sites in scene.floating_components:
                mass=scene.mass[ids]
                shift=((tx[:,ids]-x[:,ids])*mass[None,:,None]).sum(1)/mass.sum()
                zero=(scene.site_mu[sites]==0).all()
                shift=shift*scene.axis_xy*zero
                x=x.index_copy(1,ids,x[:,ids]+shift[:,None])
            current=State(x,current.q,start.v,start.omega)
            accepted=accepted&rod_domain(scene,current)
            feedback=used
            minimum=torch.minimum(minimum,used.amin(-1))
        result=State(current.x,current.q,(current.x-start.x)/dt,so3.log(so3.multiply(current.q,so3.conjugate(start.q)))/dt)
        finite=torch.stack([torch.isfinite(value).flatten(1).all(-1) for value in (result.x,result.q,result.v,result.omega)]).all(0)
        return result,{'accepted':accepted&finite,'iterations':phases,'min_alpha':minimum,'rejected_entities':rejected_entities,'rejected_physical_step':rejected_step,'rejected_model_cost_change':rejected_cost,'numerical_root_entities':numerical_roots,'contact_safety':'ANALYTIC_PLANE_ROTATION_BOUND','architecture':self.architecture_id}
