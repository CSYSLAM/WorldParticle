"""Experimental rigid COM scene with structural/drive/limit graph rows.

This separate admission path leaves the already trained contact adapter
unchanged. ROD, particles, tendons and external ownership fail closed.
"""
import numpy as np
import torch
from . import math3d as so3
from .data import State
from .general_joint_export import export_spec
from .general_joint_rows import coordinates
from .rigid_contact_scene import RigidContactScene
from .unified_graph import build_topology


def export_model(model,configuration,identifier,newton,dt,device='cpu',*,native_contact_export=False,prescribed_motion=False,packed_worlds=False):
    if model.particle_count or model.spring_count or model.attachment_particle_body_count or getattr(model,'tendon_count',0):raise ValueError('General joint prototype currently requires rigid-only assets without tendons')
    if configuration['rigid_contact_history'] or configuration['integrate_with_external_rigid_solver']:raise ValueError('Persistent contact and external ownership have separate admission gates')
    flags=model.body_flags.numpy()
    if not prescribed_motion and np.any(flags!=int(newton.BodyFlags.DYNAMIC)):raise ValueError('Prescribed/disabled body motion needs a separate kinematic path; inverse mass alone does not define ownership')
    if prescribed_motion and np.any(~np.isin(flags,[int(newton.BodyFlags.DYNAMIC),int(newton.BodyFlags.KINEMATIC)])):raise ValueError('Proxy/external body ownership has a separate admission gate')
    from .mimic_constraint import export_spec as export_mimic
    spec=export_spec(model,configuration,newton,device,allow_mimic=True)
    mimic=export_mimic(model,spec,newton,device)
    values={name:getattr(model,source).numpy().copy() for name,source in [('mass','body_mass'),('inertia','body_inertia'),('inverse_inertia','body_inv_inertia'),('body_com','body_com'),('rest_pose','body_q'),('inverse_mass','body_inv_mass'),('gravity','gravity')]}
    gravity=values['gravity'].reshape(-1,3)
    if not packed_worlds and len(gravity)!=1:raise ValueError('Packed multi-world state is not implemented')
    values['gravity']=gravity[0]
    bodies=model.shape_body.numpy();radius=model.shape_collision_radius.numpy();offset=model.shape_transform.numpy()[:,:3];margin=model.shape_margin.numpy();bounds=np.zeros(model.body_count,np.float64)
    for shape,body in enumerate(bodies):
        if body>=0:bounds[body]=max(bounds[body],np.linalg.norm(offset[shape]-values['body_com'][body])+radius[shape]+float(margin[shape]))
    values['body_radius']=bounds;values['reference_length']=np.asarray(max(np.ptp(values['rest_pose'][:,:3],axis=0).max(),2*bounds.max(),.1))
    alpha=configuration.get('rigid_avbd_contact_alpha')
    if alpha is None:alpha=configuration.get('rigid_avbd_alpha') or 0.
    metadata={'contract':'rigid_joint_contact_finite_v26','family':'mixed','official_demo':identifier,'dt':dt,'particle_count':0,'body_count':model.body_count,'contact_alpha':alpha,'friction_epsilon':configuration['friction_epsilon'],'solver_configuration':configuration,'contacts':'rigid_shape','body_position_representation':'com','cross_step_contact_history':False,'joint_history_contract':'single_axis_winding_v26','scope':'Experimental objective finite joint constitutive fixed point, stateless Coulomb contacts; source linear projection/damping approximation differs.'}
    if packed_worlds:
        from .packed_world import export
        world_values,world_metadata=export(model);values.update(world_values);metadata.update(world_metadata)
    if len(mimic.pairs):metadata['ideal_mimic_contract']='scalar_mass_tangent_v27'
    if native_contact_export:metadata['native_contact_export']='expanded_rigid_stateless_v29'
    if prescribed_motion and np.any(flags==int(newton.BodyFlags.KINEMATIC)):
        metadata['prescribed_pose_history']='accepted_com_pose_v30';values['body_flags']=flags.copy()
    return JointContactScene(values,metadata,spec,device,mimic_spec=mimic if len(mimic.pairs) else None)


class JointContactScene(RigidContactScene):
    def __init__(self,values,metadata,spec,device,*,mimic_spec=None):
        self.joint_spec=spec;self.mimic_spec=mimic_spec
        super().__init__(values,metadata,device)
        if metadata.get('prescribed_pose_history'):
            from .prescribed_motion import configure
            configure(self)

    def joint_potential(self):
        s=self.joint_spec
        linear=s.linear_projector.square().sum((-1,-2))>0;angular=s.angular_projector.square().sum((-1,-2))>0
        structural=(linear&((s.stiffness[:,:3]+s.damping[:,:3]).sum(-1)>0))|(angular&((s.stiffness[:,3:]+s.damping[:,3:]).sum(-1)>0))
        drive=(s.drive_k+s.drive_d).sum(-1)>0
        limit=((s.limit_k>0)&((s.lower>-1e10)|(s.upper<1e10))).any(-1)
        enabled=s.enabled
        if self.metadata.get('prescribed_pose_history'):enabled=enabled&self.movable[s.owners].any(-1)
        return torch.cat((structural&enabled,drive&enabled,limit&enabled))

    def all_relations(self,contact_owners,contact_active):
        s=self.joint_spec;owners=torch.cat((contact_owners,s.owners.repeat(3,1)))
        kinds=torch.cat((torch.full((len(contact_owners),),7,device=self.device,dtype=torch.long),torch.tensor([3,11,12],device=self.device).repeat_interleave(len(s.owners))))
        active=torch.cat((contact_active,self.joint_potential()))
        if self.mimic_spec is not None:
            from .mimic_constraint import relation
            ids,enabled=relation(self)
            owners=torch.cat((torch.nn.functional.pad(owners,(0,2),value=-1),ids))
            kinds=torch.cat((kinds,torch.full((len(ids),),13,device=self.device,dtype=torch.long)))
            active=torch.cat((active,enabled))
        return owners,kinds,active

    def update_graph(self,owners,active):
        owners,kinds,active=self.all_relations(owners,active)
        signature=(tuple(map(tuple,owners.tolist())),tuple(active.tolist()))
        if self._graph_signature==signature:return
        self._graph_signature=signature;self.topology_generation+=1
        graph=build_topology(self.kind,self.movable,self.object_id,torch.nn.functional.pad(owners,(0,4-owners.shape[1]),value=-1),kinds,potential_active=active,generation=self.topology_generation)
        self.layout=graph.layout;self.layout.legacy_kind=torch.where(kinds==7,6,3)
        self.interaction_id=graph.interaction_id;self.interaction_entities=[torch.where(self.interaction_id==i)[0] for i in torch.unique(self.interaction_id).tolist()]
        for key in ('_protected_topology','_hierarchy_plans','_transport_plans','_vector_active_components','_hierarchy_material_masks'):
            if hasattr(self,key):delattr(self,key)

    def joint_coordinates(self,state,previous,*,jacobian=True):
        ids=self.joint_spec.owners
        return coordinates(self.joint_spec,state.x[:,ids],state.q[:,ids],self.body_com[ids],previous_angle=previous,jacobian=jacobian)

    def initial_state(self,model,wp):
        state=super().initial_state(model,wp);previous=self.joint_spec.coordinate_offset[None,:,3]
        value=self.joint_coordinates(state,previous).axis[...,3]
        history={'joint_angle_prev':torch.where(self.joint_spec.angular_mode==1,value,0.)}
        if self.metadata.get('prescribed_pose_history'):
            from .prescribed_motion import initialize
            history.update(initialize(self,state))
        return State(state.x,state.q,state.v,state.omega,history)
