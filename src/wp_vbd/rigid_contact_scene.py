"""Rigid COM scene and active-contact graph for the shared neural response."""
from types import SimpleNamespace
import numpy as np
import torch
from . import math3d as so3
from .data import State
from .unified_graph import build_topology


def export_model(model,configuration,identifier,newton,dt):
    if model.particle_count or model.spring_count or model.attachment_particle_body_count:raise ValueError('This contact adapter currently requires rigid-only assets')
    if np.any(model.joint_type.numpy()!=int(newton.JointType.FREE)):raise ValueError('Structural joint ALM has a separate admission gate')
    if not configuration['rigid_compliant_alm'] or configuration['rigid_contact_history'] or configuration['integrate_with_external_rigid_solver']:raise ValueError('This first contact contract requires compliant ALM without cross-step warm start or external ownership')
    alpha=configuration.get('rigid_avbd_contact_alpha')
    if alpha is None:alpha=configuration.get('rigid_avbd_alpha') or 0.
    values={name:getattr(model,source).numpy().copy() for name,source in [('mass','body_mass'),('inertia','body_inertia'),('inverse_inertia','body_inv_inertia'),('body_com','body_com'),('rest_pose','body_q'),('inverse_mass','body_inv_mass'),('gravity','gravity')]}
    values['gravity']=values['gravity'].reshape(-1,3)
    if len(values['gravity'])!=1:raise ValueError('Multi-world contact ownership needs packed state support')
    values['gravity']=values['gravity'][0]
    shape_body=model.shape_body.numpy();radius=model.shape_collision_radius.numpy();offset=model.shape_transform.numpy()[:,:3]
    bounds=np.zeros(model.body_count,np.float64)
    for shape,body in enumerate(shape_body):
        if body>=0:bounds[body]=max(bounds[body],np.linalg.norm(offset[shape]-values['body_com'][body])+radius[shape]+float(model.shape_margin.numpy()[shape]))
    values['body_radius']=bounds
    values['reference_length']=np.asarray(max(np.ptp(values['rest_pose'][:,:3],axis=0).max(),2*bounds.max(),.1))
    return values,{'contract':'rigid_contact_primal_dual_v21','family':'mixed','official_demo':identifier,'dt':dt,'particle_count':0,'body_count':model.body_count,'contact_alpha':alpha,'friction_epsilon':configuration['friction_epsilon'],'solver_configuration':configuration,'contacts':'rigid_shape','body_position_representation':'com','normal_policy':'Eliminate the finite compliant normal multiplier at its constitutive fixed point; preserve Coulomb tangent dual iterations.','cross_step_contact_history':False}


class RigidContactScene:
    def __init__(self,values,metadata,device='cuda:0'):
        self.metadata=metadata;self.device=torch.device(device);self.length=float(values['reference_length'])
        self.values={name:torch.as_tensor(value,device=device).double() for name,value in values.items()}
        n=metadata['body_count'];self.n=n+1;self.world=n
        def world(value,tail):return torch.cat((value,value.new_tensor(tail).reshape(1,*value.shape[1:])),0)
        self.mass=world(self.values['mass'],[0.]);self.inertia=world(self.values['inertia'],torch.eye(3).tolist())
        self.inverse_inertia=world(self.values['inverse_inertia'],torch.zeros(3,3).tolist())
        self.body_com=world(self.values['body_com'],[0.,0.,0.]);self.body_radius=world(self.values['body_radius'],[0.])
        pose=world(self.values['rest_pose'],[0.,0.,0.,0.,0.,0.,1.]);self.rest_q=so3.normalize(pose[:,3:]);self.rest_x=pose[:,:3]+so3.rotate(self.rest_q,self.body_com)
        self.movable=torch.cat((self.values['inverse_mass']>0,torch.tensor([False],device=device)))
        self.body=torch.ones(self.n,device=device,dtype=torch.bool);self.kind=torch.full((self.n,),2,device=device,dtype=torch.long)
        self.dof=self.movable[:,None].expand(-1,6);self.object_id=torch.arange(self.n,device=device);self.origins=self.rest_x.clone();self.gravity=self.values['gravity']
        eig,frame=so3.eigh_zero_safe(self.inertia);self.inverse_inertia_sqrt=frame@torch.diag_embed(eig.clamp_min(1e-30).rsqrt())@frame.transpose(-1,-2)
        self.generalized_scale=self.mass.new_tensor([1/self.length]*3+[1.]*3)
        self.topology_generation=0;self._graph_signature=None
        self.update_graph(torch.empty(0,2,device=device,dtype=torch.long),torch.empty(0,device=device,dtype=torch.bool))

    def update_graph(self,owners,active):
        signature=(tuple(map(tuple,owners.tolist())),tuple(active.tolist()))
        if self._graph_signature==signature:return
        self._graph_signature=signature;self.topology_generation+=1
        kinds=torch.full((len(owners),),7,device=self.device,dtype=torch.long)
        padded=torch.nn.functional.pad(owners,(0,2),value=-1)
        graph=build_topology(self.kind,self.movable,self.object_id,padded,kinds,potential_active=active,generation=self.topology_generation)
        self.layout=graph.layout
        # Generic contact codec retained for the inherited seven-family MLP;
        # the hierarchy sees the distinct rigid-contact kind7 embedding.
        self.layout.legacy_kind=torch.full_like(kinds,6)
        self.interaction_id=graph.interaction_id
        self.interaction_entities=[torch.where(self.interaction_id==i)[0] for i in torch.unique(self.interaction_id).tolist()]
        for key in ('_protected_topology','_hierarchy_plans','_transport_plans','_vector_active_components','_hierarchy_material_masks'):
            if hasattr(self,key):delattr(self,key)

    def initial_state(self,model,wp):
        pose=wp.to_torch(model.body_q).double();q=so3.normalize(pose[:,3:]);x=pose[:,:3]+so3.rotate(q,self.body_com[:-1])
        velocity=wp.to_torch(model.body_qd).double()
        return State(torch.cat((x,x.new_zeros(1,3)))[None],torch.cat((q,q.new_tensor([[0.,0.,0.,1.]])))[None],torch.cat((velocity[:,:3],x.new_zeros(1,3)))[None],torch.cat((velocity[:,3:],x.new_zeros(1,3)))[None])
