"""Unified mixed DOFs and exact primitive ground geometry; no solver imports."""

import numpy as np
import torch
from types import SimpleNamespace

from .adapters import export_static, np_rotate
from .data import State
from .response_operator import RelationLayout
from .math3d import eigh_zero_safe


class MixedScene:
    def __init__(self, values, metadata):
        self.values, self.metadata = values, metadata
        self.n = len(values['rest_x'])
        values.setdefault('attachments',torch.empty(0,2,device=values['rest_x'].device,dtype=torch.long))
        values.setdefault('attachment_anchor',values['rest_x'].new_empty(0,3))
        values.setdefault('attachment_stiffness',values['rest_x'].new_empty(0))
        self.length = float(values['reference_length'])
        self.tet_floor=float(metadata.get('tet_min_det',.01 if metadata.get('contract')=='official_objective_materials_v5' else .05))
        if not 0<self.tet_floor<1:raise ValueError('Positive tetrahedral orientation floor is required')
        self.device = values['rest_x'].device
        for key,value in values.items():
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):raise ValueError(f'Nonfinite mixed field: {key}')
        if self.length<=0 or bool((self.mass<0).any()) or bool((self.movable&(self.mass<=0)).any()):raise ValueError('Invalid mixed length or mass')
        initial_point=self.rest_x[self.site_owner]+2*torch.linalg.cross(self.rest_q[self.site_owner,:3],torch.linalg.cross(self.rest_q[self.site_owner,:3],self.site_local)+self.rest_q[self.site_owner,3:]*self.site_local)+self.site_local
        if bool((initial_point[:,2]-self.site_radius-self.plane_height<0).any()):raise ValueError('Mixed asset starts below its physical plane')
        self.bending_indices = torch.where((self.bend_edges >= 0).all(-1) & (self.bend_material[:,:2].sum(-1) > 0))[0]
        self.material_damping_flags=tuple(bool((mat[:,column]>0).any()) for mat,column in ((self.tet_material,2),(self.tri_material,2),(self.bend_material,1)))
        self.values['tet_volume']=(1/(6*torch.linalg.det(self.tet_Dm))).abs()
        self.body = self.kind >= 2
        if self.attachments.ndim!=2 or self.attachments.shape[1]!=2 or bool(((self.attachments<0)|(self.attachments>=self.n)).any()):raise ValueError('Invalid attachment incidence')
        if bool((self.attachment_stiffness<0).any()) or bool(self.body[self.attachments[:,0]].any()) or bool((~self.body[self.attachments[:,1]]).any()):raise ValueError('Attachments require node/body endpoints and nonnegative stiffness')
        self.axis_z=self.rest_x.new_tensor([0,0,1])
        self.axis_xy=self.rest_x.new_tensor([1,1,0])
        self.generalized_scale=self.rest_x.new_tensor([1/self.length]*3+[1.]*3)
        from .mixed_physics import rod_measure
        identity=self.rest_x.new_tensor([0,0,0,1]).expand_as(self.joint_rest_q)
        self.rod_rest=rod_measure(identity,self.joint_rest_q)
        eigenvalue,eigenvector=eigh_zero_safe(self.inertia)
        self.inverse_inertia_sqrt=eigenvector@torch.diag_embed(eigenvalue.clamp_min(1e-10).rsqrt())@eigenvector.transpose(-1,-2)
        self.dof = self.movable[:,None] & torch.cat((torch.ones(self.n,3,device=self.device,dtype=torch.bool),self.body[:,None].expand(-1,3)),1)
        self.objects = torch.unique(self.object_id,sorted=True).tolist()
        self.object_entities = [torch.where(self.object_id == obj)[0] for obj in self.objects]
        self.object_sites = [torch.where(self.object_id[self.site_owner] == obj)[0] for obj in self.objects]
        labels={obj:obj for obj in self.objects}
        def root(obj):
            while labels[obj]!=obj:obj=labels[obj]
            return obj
        for pair,k in zip(self.attachments.tolist(),self.attachment_stiffness.tolist()):
            if k>0 and all(bool(self.movable[i]) for i in pair):
                a,b=[root(int(self.object_id[i])) for i in pair]
                labels[max(a,b)]=min(a,b)
        self.interaction_id=self.object_id.clone()
        for obj in self.objects:self.interaction_id[self.object_id==obj]=root(obj)
        components=torch.unique(self.interaction_id,sorted=True).tolist()
        self.interaction_entities=[torch.where(self.interaction_id==obj)[0] for obj in components]
        self.interaction_sites=[torch.where(self.interaction_id[self.site_owner]==obj)[0] for obj in components]
        self.floating_components=[(ids,sites) for ids,sites in zip(self.interaction_entities,self.interaction_sites) if bool(self.movable[ids].all())]
        self.cloth_retraction_components=[]
        for ids in self.interaction_entities:
            mobile=ids[self.movable[ids]]
            if len(mobile)>=3 and bool((self.kind[ids]==1).all()):
                points=self.rest_x[mobile].double().cpu().numpy()
                if np.linalg.matrix_rank(points-points.mean(0),tol=1e-9)>=2:self.cloth_retraction_components.append(mobile)
        from .cloth_bending import detect_bending_plan
        self.bending_plans=[]
        for ids in self.interaction_entities:
            if bool((self.kind[ids]==1).all()):
                plan=detect_bending_plan(self,ids)
                if plan is not None:self.bending_plans.append(plan)
        planned={tuple(plan.mobile.tolist()) for plan in self.bending_plans}
        self.non_bending_cloth_components=[ids for ids in self.cloth_retraction_components if tuple(ids.tolist()) not in planned]
        self.tet_components=[]
        for component in torch.unique(self.interaction_id[self.tets[:,0]]).tolist():
            ids=torch.where(self.interaction_id==component)[0]
            rows=torch.where(self.interaction_id[self.tets[:,0]]==component)[0]
            self.tet_components.append((ids,rows,int(ids[0])))
        self.box_sites = [torch.tensor(row,device=self.device) for row in metadata['box_sites']]
        rows,tags,materials,slices = [],[],[],{}
        entries=[('tets',0,self.tet_material[:,:3].sum(-1)),('triangles',1,self.tri_material[:,:3].sum(-1)),('bend_edges',2,self.bend_material[:,:2].sum(-1)),('joints',3,self.joint_stiffness.sum(-1)),('attachments',5,self.attachment_stiffness)]
        for key,tag,weight in entries:
            start=len(rows)
            source=getattr(self,key)
            keep=(source >= 0).all(-1) & (weight > 0)
            selected=torch.where(keep)[0]
            for i in selected.tolist():
                row=source[i].tolist()
                rows.append(row+[-1]*(4-len(row)))
                tags.append(4 if key=='joints' and int(self.joint_tag[i])==1 else tag)
                materials.append(float(weight[i]))
            slices[key]=(slice(start,len(rows)),selected)
        self.structural_slices=slices
        self.contact_start=len(rows)
        for owner in self.site_owner.tolist():
            rows.append([owner,-1,-1,-1]);tags.append(6);materials.append(0.)
        incidence=torch.tensor(rows,device=self.device,dtype=torch.long).reshape(-1,4)
        relation_kind=torch.tensor(tags,device=self.device,dtype=torch.long)
        # Canonical, geometric relation pooling. No dynamic responses are passed
        # through the common static ground into otherwise independent objects.
        valid=incidence>=0
        active=self.movable[incidence.clamp_min(0)]&valid
        first_active=active.to(torch.int64).argmax(-1)
        representative=incidence.gather(1,first_active[:,None]).squeeze(-1)
        representative=torch.where(active.any(-1),representative,incidence[:,0])
        owner=self.interaction_id[representative]
        center=self.rest_x[incidence.clamp_min(0)]
        center=(center*valid[:,:,None]).sum(1)/valid.sum(1)[:,None]
        keys=torch.cat((owner[:,None],relation_kind[:,None],center),1).cpu().numpy()
        order=np.lexsort(tuple(keys[:,i] for i in reversed(range(keys.shape[1]))))
        pool=np.empty(len(rows),np.int64);components=[];counter=-1;previous=None;count=0
        for i in order:
            group=(int(keys[i,0]),int(keys[i,1]))
            if group!=previous or count==8:
                counter+=1;components.append(group[0]);count=0
            pool[i]=counter;previous=group;count+=1
        self.layout=RelationLayout(incidence,self.kind,relation_kind,torch.as_tensor(pool,device=self.device),torch.tensor(components,device=self.device),conditioning_mask=self.movable)
        self.relation_weight=torch.tensor(materials,device=self.device)
        self.origins=torch.zeros_like(self.rest_x)
        for ids in self.object_entities:
            self.origins.index_copy_(0,ids,self.rest_x[ids].mean(0).expand(len(ids),-1))
        self.objective_scene=SimpleNamespace(**{key:value.double() if value.is_floating_point() else value for key,value in self.values.items()},body=self.body,bending_indices=self.bending_indices,material_damping_flags=self.material_damping_flags,box_sites=self.box_sites,axis_z=self.axis_z.double(),axis_xy=self.axis_xy.double(),rod_rest=self.rod_rest.double())
        if metadata.get('history_contract') == 'rod_compliant_alm_v1':
            from .rod_solver import prepare
            prepare(self)

    def __getattr__(self,key):
        if key in self.values:return self.values[key]
        raise AttributeError(key)


def export_mixed(model, groups, spec, newton):
    gravity=np.asarray(spec['gravity'],np.float32)
    parameters={}
    for group in groups:
        if group['family'] in ('joint','rod'):parameters.update(group['params'])
    node=export_static('softbody',model,parameters,gravity,newton)
    body=export_static('joint',model,parameters,gravity,newton)
    n=model.particle_count
    output={key:np.concatenate((node[key],body[key])) for key in ('rest_x','rest_q','mass','movable','inertia','kind','object_id')}
    output.update({key:value for key,value in node.items() if key not in output})
    output.update({key:value for key,value in body.items() if key not in output})
    output['joints']=body['joints']+n
    output['body_com']=np.concatenate((np.zeros((n,3),np.float32),body['body_com']))
    for index,group in enumerate(groups):
        ids=slice(group['particle_start'],group['particle_end']) if group['particle_end']>group['particle_start'] else slice(n+group['body_start'],n+group['body_end'])
        output['kind'][ids]={'softbody':0,'cloth':1,'joint':2,'rigid':2,'rod':3}[group['family']]
        output['object_id'][ids]=index
    if len(body['rest_x'])>model.body_count:
        output['object_id'][-1]=len(groups)
    source_joints=[j for j,t in enumerate(model.joint_type.numpy()) if t!=int(newton.JointType.FREE)]
    joint_stiffness=[]
    for i,j in enumerate(source_joints):
        if int(model.joint_type.numpy()[j])==int(newton.JointType.ROD):
            d=int(model.joint_qd_start.numpy()[j]);k=model.joint_target_ke.numpy()[d:d+6]
            joint_stiffness.append(k)
        else:joint_stiffness.append([parameters['linear_ke']]*3+[parameters['angular_ke']]*3)
    output['joint_stiffness']=np.asarray(joint_stiffness,np.float32)
    output.update(attachments=np.empty((0,2),np.int64),attachment_anchor=np.empty((0,3),np.float32),attachment_stiffness=np.empty(0,np.float32))
    if model.attachment_particle_body_count:
        enabled=model.attachment_particle_body_enabled.numpy()
        bodies=model.attachment_particle_body_body.numpy()[enabled]
        particles=model.attachment_particle_body_particle.numpy()[enabled]
        output['attachments']=np.stack((particles,n+bodies),-1).astype(np.int64)
        output['attachment_anchor']=model.attachment_particle_body_body_point.numpy()[enabled]-model.body_com.numpy()[bodies]
        output['attachment_stiffness']=model.attachment_particle_body_stiffness.numpy()[enabled]
        if np.any(model.attachment_particle_body_damping.numpy()[enabled]!=0):raise ValueError('Nonzero attachment damping is outside this contract')
    # Ghost world has no physical shape. All other supports are explicit.
    owners=list(range(n));local=[np.zeros(3)]*n;physical=[0.]*n
    plane=spec['plane'];margins=[plane['activation_radius']+plane['rigid_margin']]*n
    mu=[float(np.sqrt(plane['particle_mu']*plane['ground_mu']))]*n
    boxes=[]
    com=model.body_com.numpy();shape_body=model.shape_body.numpy()
    for s,b in enumerate(shape_body):
        if b<0:continue
        kind=int(model.shape_type.numpy()[s]);xf=model.shape_transform.numpy()[s];scale=model.shape_scale.numpy()[s]
        if kind==int(newton.GeoType.BOX):
            points=[np.array([1 if i&1 else -1,1 if i&2 else -1,1 if i&4 else -1])*scale for i in range(8)]
            boxes.append(list(range(len(owners),len(owners)+8)));radius=0.
        elif kind==int(newton.GeoType.CAPSULE):
            points=[np.array([0,0,scale[1]]),np.array([0,0,-scale[1]])];radius=float(scale[0])
        else:raise ValueError('Mixed ground adapter supports BOX and CAPSULE only')
        for point in points:
            owners.append(n+int(b));local.append(xf[:3]+np_rotate(xf[3:],point)-com[b]);physical.append(radius)
            margins.append(plane['rigid_margin']+float(model.shape_margin.numpy()[s]));mu.append(float(np.sqrt(model.shape_material_mu.numpy()[s]*plane['ground_mu'])))
    output.update(site_owner=np.asarray(owners,np.int64),site_local=np.asarray(local,np.float32),site_radius=np.asarray(physical,np.float32),site_margin=np.asarray(margins,np.float32),site_mu=np.asarray(mu,np.float32),plane_height=np.asarray(plane['height'],np.float32),contact_ke=np.asarray(plane['ke'],np.float32),contact_kd=np.asarray(plane['kd'],np.float32),friction_epsilon=np.asarray(plane['friction_epsilon'],np.float32))
    metadata={'family':'mixed','contacts':'analytic_plane','contract':spec.get('contract','finite_elastic_plane_ipc_v1'),'groups':groups,'box_sites':boxes,'particle_count':n,'body_count':model.body_count,'dt':spec['dt'],'spec':spec}
    return output,metadata


def tensor_scene(static, metadata, device='cuda:0'):
    values={key:torch.as_tensor(np.asarray(value).copy(),device=device,dtype=torch.bool if np.asarray(value).dtype.kind=='b' else torch.long if np.asarray(value).dtype.kind in 'iu' else torch.float32) for key,value in static.items()}
    return MixedScene(values,metadata)


def initial_state(scene,model):
    state=State(scene.rest_x[None].clone(),scene.rest_q[None].clone(),torch.zeros(1,scene.n,3,device=scene.device),torch.zeros(1,scene.n,3,device=scene.device))
    n=scene.metadata['particle_count'];b=scene.metadata['body_count']
    if n:state.v[0,:n]=torch.as_tensor(model.particle_qd.numpy(),device=scene.device)
    if b:
        state.v[0,n:n+b]=torch.as_tensor(model.body_qd.numpy()[:,:3],device=scene.device)
        state.omega[0,n:n+b]=torch.as_tensor(model.body_qd.numpy()[:,3:],device=scene.device)
    if scene.metadata.get('history_contract'):
        from .rod_adapter import initial_history
        state.history=initial_history(scene,state)
    return state
