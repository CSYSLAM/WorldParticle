"""Native rods, finite joints and elastic self-contact in one ownership space."""
from types import SimpleNamespace

import numpy as np
import torch

from .general_joint_export import export_spec, read_controls
from .mixed_contact_scene import export_model as export_elastic
from .rod_adapter import export_model as export_rod_material
from .rod_contact_scene import RodContactScene
from .self_contact_query import interaction_geometry
from .surface_joint_scene import SurfaceJointScene
from .elastic_rod_configuration import decode_configuration,record_configuration

CONTRACT='elastic_rod_surface_joint_self_contact_v222'


def export_model(model,configuration,identifier,newton,dt,device='cpu',*,motion_budget,
                 full_surface=False,broad_phase='chunked',max_rows=200000):
    configuration=decode_configuration(configuration)
    if not configuration['particle_enable_self_contact']:
        raise ValueError('The composed path requires explicit particle self-contact')
    if not model.particle_count:
        raise ValueError('The composed path requires authored elastic particles')
    if not np.isfinite(motion_budget) or motion_budget<=0:
        raise ValueError('Finite positive shared detection motion budget required')
    if type(max_rows) is not int or max_rows<=0:
        raise ValueError('Positive complete self-contact capacity required')
    allowed={int(getattr(newton.GeoType,name)) for name in
             ('PLANE','SPHERE','BOX','CAPSULE','CYLINDER','CONE','ELLIPSOID')}
    if not np.isin(model.shape_type.numpy(),list(allowed)).all():
        raise ValueError('Composed elastic/rod contact has no mesh force contract')
    interaction_geometry(configuration)
    rod,rod_metadata=export_rod_material(model,configuration,identifier,newton,dt,
        allow_general_joints=True,allow_elastic_particles=True)
    spec=export_spec(model,configuration,newton,device,allow_rods=True)
    source_ids=np.concatenate((rod['source_joint_ids'],spec.source_joint_ids.cpu().numpy()))
    expected=np.flatnonzero(model.joint_type.numpy()!=int(newton.JointType.FREE))
    if len(np.unique(source_ids))!=len(source_ids) or not np.array_equal(np.sort(source_ids),expected):
        raise ValueError('Every authored non-FREE joint needs exactly one constitutive owner')
    n=model.particle_count
    spec.owners=spec.owners+n
    base=export_elastic(model,configuration,identifier,newton,dt,device,
                        finite_joint_export=True,self_contact_export=True)
    if len(base.joints) or base.n!=n+model.body_count+1:
        raise AssertionError('One world slot and no duplicate legacy joint energies required')
    values={**base.values,'kind':base.kind.clone(),'object_id':base.object_id.clone()}
    values['kind'][torch.as_tensor(rod['joints'].ravel(),device=device)+n]=3
    # Material connectivity defines cable identity. Finite joints connect
    # objects in the interaction graph without merging their material identity.
    labels=values['object_id'].cpu().numpy();parent=list(range(int(labels.max())+1))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for a,b in rod['joints']+n:parent[root(int(labels[b]))]=root(int(labels[a]))
    canonical={};groups=[]
    for node,label in enumerate(labels):
        key=root(int(label))
        if key not in canonical:
            canonical[key]=len(canonical)
            family={0:'softbody',1:'cloth',2:'rigid',3:'rod'}[int(values['kind'][node])]
            groups.append({'name':f'object_{canonical[key]}','family':family})
        values['object_id'][node]=canonical[key]
    metadata={**base.metadata,'contract':CONTRACT,'groups':groups,'mixed_finite_joints':True,
        'solver_configuration':record_configuration(configuration),
        'joint_history_contract':'single_axis_winding_v26','joint_alpha':rod_metadata['joint_alpha'],
        'joint_gamma':rod_metadata['joint_gamma'],
        'self_contact_EE_topology_rule':'union_of_source_directional_exclusions',
        'self_shape_full_surface':bool(full_surface),'self_contact_broad_phase':broad_phase,
        'self_contact_max_rows':max_rows,'self_contact_motion_budget':float(motion_budget),
        'contacts':'analytic_rigid_weighted_soft_reciprocal_VF_EE_and_native_ROD',
        'scope':'Owned elastic particles, native ROD material, finite joints and analytic/self contacts. No mesh, attachment, external ownership or quality/realtime claim.'}
    return ElasticRodScene(values,metadata,spec,rod,n,device,model)


class ElasticRodScene(SurfaceJointScene):
    _prepare_forest=RodContactScene._prepare_forest

    def __init__(self,values,metadata,spec,rod,offset,device,model):
        integer={'joints','source_joint_ids','object_id'}
        keys=('joints','source_joint_ids','object_id','anchor_p','anchor_c','anchor_qp','anchor_qc',
              'rod_rest_bend','rod_rest_twist','joint_stiffness','rod_damping','rod_dahl_eps',
              'rod_dahl_tau','rod_inverse_mass','rod_inverse_inertia')
        self.rod=SimpleNamespace(**{key:torch.as_tensor(rod[key],device=device,
            dtype=torch.long if key in integer else torch.float64) for key in keys})
        self.rod.joints=self.rod.joints+offset
        for key in ('rod_inverse_mass','rod_inverse_inertia'):
            source=getattr(self.rod,key)
            padded=source.new_zeros(len(values['rest_x']),*source.shape[1:])
            padded[offset:offset+len(source)]=source
            setattr(self.rod,key,padded)
        super().__init__(values,metadata,spec)
        self._prepare_forest()
        self.rod_node_mask=torch.zeros(self.n,device=self.device,dtype=torch.bool)
        self.rod_node_mask[self.rod.joints.flatten()]=True
        self.joint_controls=read_controls(spec,model.control(clone_variables=False))

    def all_relations(self,owners,active):
        ids,kinds,enabled=super().all_relations(owners,active)
        rod_ids=torch.nn.functional.pad(self.rod.joints,(0,2),value=-1)
        rod_active=(self.rod.joint_stiffness+self.rod.rod_damping).sum(-1)>0
        return (torch.cat((ids,rod_ids)),
                torch.cat((kinds,torch.full_like(rod_active,4,dtype=torch.long))),
                torch.cat((enabled,rod_active)))
