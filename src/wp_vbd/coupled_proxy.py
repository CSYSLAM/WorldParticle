"""One shared response on native ModelView primary and finite-mass proxies.

Proxy bodies remain dynamic with the coupler's effective inertial properties.
Other solvers retain ownership of their primary states. No full VBD is used.
"""
import numpy as np
import torch
from newton._src.solvers.coupled.interface import CouplingInterface
from newton._src.solvers.coupled.model_view import ModelView

from .general_joint_export import export_spec,read_controls
from .mixed_contact_runtime import MixedContactExampleSolver,native_tensor
from .surface_joint_runtime import SurfaceJointExampleSolver
from .mixed_contact_runtime import MixedCollisionOracle
from .surface_joint_collision import SurfaceJointContacts
from .surface_joint_scene import SurfaceJointScene

CONTRACT='elastic_surface_joint_dynamic_proxy_v300'


def validate_proxy_view(model,configuration,newton):
    if not isinstance(model,ModelView):raise ValueError('A native coupled ModelView is required')
    if configuration['integrate_with_external_rigid_solver']:
        raise ValueError('Externally integrated shared-model bodies require their distinct boundary contract')
    body=model.body_flags.numpy();particle=model.particle_flags.numpy()
    dynamic=int(newton.BodyFlags.DYNAMIC);kinematic=int(newton.BodyFlags.KINEMATIC);proxy=int(newton.BodyFlags.PROXY)
    if any(int(flag) not in {dynamic,kinematic,dynamic|proxy,kinematic|proxy} for flag in body):
        raise ValueError('Invalid native body ownership flags')
    if not np.any(body&proxy) and not np.any(particle&int(newton.ParticleFlags.PROXY)):
        raise ValueError('An explicit native proxy marker is required')
    mass=model.body_mass.numpy();inverse=model.body_inv_mass.numpy()
    inertia=model.body_inertia.numpy();inverse_inertia=model.body_inv_inertia.numpy()
    if not all(np.isfinite(value).all() for value in (mass,inverse,inertia,inverse_inertia)):
        raise ValueError('Finite view-local inertial properties required')
    if np.any(mass<0) or np.any(inverse<0):raise ValueError('Nonnegative native masses required')
    positive=inverse>0
    if np.any(positive& (mass<=0)) or not np.allclose(mass[positive]*inverse[positive],1,rtol=2e-5,atol=1e-7):
        raise ValueError('Inconsistent native view mass/inverse mass')
    if positive.any():
        selected=inertia[positive]
        if not np.allclose(selected,selected.transpose(0,2,1),rtol=1e-5,atol=1e-9) or np.any(np.linalg.eigvalsh(selected)<=0):
            raise ValueError('Dynamic proxy inertia must be symmetric positive definite')
        if not np.allclose(selected@inverse_inertia[positive],np.eye(3),rtol=2e-4,atol=2e-5):
            raise ValueError('Inconsistent native view inertia/inverse inertia')


def export_model(model,configuration,identifier,newton,dt,device='cpu',*,full_surface=False):
    from .mixed_contact_scene import export_model as export_elastic
    validate_proxy_view(model,configuration,newton)
    analytic={int(getattr(newton.GeoType,name)) for name in ('PLANE','SPHERE','BOX','CAPSULE','CYLINDER','CONE','ELLIPSOID')}
    if any(int(kind) not in analytic for kind in model.shape_type.numpy()):
        raise ValueError('Mesh proxy geometry needs its independent force/query contract')
    spec=export_spec(model,configuration,newton,device);spec.owners+=model.particle_count
    base=export_elastic(model,configuration,identifier,newton,dt,device,finite_joint_export=True,coupled_proxy_view=True)
    metadata={**base.metadata,'contract':CONTRACT,'mixed_finite_joints':True,
        'shape_full_surface':bool(full_surface),
        'joint_history_contract':'single_axis_winding_v26','contacts':'analytic_rigid_and_weighted_soft_features',
        'ownership':'Native view-local PROXY markers; dynamic proxies keep their effective mass/inertia and finite joints.',
        'scope':'Experimental native proxy exchange; mesh, self-contact, ADMM dual forces and all51 training are not certified.'}
    scene=SurfaceJointScene(base.values,metadata,spec);scene.joint_controls=read_controls(spec,model.control(clone_variables=False))
    proxy=torch.cat((torch.as_tensor((model.particle_flags.numpy()&int(newton.ParticleFlags.PROXY))!=0,device=device),
        torch.as_tensor((model.body_flags.numpy()&int(newton.BodyFlags.PROXY))!=0,device=device),torch.zeros(1,device=device,dtype=torch.bool)))
    scene.proxy_mask=proxy;scene.owned_response=scene.movable&~proxy
    if not bool(scene.owned_response.any()):raise ValueError('The learned view needs a primary mobile response owner')
    return scene


class ProxyCollisionOracle(MixedCollisionOracle):
    def __init__(self,model,scene,wp,newton):
        self.newton=newton
        super().__init__(model,scene,wp,newton,full_surface=scene.metadata['shape_full_surface'])

    def snapshot(self,state):
        return SurfaceJointContacts.snapshot(self.query_model,self.contacts,self.wp,self.newton)


class CoupledProxySurfaceSolver(SurfaceJointExampleSolver,CouplingInterface):
    def __init__(self,network,scene,dt,wp,newton,source_model,initial_native,*,learned=True):
        if scene.metadata.get('contract')!=CONTRACT:raise ValueError('An explicit native dynamic-proxy scene is required')
        MixedContactExampleSolver.__init__(self,network,scene,dt,wp,newton,source_model,learned=learned,oracle_type=ProxyCollisionOracle)
        self.newton=newton;self.model_view_refreshes=0
        initial=self.read(initial_native);spec=scene.joint_spec
        angle=scene.joint_coordinates(initial,spec.coordinate_offset[None,:,3]).axis[...,3]
        self.history['joint_angle_prev']=torch.where(spec.angular_mode==1,angle,0.).detach().clone()
        self.enable_proxy_feedback(scene.proxy_mask,scene.owned_response)
        for name in ('body_q','shape_margin','shape_collision_radius','shape_flags','edge_rest_angle','edge_rest_length',
            'joint_type','joint_parent','joint_child','joint_X_p','joint_X_c','joint_axis','joint_dof_dim',
            'joint_q','joint_q_start','joint_qd_start','joint_target_q_start'):
            value=getattr(source_model,name,None)
            if value is not None:self.static_source[name]=native_tensor(wp,value).clone()

    def coupling_supports_inertial_property_refresh(self):
        # This host-side rebuild is intentionally uncaptured.
        return False

    def notify_model_changed(self,flags):
        allowed=int(self.newton.ModelFlags.BODY_INERTIAL_PROPERTIES|self.newton.ModelFlags.BODY_PROPERTIES)
        if int(flags)&~allowed:raise NotImplementedError('Proxy model changes beyond body inertia require an explicit scene/history rebuild')
        inertial={'body_mass','body_inv_mass','body_inertia','body_inv_inertia'}
        for name,expected in self.static_source.items():
            if name not in inertial and not torch.equal(native_tensor(self.wp,getattr(self.source_model,name)),expected):
                raise ValueError('Inertial refresh changed non-inertial source field: '+name)
        new_scene=export_model(self.source_model,self.scene.metadata['solver_configuration'],self.scene.metadata['official_demo'],self.newton,self.dt,str(self.scene.device),full_surface=self.scene.metadata['shape_full_surface'])
        if not torch.equal(new_scene.proxy_mask,self.scene.proxy_mask) or not torch.equal(new_scene.movable,self.scene.movable):
            raise ValueError('Ownership/response topology changed during an inertial refresh')
        # Construct and validate the replacement before changing live state.
        candidate=CoupledProxySurfaceSolver(self.network,new_scene,self.dt,self.wp,self.newton,self.source_model,self.source_model.state(),learned=self.learned)
        self.scene=candidate.scene;self.oracle=candidate.oracle;self.static_source=candidate.static_source
        self.model_view_refreshes+=1
