"""Native boundary for the shared elastic/ROD/self-contact response."""
import torch

from . import elastic_rod_system as physics
from .elastic_rod_scene import CONTRACT
from .general_joint_export import export_spec,read_controls
from .mixed_contact_runtime import MixedContactExampleSolver,native_tensor
from .self_contact_runtime import SelfContactCollisionOracle


class ElasticRodCollisionOracle(SelfContactCollisionOracle):
    validate_snapshot_history=staticmethod(physics.validate_history)

    def material_reference(self,state):
        if bool(state.history['rod_initialized'].all()):
            return state.history['rod_previous_x'][0,:self.scene.metadata['particle_count']].detach().clone()
        return super().material_reference(state)


class ElasticRodExampleSolver(MixedContactExampleSolver):
    def __init__(self,network,scene,dt,wp,newton,source_model,initial_native,*,learned=True):
        if scene.metadata.get('contract')!=CONTRACT or not network.config.get('elastic_rod_self_contact',False):
            raise ValueError('Explicit shared elastic/ROD scene and model configuration required')
        super().__init__(network,scene,dt,wp,newton,source_model,learned=learned,
                         oracle_type=ElasticRodCollisionOracle)
        self.newton=newton
        self.history={k:v.detach().clone() for k,v in physics.initial_history(scene,self.read(initial_native)).items()}
        # Material changes rebuild this scene/history. Finite joint control
        # targets, rates and efforts remain exogenous inputs on every step.
        for name in ('body_q','shape_margin','shape_collision_radius','shape_flags','edge_rest_angle',
                     'edge_rest_length','joint_type','joint_parent','joint_child','joint_X_p','joint_X_c',
                     'joint_axis','joint_dof_dim','joint_q','joint_q_start','joint_qd_start',
                     'joint_target_q_start','joint_target_q','joint_target_ke','joint_target_kd','joint_enabled'):
            array=getattr(source_model,name,None)
            if array is not None:self.static_source[name]=native_tensor(wp,array).clone()
        extension=getattr(source_model,'vbd',None)
        self.static_rod_material={name:native_tensor(wp,getattr(extension,name)).clone()
            for name in ('dahl_eps_max','dahl_tau') if extension is not None and getattr(extension,name,None) is not None}

    def validate_start(self,start):physics.validate_history(self.scene,start)

    def refresh_control(self,control):
        model=self.source_model;scene=self.scene
        extension=getattr(model,'vbd',None)
        for name,expected in self.static_rod_material.items():
            if not torch.equal(native_tensor(self.wp,getattr(extension,name)),expected):
                raise ValueError('Rod material changed; rebuild shared scene/history: '+name)
        if control is None:control=model.control(clone_variables=False)
        for name in ('tri_activations','tet_activations'):
            value=getattr(control,name,None)
            if value is not None and bool((native_tensor(self.wp,value)!=0).any()):
                raise ValueError('Active elastic control needs an explicit material operator: '+name)
        spec=export_spec(model,scene.metadata['solver_configuration'],self.newton,str(scene.device),allow_rods=True)
        spec.owners=spec.owners+scene.metadata['particle_count']
        for key in ('owners','source_joint_ids','frames','angular_mode','linear_axes','angular_axes',
                    'coordinate_offset','rest_parent','rest_child','dof_index','target_index'):
            if not torch.equal(getattr(spec,key),getattr(scene.joint_spec,key)):
                raise ValueError('Shared finite-joint coordinates changed; rebuild scene/history: '+key)
        values=read_controls(spec,control)
        if any(not bool(torch.isfinite(value).all()) for value in values):
            raise ValueError('Nonfinite shared finite-joint controls')
        effort=native_tensor(self.wp,control.joint_f)
        used=torch.zeros(len(effort),device=effort.device,dtype=torch.bool)
        index=spec.dof_index[spec.dof_index>=0]
        if len(index):used[index.to(used.device)]=True
        if bool((effort[~used]!=0).any()):
            raise ValueError('Unmapped FREE/BALL/ROD generalized effort needs its own force adapter')
        scene.joint_spec=spec;scene.joint_controls=values
