"""Seeded, solver-free asset reconstruction for independent VBD trajectories."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch

SCHEMA='native_augmentation_v1'
FAMILIES=('rigid','soft','cloth','cable_cloth','joint','rigid_soft',
          'rigid_cloth','soft_cloth','joint_soft','multitype')


def specification(family,seed):
    rng=np.random.default_rng(seed)
    friction=float(rng.uniform(.35,1.25));size=float(rng.uniform(.05,.085))
    spec={'schema':'native_shared_asset_v1','name':f'augmentation_{family}_{seed}',
          'simulation':{'dt':1/1800,'substeps':10,'motion_budget':.025},
          'contact':{'friction':friction,'rigid_ke':20000.,'rigid_kd':10.,
                     'shape_margin':.001,'soft_ke':12000.,'soft_kd':10.,
                     'self_margin':.004,'self_gap':.002},
          'solver':{'rigid_joint_linear_ke':100000.,'rigid_joint_angular_ke':100000.},
          'objects':[],'joints':[]}
    def rigid(name='payload',z=.15):
        kind=rng.choice(['box','sphere','capsule'])
        geometry=({'type':'box','half_extents':[.022,.025,.022]} if kind=='box' else
                  {'type':kind,'radius':.022,**({'half_height':.015} if kind=='capsule' else {})})
        return {'name':name,'family':'rigid','geometry':geometry,
            'material':{'density':float(rng.uniform(300,1100))},
            'position':[float(rng.uniform(-.012,.012)),0.,z],
            'velocity':[float(rng.uniform(-.08,.08)),0.,0.]}
    def soft(z=.006):
        return {'name':'elastic','family':'softbody','geometry':{'type':'grid','resolution':[3,3,3],'size':[size,size,size]},
            'material':{'density':float(rng.uniform(250,650)),'mu':float(rng.uniform(2000,12000)),
                        'lambda':float(rng.uniform(3000,18000)),'damping':.2},
            'position':[-size/2,-size/2,z],'particle_radius':.002}
    def cloth(z=.07,pinned=True):
        return {'name':'fabric','family':'cloth','geometry':{'type':'grid','resolution':[4,4],'size':[.18,.18]},
            'material':{'stretch':float(rng.uniform(100,700)),'area':float(rng.uniform(200,900)),
                        'bending':float(rng.uniform(.0005,.008)),'mass':.0005,'damping':.02},
            'position':[-.09,-.09,z],'particle_radius':.002,'pins':[0,4,20,24] if pinned else []}
    if family=='rigid':spec['objects']=[rigid(z=.055),rigid('upper',z=.15)]
    elif family=='soft':spec['objects']=[soft(z=float(rng.uniform(.025,.09)))]
    elif family=='cloth':spec['objects']=[cloth(z=float(rng.uniform(.05,.12)),pinned=False)]
    elif family=='rigid_soft':spec['objects']=[soft(),rigid(z=size+.065)]
    elif family=='rigid_cloth':spec['objects']=[cloth(),rigid(z=.15)]
    elif family=='soft_cloth':spec['objects']=[cloth(z=.04),soft(z=.09)]
    elif family in ('joint','joint_soft'):
        p=rigid('press',z=.13);p['geometry']={'type':'box','half_extents':[.035,.035,.012]};p['velocity']=[0.,0.,0.]
        spec['objects']=[p]+([soft()] if family=='joint_soft' else [rigid('rest',z=.026)])
        spec['joints']=[{'type':'prismatic','parent':'world','child':'press','axis':[0,0,1],
            'parent_frame':[0,0,.13,0,0,0,1],'target_ke':float(rng.uniform(800,3000)),
            'target_kd':30.,'target_pos':0.,'limit_lower':-.085,'limit_upper':.01,'limit_ke':10000.}]
    elif family in ('cable_cloth','multitype'):
        spec['objects']=[cloth(z=.035)]
        if family=='multitype':spec['objects'] += [soft(z=.07),rigid(z=.07+size+.12)]
        height=.07+size+.035 if family=='multitype' else .14
        points=[[float(x),.015*float(np.sin(i*.7)),height] for i,x in enumerate(np.linspace(-.07,.07,7))]
        spec['objects'].append({'name':'cable','family':'rod',
            'geometry':{'type':'polyline','points':points,'radius':.005},
            'material':{'stretch':float(rng.uniform(500,1800)),'shear':float(rng.uniform(300,1200)),
                'bending':float(rng.uniform(.01,.08)),'twist':float(rng.uniform(.005,.03)),
                'density':float(rng.uniform(400,1200)),'bend_damping':.001}})
    else:raise ValueError(family)
    return spec


def build(spec,configuration,newton,wp,device):
    from .native_assets import build as native_build
    encoded=json.dumps(spec,sort_keys=True).encode()
    source={**spec,'_directory':str(Path.cwd()),'_asset_file':spec['name']+'.json',
            '_asset_sha256':hashlib.sha256(encoded).hexdigest()}
    model,cfg,provenance=native_build(source,newton,wp,device=device)
    cfg.update(integrate_with_external_rigid_solver=False,rigid_contact_history=False,
               iterations=32,rigid_body_particle_contact_buffer_size=1024)
    has_rod=provenance['has_native_rods']
    self_contact=has_rod or 'soft_cloth' in spec['name']
    cfg['particle_enable_self_contact']=bool(self_contact)
    if self_contact:
        for field in ('particle_external_vertex_contact_filtering_map','particle_external_edge_contact_filtering_map'):
            if cfg.get(field) is None:cfg[field]={}
    if configuration:cfg.update(configuration)
    return model,cfg,provenance


def export_scene(model,cfg,meta,newton,wp,device):
    dt=meta['dt'];kind=meta['family']
    if kind in ('cable_cloth','multitype'):
        from .elastic_rod_scene import export_model
        from .elastic_rod_runtime import ElasticRodCollisionOracle
        scene=export_model(model,cfg,meta['dataset_id'],newton,dt,device,
                           motion_budget=.025,full_surface=True,max_rows=50000)
        scene.geometry_oracle=ElasticRodCollisionOracle(model,scene,wp,newton)
    elif kind=='soft_cloth':
        from .self_contact_scene import export_model
        from .self_contact_runtime import SelfContactCollisionOracle
        scene=export_model(model,cfg,meta['dataset_id'],newton,dt,device,
                           motion_budget=.025,full_surface=True,max_rows=50000)
        from .elastic_rod_configuration import record_configuration
        scene.metadata['solver_configuration']=record_configuration(cfg)
        scene.geometry_oracle=SelfContactCollisionOracle(model,scene,wp,newton)
    else:
        from .surface_joint_scene import export_model
        from .surface_joint_runtime import SurfaceJointCollisionOracle
        scene=export_model(model,cfg,meta['dataset_id'],newton,dt,device)
        scene.metadata['detection_motion_budget']=.025
        scene.geometry_oracle=SurfaceJointCollisionOracle(model,scene,wp,newton)
    scene.metadata.update(compact_numerical_diagnostics=True,bounded_motion_forecast_target_only=True,
                          elastic_shape_response=True,training_schema=SCHEMA,
                          joint_tree_coarse_modes='scalar_tree_twelve_columns_v307')
    return scene


def load_reference(path,wp,newton,device='cpu',*,state_device=None):
    from .elastic_rod_configuration import decode_configuration
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']));states={key:torch.as_tensor(z[key].copy(),device=state_device or device)
            for key in z.files if key.startswith(('history__','input_')) or key in ('x','q','v','omega')}
    if meta['schema']!=SCHEMA or meta['qualification']['status']!='PASS':raise ValueError('Qualified native augmentation required')
    model,cfg,_=build(meta['specification'],decode_configuration(meta['configuration']),newton,wp,device)
    scene=export_scene(model,cfg,meta,newton,wp,device)
    return scene,states


def advance(model,scene,current,actions,dt,wp,*,learned=True):
    from .general_joint_data import set_controls
    from .mixed_contact_data import apply_pin_input
    from .mixed_experiment import external_inputs
    from .self_contact_repair import configure
    configure(scene,model.config)
    set_controls(scene,actions,0)
    current=apply_pin_input(scene,current,actions,0)
    scene.candidates=scene.geometry_oracle.query(current)
    scene.candidate_kind=scene.candidates.relation_kind
    return model(scene,current,dt,learned=learned,**external_inputs(actions,slice(0,1)))
