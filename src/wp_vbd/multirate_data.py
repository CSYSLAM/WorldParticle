"""Reuse every existing demo adapter for variable-step autonomous training."""

from .mixed_experiment import load_sample,controlled_start,external_inputs


def load_reference(row,wp,newton,device,*,validation=False):
    path=row.get('validation_path',row['path']) if validation else row['path']
    loader=row['loader']
    if loader=='legacy':
        scene,states=load_sample(path,device,state_device='cpu')
    else:
        from .rigid_contact_data import load_reference as rigid
        from .general_joint_data import load_reference as joint
        from .surface_joint_data import load_reference as surface
        from .self_contact_data import load_reference as self_contact
        from .mixed_contact_data import load_reference as mixed
        from .rod_contact_data import load_reference as rod
        from .elastic_rod_data import load_reference as elastic_rod
        from .robot_cloth_data import load_reference as robot_cloth
        loaders={'rigid':rigid,'joint':joint,'surface':surface,'self':self_contact,
                 'mixed':mixed,'rod':rod,'elastic_rod':elastic_rod,'robot_cloth':robot_cloth}
        options={'state_device':'cpu'}
        if loader=='self':
            options['compact_reference_constants']=True
        scene,states=loaders[loader](path,wp,newton,device,**options)
    if loader in ('self','elastic_rod'):
        from .self_contact_runtime import SelfContactCollisionOracle
        from .elastic_rod_runtime import ElasticRodCollisionOracle
        for key in ('self_contact_max_rows','self_contact_motion_budget',
                    'self_contact_responsive_pairs_only'):
            if key in row:
                scene.metadata[key]=row[key]
        oracle=ElasticRodCollisionOracle if loader=='elastic_rod' else SelfContactCollisionOracle
        scene.geometry_oracle=oracle(scene.geometry_oracle.source_model,scene,wp,newton)
    elif loader in ('rigid','joint') and 'detection_motion_budget' in row:
        from .rigid_contact_runtime import CollisionOracle
        scene.geometry_oracle=CollisionOracle(scene.geometry_oracle.source_model,scene,wp,newton,
                                              motion_budget=row['detection_motion_budget'])
    elif loader=='surface' and 'detection_motion_budget' in row:
        # A macro-step can travel farther than a native teacher step. The
        # same envelope must govern both candidate generation and runtime
        # motion checks; changing only metadata would leave stale candidates.
        from .surface_joint_runtime import SurfaceJointCollisionOracle
        scene.metadata['detection_motion_budget']=float(row['detection_motion_budget'])
        scene.geometry_oracle=SurfaceJointCollisionOracle(scene.geometry_oracle.source_model,scene,wp,newton)
    return scene,{key:value.cpu() for key,value in states.items()}


def advance(network,row,scene,current,actions,dt,wp,*,learned=True):
    """Read only exogenous input tensors, with history from the predicted state."""
    from .self_contact_repair import configure
    configure(scene,network.config)
    if scene.metadata.get('contract')=='elastic_surface_joint_contact_v38':
        if network.config.get('learned_contact_tangent_seed',False):
            scene.metadata['surface_tangent_memory']=True
            scene.metadata['surface_tangent_history_capacity']=int(network.config.get('surface_tangent_history_capacity',8192))
            from .surface_tangent_memory import initial_history
            missing={'surface_contact_shape','surface_contact_corners','surface_contact_weights',
                     'surface_contact_local_anchor','surface_contact_normal','surface_contact_tangent'}-set(current.history or {})
            if missing:
                if missing!={'surface_contact_shape','surface_contact_corners','surface_contact_weights',
                             'surface_contact_local_anchor','surface_contact_normal','surface_contact_tangent'}:
                    raise ValueError('Partial surface tangent history in a training state')
                current.history={**(current.history or {}),**initial_history(current,scene.metadata['surface_tangent_history_capacity'])}
    loader=row['loader']
    if loader=='elastic_rod':
        from .elastic_rod_data import step
        return step(network,scene,current,actions,0,wp,dt=dt,learned=learned)
    if loader=='rod':
        from .rod_caller_window import enabled,step
        if enabled(scene):
            return step(network,scene,current,actions,0,wp,dt=dt,learned=learned)
        from .rod_contact_data import apply_prescribed_input,query
        current=apply_prescribed_input(scene,current,actions,0)
        query(scene,current,wp)
    elif loader=='legacy':
        current=controlled_start(scene,current,actions,0)
    elif loader in ('surface','self','mixed','robot_cloth'):
        from .general_joint_data import set_controls
        from .mixed_contact_data import apply_pin_input,query
        if loader!='mixed':
            set_controls(scene,actions,0)
        current=apply_pin_input(scene,current,actions,0)
        query(scene,current,wp)
    else:
        from .general_joint_data import set_controls,apply_prescribed_input
        from .rigid_contact_data import query
        if loader=='joint':
            set_controls(scene,actions,0)
            current=apply_prescribed_input(scene,current,actions,0)
        query(scene,current,wp)
    return network(scene,current,dt,learned=learned,**external_inputs(actions,slice(0,1)))
