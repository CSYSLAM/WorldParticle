"""Choose physical adapters from authored degrees of freedom, using one NN."""
import numpy as np


def model_configuration(base,model,motion_budget,newton):
    result=dict(base);result['native_contact_motion_budget']=float(motion_budget)
    if model.particle_count:
        result['particle_self_contact_v58']=True
        if np.any(model.joint_type.numpy()==int(newton.JointType.ROD)):
            result.update(elastic_rod_self_contact=True,elastic_rod_refresh_self_geometry=True)
        else:result['self_contact_refresh_geometry']=True
    return result


def prepare(network,model,configuration,provenance,wp,newton,device,*,learned=True):
    dt=provenance['dt'];motion=provenance['motion_budget'];identifier='asset:'+provenance['name']
    kinds=model.joint_type.numpy();rod=np.any(kinds==int(newton.JointType.ROD))
    finite=np.any(~np.isin(kinds,[int(newton.JointType.ROD),int(newton.JointType.FREE)]))
    if model.particle_count:
        if rod:
            from .elastic_rod_scene import export_model
            from .elastic_rod_runtime import ElasticRodExampleSolver as Runtime
        else:
            from .self_contact_scene import export_model
            from .self_contact_runtime import SelfContactExampleSolver as Runtime
        scene=export_model(model,configuration,identifier,newton,dt,device,
                           motion_budget=motion,full_surface=True,broad_phase='kdtree')
        # The legacy self exporter records arbitrary dictionaries as text.
        # New authored assets carry explicit portable reciprocal filter maps.
        from .elastic_rod_configuration import decode_configuration
        scene.metadata['solver_configuration']=decode_configuration(configuration)
        runtime=Runtime(network,scene,dt,wp,newton,model,model.state(),learned=learned)
    elif rod:
        if finite:
            from .rod_joint_scene import export_model
            from .rod_joint_runtime import RodJointExampleSolver as Runtime
        else:
            from .rod_contact_scene import export_model
            from .rod_contact_runtime import RodContactExampleSolver as Runtime
        scene=export_model(model,configuration,identifier,newton,dt,device)
        runtime=Runtime(network,scene,model.state(),dt,wp,newton,model,learned=learned)
    else:
        from .general_joint_scene import export_model
        from .general_joint_runtime import JointExampleSolver
        if not model.body_count:raise ValueError('Asset contains no physical state')
        scene=export_model(model,configuration,identifier,newton,dt,device)
        runtime=JointExampleSolver(network,scene,scene.initial_state(model,wp),dt,wp,newton,model,learned=learned)
    scene.metadata['native_asset_provenance']=provenance
    return scene,runtime
