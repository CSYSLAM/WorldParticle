"""Train a controlled rod window from its predicted state and caller memory."""
import math

from .rigid_caller_program import SCHEMA,CallerMemory,physical_state,attach_memory


def enabled(scene):
    value=scene.metadata.get('rigid_caller_training_schema')
    if value is not None and value!=SCHEMA:raise ValueError('Unknown rigid caller training schema')
    return value==SCHEMA


def step(network,scene,current,states,index,wp,*,before_model=None,dt=None,learned=True):
    """One own query and NN call, followed by an accepted caller transaction.

    Only exogenous scalar controls are read from the per-step dataset. Pose
    commands, feedback wrenches and the next frame's target are recomputed
    from current predictions. Raw output and physical history stay unedited.
    """
    if not enabled(scene) or not hasattr(scene,'caller_program'):
        raise ValueError('Explicit reconstructed caller program required')
    if not network.config.get('rod_caller_pose_history') or not network.config.get('rod_prescribed_fresh_inputs'):
        raise ValueError('Caller windows require explicit raw velocity history and fresh prescribed inputs')
    if any(name in states for name in ('input_force','input_torque','input_pin_x','input_pin_q')):
        raise ValueError('Caller windows must not expose recorded feedback wrenches or prescribed pose labels')
    if len(current.x)!=1:raise ValueError('One physical sequence per owned collision query required')
    memory=CallerMemory.from_history(current.history);physical=physical_state(current)
    program=scene.caller_program
    tick=scene.metadata['dt'] if dt is None else dt
    ratio=tick/scene.metadata['dt']
    elapsed_steps=round(ratio) if math.isfinite(ratio) else 0
    if elapsed_steps<1 or abs(ratio-elapsed_steps)>1e-8:
        raise ValueError('Caller windows require complete original physical clock ticks')
    if bool((memory.phase+elapsed_steps>program.steps_per_sample).any()):
        raise ValueError('Split the student window at the caller sample boundary')
    if index<0:raise ValueError('Nonnegative caller input index required')
    def action(name):
        value=states['input_caller_'+name][index:index+1]
        if len(value)!=1:raise ValueError('Complete exogenous caller controls required: '+name)
        return value
    prepared,force,torque=program.prepare(physical,memory,scene.gravity,action('stiffness'),action('damping'),action('picked_body'))
    if scene.metadata.get('finite_rod_training_schema'):
        scene.joint_controls=tuple(states[name][index:index+1] for name in ('input_joint_target','input_joint_target_rate','input_joint_effort'))
    from .rod_contact_data import query
    query(scene,prepared,wp)
    if before_model is not None:before_model()
    predicted,diagnostic=network(scene,prepared,tick,
                                 learned=learned,force=force,torque=torque)
    committed=program.commit(predicted,memory,diagnostic['accepted'],elapsed_steps=elapsed_steps)
    diagnostic.update(caller_input_schema=SCHEMA,caller_controls_from_predicted_state=True,
        caller_target_refreshed=bool(((committed.phase==0)&diagnostic['accepted']).any()))
    if not bool(diagnostic['accepted'].all()):return current,diagnostic
    return attach_memory(predicted,committed),diagnostic
