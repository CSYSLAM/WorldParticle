"""Original Newton asset, controls and rendering with a learned shared solver."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
RIGID_CONTACT_DEMOS=('vbd.example_vbd_rigid_rigid_contact','contacts.example_balance_bird','contacts.example_domino_spiral','basic.example_basic_shapes')
GENERAL_JOINT_DEMOS=('basic.example_basic_joints','basic.example_basic_mimic_joint','contacts.example_newton_cradle','basic.example_basic_conveyor_forces','basic.example_basic_conveyor')
RIGID_BACKENDS=(*RIGID_CONTACT_DEMOS,*GENERAL_JOINT_DEMOS)


def main():
    from wp_vbd.official_adapter import ADMITTED
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--demo',choices=(*ADMITTED,*RIGID_CONTACT_DEMOS,*GENERAL_JOINT_DEMOS),default='softbody.example_softbody_hanging');p.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/shared_cycle_v10/shared_strict.pt');p.add_argument('--frames',type=int);p.add_argument('--headless',action='store_true');p.add_argument('--material-scale',type=float,default=1.);p.add_argument('--lambda-scale',type=float,default=1.);p.add_argument('--damping-scale',type=float,default=1.);p.add_argument('--friction',type=float);p.add_argument('--report',type=Path);p.add_argument('--test-final',action='store_true');p.add_argument('--screenshot',type=Path)
    for name,unit in [('joint-linear-ke','N/m'),('joint-angular-ke','Nm/rad'),('joint-linear-kd','Ns/m'),('joint-angular-kd','Nms/rad')]:p.add_argument('--'+name,type=float,help='Absolute finite joint material override in '+unit)
    p.add_argument('--contact-damping',type=float,help='Absolute shape kd override for the rigid-contact research backend')
    p.add_argument('--contact-stiffness',type=float,help='Absolute shape contact ke override [N/m]')
    p.add_argument('--motor-friction',type=float,help='Conveyor motor traction coefficient, separate from shape friction')
    p.add_argument('--belt-speed-scale',type=float,default=1.,help='Scale rotating belt speed or motor velocity fields, preserving any authored startup ramp')
    p.add_argument('--pitch',type=float,help='Lead-screw pitch in m/revolution for basic_mimic_joint')
    p.add_argument('--trajectory-output',type=Path,help='Headless diagnostic: save current native poses once per frame for offline behavior comparison')
    p.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0')
    p.add_argument('--gpu-memory-gib',type=float,default=12.,help='Process GPU budget; reserve two GiB outside the Torch allocator')
    p.add_argument('--eager',action='store_true',help='Run the same learned physical step without CUDA graph capture')
    p.add_argument('--neural-graphs',action='store_true',help='Experimental replay of pure attention blocks; all physics remains eager')
    p.add_argument('--disable-learning',action='store_true',help='Diagnostic ablation: same physical budget with neural response disabled')
    p.add_argument('--physical-vector',action='store_true',help='Same directional subspace and prescribed physical metric gate, without the NN')
    p.add_argument('--disable-hierarchy',action='store_true',help='Keep trained response weights and disable only the new hierarchical augmentation')
    a=p.parse_args()
    if a.neural_graphs and a.device=='cpu':p.error('--neural-graphs requires CUDA')
    if a.pitch is not None and a.demo!='basic.example_basic_mimic_joint':p.error('--pitch requires the lead-screw mimic demo')
    if a.motor_friction is not None and a.demo!='basic.example_basic_conveyor_forces':p.error('Motor traction requires the conveyor-force demo')
    if a.belt_speed_scale!=1. and a.demo not in ('basic.example_basic_conveyor_forces','basic.example_basic_conveyor'):p.error('Belt speed requires a conveyor demo')
    if a.contact_stiffness is not None and (a.demo not in RIGID_BACKENDS or not 0<=a.contact_stiffness<float('inf')):p.error('Contact stiffness requires a rigid backend and a finite nonnegative value')
    if a.motor_friction is not None and not 0<=a.motor_friction<float('inf') or not 0<=a.belt_speed_scale<float('inf'):p.error('Motor coefficients must be finite and nonnegative')
    if a.trajectory_output and not a.headless:p.error('--trajectory-output currently requires headless inference without parameter resets')
    if a.trajectory_output and a.trajectory_output.exists():raise FileExistsError('Diagnostic trajectories are immutable')
    if a.contact_damping is not None and (a.demo not in RIGID_BACKENDS or not 0<=a.contact_damping<float('inf')):p.error('--contact-damping requires a rigid contact/joint demo and a finite nonnegative value')
    joint_overrides={name:getattr(a,name) for name in ('joint_linear_ke','joint_angular_ke','joint_linear_kd','joint_angular_kd')}
    if any(value is not None for value in joint_overrides.values()) and a.demo not in GENERAL_JOINT_DEMOS:p.error('Joint material overrides require a general-joint demo')
    if any(value is not None and not 0<=value<float('inf') for value in joint_overrides.values()):p.error('Joint material coefficients must be finite and nonnegative')
    if a.disable_learning and a.physical_vector:p.error('Choose one physical ablation')
    runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    implementation={name:hashlib.sha256((ROOT/'src/wp_vbd'/name).read_bytes()).hexdigest() for name in ('rod_solver.py','rod_rows.py','rod_chain.py','mixed_solver.py','runtime.py','official_inference.py','protected_response.py','inverse_response.py','vector_response.py','vector_ports.py','heterogeneous_backbone.py','hierarchical_response.py','vector_transport.py','unified_graph.py','constitutive_coarse.py')}
    if a.demo in RIGID_BACKENDS:
        implementation.update({path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'src/wp_vbd').glob('rigid_contact*.py')})
        implementation['rigid_collision.py']=hashlib.sha256((ROOT/'src/wp_vbd/rigid_collision.py').read_bytes()).hexdigest()
        for name in ('official.py','contact_observables.py'):implementation[name]=hashlib.sha256((ROOT/'src/wp_vbd'/name).read_bytes()).hexdigest()
    if a.demo in GENERAL_JOINT_DEMOS:
        implementation.update({path.name:hashlib.sha256(path.read_bytes()).hexdigest() for path in (ROOT/'src/wp_vbd').glob('general_joint*.py')})
        implementation['mimic_constraint.py']=hashlib.sha256((ROOT/'src/wp_vbd/mimic_constraint.py').read_bytes()).hexdigest()
        for name in ('prescribed_motion.py','host_scalar_compat.py'):implementation[name]=hashlib.sha256((ROOT/'src/wp_vbd'/name).read_bytes()).hexdigest()
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.official_adapter import export_official
    from wp_vbd.official_inference import LearnedExampleSolver
    from wp_vbd.mixed_state import tensor_scene,initial_state
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.gpu_budget import GpuBudget
    implementation['gpu_budget.py']=hashlib.sha256((ROOT/'src/wp_vbd/gpu_budget.py').read_bytes()).hexdigest()
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    budget=GpuBudget(a.device,a.gpu_memory_gib)
    if a.material_scale<=0 or a.friction is not None and a.friction<0:p.error('Physical parameters must be positive/nonnegative')
    checkpoint_bytes=a.checkpoint.read_bytes();stored=torch.load(io.BytesIO(checkpoint_bytes),weights_only=True,map_location=a.device);model=MixedSolver(stored['config']['model']).to(a.device).eval();model.load_state_dict(stored['model'])
    from wp_vbd.neural_cuda_graph import enable_attention_replay,attention_replay_report
    if a.neural_graphs:enable_attention_replay(model)
    implementation['neural_cuda_graph.py']=hashlib.sha256((ROOT/'src/wp_vbd/neural_cuda_graph.py').read_bytes()).hexdigest()
    if a.demo in RIGID_BACKENDS and not model.config.get('shared_hierarchy'):p.error('The rigid contact/joint research backend requires a shared hierarchy checkpoint (v20 or newer)')
    if a.disable_hierarchy:
        if not stored['config']['model'].get('shared_hierarchy'):p.error('Hierarchical checkpoint required')
        if a.disable_learning or a.physical_vector:p.error('Hierarchy ablation retains the remaining neural response')
        model.protected_response.hierarchy_enabled=False
    if a.physical_vector:
        if not stored['config']['model'].get('protected_vector_ports'):raise ValueError('Directional checkpoint required')
        model.protected_response.mode='physical_vector'
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(a.device)
    viewer=newton.viewer.ViewerNull(num_frames=a.frames or 120) if a.headless else newton.viewer.ViewerGL(width=1920,height=1080,vsync=False)
    if not a.headless:viewer.renderer.set_title('WorldParticle | '+('physical vector subspace' if a.physical_vector else 'physical ablation' if a.disable_learning else 'shared inference')+' | '+a.demo)
    if not a.headless and a.demo=='basic.example_basic_mimic_joint':
        # Leave room for the mechanism; the native plots remain expandable.
        logger=viewer._plot_logger;draw_plots=logger.draw
        def compact_plots(ui):
            if logger._scalar_buffers or logger._array_buffers:ui.imgui.set_next_window_collapsed(True,ui.imgui.Cond_.once)
            draw_plots(ui)
        logger.draw=compact_plots
    original_init=newton.solvers.SolverVBD.__init__;original_step=newton.solvers.SolverVBD.step
    def forbidden(*unused,**kwargs):raise AssertionError('Full SolverVBD is forbidden in learned inference')
    def build_live(scale,friction,lambda_scale=1.,damping_scale=1.):
        if not all(np.isfinite(v) for v in (scale,lambda_scale,damping_scale)) or scale<=0 or min(lambda_scale,damping_scale)<0 or friction is not None and (not np.isfinite(friction) or friction<0):raise ValueError('Invalid material/friction parameter')
        if a.contact_damping is not None and (not np.isfinite(a.contact_damping) or a.contact_damping<0):raise ValueError('Contact damping must be finite and nonnegative')
        if any(value is not None and (not np.isfinite(value) or value<0) for value in (a.contact_stiffness,a.motor_friction)) or not np.isfinite(a.belt_speed_scale) or a.belt_speed_scale<0:raise ValueError('Contact stiffness, motor traction and belt speed must be finite and nonnegative')
        # Restore signatures for constructor argument introspection. The
        # helper substitutes InferenceOnly before constructing the Example.
        newton.solvers.SolverVBD.__init__=original_init;newton.solvers.SolverVBD.step=original_step
        arguments=('--pitch',str(a.pitch)) if a.pitch is not None else ()
        try:fresh,solvers,_=make_example(a.demo,reference=False,viewer=viewer,arguments=arguments,host_scalar_compat=a.demo=='basic.example_basic_conveyor')
        finally:newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden
        for name in ('tet_materials','tri_materials','edge_bending_properties'):
            array=getattr(fresh.model,name,None)
            if array is not None and array.size:
                value=array.numpy();value[:,:min(3,value.shape[1])]*=scale
                if name=='edge_bending_properties':value[:,1]*=damping_scale
                else:value[:,1]*=lambda_scale;value[:,2]*=damping_scale
                array.assign(value)
        if a.demo.startswith('vbd.example_cable_'):
            if lambda_scale!=1.:raise ValueError('Lamé lambda is not a rod material input; use material and damping scales')
            fresh.model.joint_target_ke.assign(fresh.model.joint_target_ke.numpy()*scale)
            fresh.model.joint_target_kd.assign(fresh.model.joint_target_kd.numpy()*damping_scale)
        if friction is not None:
            fresh.model.soft_contact_mu=friction
            value=fresh.model.shape_material_mu.numpy();value[:]=friction;fresh.model.shape_material_mu.assign(value)
        if a.motor_friction is not None:fresh.conveyor.conv_friction.assign(np.full_like(fresh.conveyor.conv_friction.numpy(),a.motor_friction))
        if a.demo=='basic.example_basic_conveyor_forces':
            for field in ('conv_const_vel','conv_pivot_angvel'):
                array=getattr(fresh.conveyor,field);array.assign(array.numpy()*a.belt_speed_scale)
        if a.demo=='basic.example_basic_conveyor':fresh.belt_angular_speed*=a.belt_speed_scale
        if a.demo in GENERAL_JOINT_DEMOS:
            from wp_vbd.general_joint_scene import export_model as export_joint_model
            from wp_vbd.general_joint_runtime import JointExampleSolver
            if lambda_scale!=1.:raise ValueError('Lamé lambda is not a rigid-joint parameter')
            configuration=solvers[0][1].copy()
            for key in ('rigid_joint_linear_ke','rigid_joint_angular_ke'):configuration[key]*=scale
            for key in ('rigid_joint_linear_kd','rigid_joint_angular_kd'):configuration[key]*=damping_scale
            for name in joint_overrides:
                value=getattr(a,name)
                if value is not None:
                    if not np.isfinite(value) or value<0:raise ValueError('Invalid joint material override')
                    configuration['rigid_'+name]=value
            fresh.model.joint_target_ke.assign(fresh.model.joint_target_ke.numpy()*scale)
            fresh.model.joint_target_kd.assign(fresh.model.joint_target_kd.numpy()*damping_scale)
            if a.contact_damping is not None:fresh.model.shape_material_kd.assign(np.full_like(fresh.model.shape_material_kd.numpy(),a.contact_damping))
            if a.contact_stiffness is not None:fresh.model.shape_material_ke.assign(np.full_like(fresh.model.shape_material_ke.numpy(),a.contact_stiffness))
            native_observables=getattr(fresh,'solver_observables',None)
            scene=export_joint_model(fresh.model,configuration,a.demo,newton,fresh.sim_dt,a.device,native_contact_export=native_observables is not None,prescribed_motion=bool(np.any(fresh.model.body_flags.numpy()==int(newton.BodyFlags.KINEMATIC))));metadata=scene.metadata
            scene.site_mu=wp.to_torch(fresh.model.shape_material_mu).double()
            state=scene.initial_state(fresh.model,wp);learned=JointExampleSolver(model,scene,state,fresh.sim_dt,wp,newton,fresh.model,learned=not a.disable_learning)
            if native_observables is not None:fresh.solver_observables=learned.observables(contacts=fresh.contacts,kinds=native_observables.kinds)
        elif a.demo in RIGID_CONTACT_DEMOS:
            from wp_vbd.rigid_contact_scene import export_model,RigidContactScene
            from wp_vbd.rigid_contact_runtime import ContactExampleSolver
            if lambda_scale!=1.:raise ValueError('Lamé lambda is not a rigid contact parameter; use material scale for shape contact ke')
            fresh.model.shape_material_ke.assign(fresh.model.shape_material_ke.numpy()*scale)
            if a.contact_stiffness is not None:fresh.model.shape_material_ke.assign(np.full_like(fresh.model.shape_material_ke.numpy(),a.contact_stiffness))
            kd=fresh.model.shape_material_kd.numpy()*damping_scale
            if a.contact_damping is not None:kd[:]=a.contact_damping
            fresh.model.shape_material_kd.assign(kd)
            static,metadata=export_model(fresh.model,solvers[0][1],a.demo,newton,fresh.sim_dt)
            scene=RigidContactScene(static,metadata,device=a.device)
            scene.site_mu=wp.to_torch(fresh.model.shape_material_mu).double()
            state=scene.initial_state(fresh.model,wp)
            learned=ContactExampleSolver(model,scene,state,fresh.sim_dt,wp,newton,fresh.model,learned=not a.disable_learning)
        else:
            static,metadata=export_official(fresh,solvers[0][1],a.demo,newton)
            scene=tensor_scene(static,metadata,a.device);state=initial_state(scene,fresh.model)
            capture=a.device.startswith('cuda') and not a.eager and not a.neural_graphs and not model.config.get('compact_neural_relations',False)
            learned=LearnedExampleSolver(model,scene,state,fresh.sim_dt,wp,learned=not a.disable_learning,capture=capture)
        if friction is not None and metadata.get('history_contract') and metadata.get('contacts')=='none':
            raise ValueError('This authored rod scene has no contacts; friction is not an active physical input')
        fresh.solver=learned
        fresh.applied_material_parameters={'material_scale':scale,'lambda_scale':lambda_scale,'damping_scale':damping_scale,'friction_override':friction,'contact_damping_override':a.contact_damping if a.demo in RIGID_CONTACT_DEMOS else None}
        if a.demo in RIGID_BACKENDS:fresh.applied_material_parameters['contact_stiffness_override']=a.contact_stiffness
        if a.demo=='basic.example_basic_conveyor_forces':fresh.applied_material_parameters.update(motor_friction_override=a.motor_friction,belt_speed_scale=a.belt_speed_scale)
        if a.demo=='basic.example_basic_conveyor':fresh.applied_material_parameters.update(belt_speed_scale=a.belt_speed_scale,belt_angular_speed_rad_per_s=fresh.belt_angular_speed)
        if a.demo in GENERAL_JOINT_DEMOS:
            fresh.applied_material_parameters.update({name:configuration['rigid_'+name] for name in joint_overrides})
            fresh.applied_material_parameters['contact_damping_override']=a.contact_damping
        if not a.headless:
            # Draw each accepted physical substep while preserving the
            # original frame/controller clock and original render method.
            def show_substep(state_out):
                previous=fresh.state_0;clock=fresh.sim_time
                fresh.state_0=state_out;fresh.sim_time=learned.steps*fresh.sim_dt
                try:
                    if viewer.is_running():fresh.render()
                finally:fresh.state_0=previous;fresh.sim_time=clock
            learned.on_substep=show_substep
        return fresh,learned
    example,solver=build_live(a.material_scale,a.friction,a.lambda_scale,a.damping_scale)
    if not a.headless and a.demo not in RIGID_BACKENDS:
        positions=solver.scene.rest_x;center=positions.mean(0).cpu().numpy()
        distance=max(.85,1.1*float((positions.amax(0)-positions.amin(0)).norm()))
        viewer.set_camera(wp.vec3(*(center+np.array([distance,-distance,.8*distance]))));viewer.camera.look_at(center)
        right=viewer.camera.get_right();viewer.camera.look_at(center-.2*distance*np.array([right.x,right.y,right.z]))
    failure=None;frames=0;begin=time.perf_counter();needs_reset=False;resets=0
    trajectory=[]
    def record_frame():
        if a.trajectory_output:
            particle=example.state_0.particle_q;body=example.state_0.body_q;velocity=example.state_0.body_qd
            trajectory.append((solver.steps,particle.numpy().copy() if particle is not None else np.empty((0,3),np.float32),body.numpy().copy() if body is not None else np.empty((0,7),np.float32),velocity.numpy().copy() if velocity is not None else np.empty((0,6),np.float32)))
    record_frame()
    pending_scale=a.material_scale;pending_friction=float(solver.scene.site_mu[0]) if len(solver.scene.site_mu) else float(example.model.soft_contact_mu)
    pending_lambda=a.lambda_scale;pending_damping=a.damping_scale
    pending_contact_damping=a.contact_damping if a.contact_damping is not None else 0.
    friction_override_requested=a.friction is not None;contact_damping_override_requested=a.contact_damping is not None
    pending_joint={name:solver.scene.metadata['solver_configuration']['rigid_'+name] for name in joint_overrides} if a.demo in GENERAL_JOINT_DEMOS else {}
    pending_pitch=getattr(example,'pitch',0.)
    pending_contact_ke=a.contact_stiffness if a.contact_stiffness is not None else float(example.model.shape_material_ke.numpy()[0]) if a.demo in RIGID_BACKENDS else 0.
    contact_ke_requested=a.contact_stiffness is not None
    pending_motor_mu=float(example.conveyor.conv_friction.numpy()[0]) if a.demo=='basic.example_basic_conveyor_forces' else 0.
    pending_belt_speed=a.belt_speed_scale;motor_mu_requested=a.motor_friction is not None
    joint_override_requested={name:value is not None for name,value in joint_overrides.items()}
    if not a.headless:
        viewer.gui.ui.io.set_ini_filename(str(ROOT/'.local/official-viewer.ini'))
        def ui(imgui):
            nonlocal pending_scale,pending_friction,pending_lambda,pending_damping,pending_contact_damping,pending_pitch,friction_override_requested,contact_damping_override_requested,needs_reset
            nonlocal pending_contact_ke,contact_ke_requested,pending_motor_mu,pending_belt_speed,motor_mu_requested
            imgui.text('Physical vector subspace' if a.physical_vector else 'Physical ablation: neural response disabled' if a.disable_learning else 'Shared learned inference')
            imgui.text_wrapped(a.demo)
            imgui.text(f'Physical time: {example.sim_time:.3f} s; substeps: {solver.steps}')
            imgui.text(f'Authored h: {example.sim_dt:g}')
            if a.demo=='basic.example_basic_conveyor_forces':
                imgui.set_next_item_width(260)
                changed,pending_motor_mu=imgui.input_float('Motor traction mu',pending_motor_mu,.1,0.,'%.3f');motor_mu_requested|=changed
            if a.demo in ('basic.example_basic_conveyor_forces','basic.example_basic_conveyor'):
                imgui.set_next_item_width(260)
                _,pending_belt_speed=imgui.input_float('Belt speed scale',pending_belt_speed,.1,0.,'%.3f')
            if a.demo in RIGID_BACKENDS:
                imgui.set_next_item_width(260)
                changed,pending_contact_ke=imgui.input_float('Contact ke override [N/m]',pending_contact_ke,100.,0.,'%.3g');contact_ke_requested|=changed
            if a.demo=='basic.example_basic_mimic_joint':
                imgui.set_next_item_width(260)
                _,pending_pitch=imgui.input_float('Screw pitch [m/rev]',pending_pitch,.1,0.,'%.3f')
                imgui.text(f'Nut travel: {example.nut_travel:.3f} m')
                imgui.text(f'Screw angle: {example.screw_angle:.3f} rad')
            imgui.set_next_item_width(260)
            _,pending_scale=imgui.input_float('Contact stiffness scale' if a.demo in RIGID_CONTACT_DEMOS else 'Joint stiffness scale' if a.demo in GENERAL_JOINT_DEMOS else 'Material scale',pending_scale,.05,0.,'%.3f')
            if a.demo in GENERAL_JOINT_DEMOS:
                for name,label in [('joint_linear_ke','Linear ke [N/m]'),('joint_angular_ke','Angular ke [Nm/rad]'),('joint_linear_kd','Linear kd [Ns/m]'),('joint_angular_kd','Angular kd [Nms/rad]')]:
                    imgui.set_next_item_width(260)
                    changed,pending_joint[name]=imgui.input_float(label,pending_joint[name],.05,0.,'%.3g');joint_override_requested[name]|=changed
            if solver.scene.metadata.get('particle_count',0)>0:
                imgui.set_next_item_width(260)
                _,pending_lambda=imgui.input_float('Lambda scale',pending_lambda,.05,0.,'%.3f')
            imgui.set_next_item_width(260)
            if a.demo in RIGID_CONTACT_DEMOS:
                changed,pending_contact_damping=imgui.input_float('Contact damping kd',pending_contact_damping,.05,0.,'%.3f')
                contact_damping_override_requested|=changed
            else:_,pending_damping=imgui.input_float('Damping scale',pending_damping,.05,0.,'%.3f')
            if len(solver.scene.site_mu):
                imgui.set_next_item_width(260)
                changed,pending_friction=imgui.input_float('Friction mu override',pending_friction,.05,0.,'%.3f')
                friction_override_requested|=changed
            if imgui.button('Apply parameters + reset'):needs_reset=True
            imgui.text_wrapped('Research model; official acceptance is in progress')
            if failure:imgui.text_wrapped(failure)
        viewer.register_ui_callback(ui,position='panel')
    stream=wp.stream_from_torch(torch.cuda.current_stream()) if a.device.startswith('cuda') else None
    def scope():return wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice('cpu')
    try:
        while viewer.is_running() and (a.frames is None or frames<a.frames):
            if needs_reset:
                try:
                    if contact_damping_override_requested:a.contact_damping=pending_contact_damping
                    if contact_ke_requested:a.contact_stiffness=pending_contact_ke
                    if motor_mu_requested:a.motor_friction=pending_motor_mu
                    a.belt_speed_scale=pending_belt_speed
                    if a.demo=='basic.example_basic_mimic_joint':a.pitch=pending_pitch
                    for name,requested in joint_override_requested.items():
                        if requested:setattr(a,name,pending_joint[name])
                    example,solver=build_live(pending_scale,pending_friction if friction_override_requested and len(solver.scene.site_mu) else None,pending_lambda,pending_damping);failure=None;resets+=1
                    if a.demo in GENERAL_JOINT_DEMOS:pending_joint.update({name:solver.scene.metadata['solver_configuration']['rigid_'+name] for name in pending_joint})
                except (ValueError,RuntimeError) as error:failure=str(error)
                needs_reset=False
            if viewer.should_step() and not failure:
                try:
                    with scope():
                        example.step()
                        if (a.test_final or a.demo=='basic.example_basic_conveyor_forces') and hasattr(example,'test_post_step'):example.test_post_step()
                    frames+=1
                    budget.observe()
                    record_frame()
                    if a.headless and frames%60==0:print(f'progress {a.demo}: {frames} frames, {solver.steps} physical substeps, t={example.sim_time:.3f}s',flush=True)
                except (RuntimeError,ValueError) as error:
                    failure=str(error)
                    if a.headless:break
            with scope():example.render()
        final_test='NOT_RUN';task_metrics=None
        if a.test_final and not failure:
            try:example.test_final();final_test='PASS'
            except Exception as error:final_test='FAIL: '+str(error)
            if hasattr(example,'analysis_metrics'):
                task_metrics=example.analysis_metrics()
            elif hasattr(example,'_measure_twists'):
                task_metrics={'twists_radians':example._measure_twists(),'target_tip_twist_radians':example.TARGET_TIP_TWIST}
            elif hasattr(example,'_measure_twist_profile'):
                task_metrics={'twist_profiles_radians':{case['label']:example._measure_twist_profile(case) for case in example.cases}}
            elif hasattr(example,'_measured_tip_state'):
                task_metrics={'tip_dz_dx_dy':example._measured_tip_state()}
            elif hasattr(example,'_metrics'):
                task_metrics={case['label']:example._metrics(case) for case in example.cases}
            elif hasattr(example,'measure_area_ratios'):
                source_module=sys.modules[type(example).__module__]
                task_metrics={'poisson_ratios':example.POISSON_RATIOS,'measured_area_ratios':example.measure_area_ratios(),'theoretical_area_ratios':[source_module._theoretical_area_ratio(example.STRETCH,nu) for nu in example.POISSON_RATIOS],'authored_stretch':example.STRETCH,'ramp_frames':example.RAMP_FRAMES,'ramp_completed':frames>=example.RAMP_FRAMES}
            if task_metrics is not None:task_metrics=json.loads(json.dumps(task_metrics,default=lambda value:value.tolist()))
        if a.screenshot and not a.headless and viewer.is_running():
            from PIL import Image
            a.screenshot.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(a.screenshot)
    finally:viewer.close()
    if a.device.startswith('cuda'):torch.cuda.synchronize()
    budget.observe(force=True)
    report={'demo':a.demo,'status':'FAIL' if failure or final_test.startswith('FAIL') else 'INFERENCE_EXECUTED','failure':failure,'frames':frames,'physical_substeps':solver.steps,'physical_seconds':solver.steps*example.sim_dt,'authored_frame_time':example.sim_time,'seconds':time.perf_counter()-begin,'dt':example.sim_dt,'checkpoint_sha256':hashlib.sha256(checkpoint_bytes).hexdigest(),'full_vbd_calls':0,'trajectory_files_read':0,'original_example_step_controls_and_render':True,'test_final':final_test,'parameter_resets':resets}
    report['gpu_resource_budget']=budget.report()
    report['neural_graph_replay']=attention_replay_report(model)
    if a.trajectory_output:
        a.trajectory_output.parent.mkdir(parents=True,exist_ok=True)
        static={name:getattr(example.model,name).numpy() if getattr(example.model,name,None) is not None else np.empty(shape,np.float32) for name,shape in (('body_mass',(0,)),('body_inertia',(0,3,3)),('body_com',(0,3)),('gravity',(0,3)))}
        np.savez_compressed(a.trajectory_output,physical_substep=np.asarray([r[0] for r in trajectory]),particle_x=np.stack([r[1] for r in trajectory]),body_pose=np.stack([r[2] for r in trajectory]),body_velocity=np.stack([r[3] for r in trajectory]),**static,metadata=np.asarray(json.dumps({'demo':a.demo,'checkpoint_sha256':report['checkpoint_sha256'],'dt':example.sim_dt,'body_position_representation':'native_body_origin','body_velocity_representation':'native COM linear velocity then world angular velocity','purpose':'Offline behavior diagnostics; poses/velocities/material constants, no complete restart history'})))
        report['trajectory_output']={'path':str(a.trajectory_output.resolve()),'sha256':hashlib.sha256(a.trajectory_output.read_bytes()).hexdigest(),'samples':len(trajectory),'purpose':'Offline pose comparison, no teacher input or restart claim','timing':'Per-frame observation copies included in seconds; final archive compression excluded'}
    report['applied_parameters']=example.applied_material_parameters
    report['runner_source_sha256_at_start']=runner_sha256
    report['learning_enabled']=not (a.disable_learning or a.physical_vector)
    report['device']=a.device
    if a.demo in RIGID_CONTACT_DEMOS:
        contact_key='rigid_contact_train_default' if a.demo=='vbd.example_vbd_rigid_rigid_contact' else a.demo.split('.')[-1].removeprefix('example_')+'_train_default'
        report['active_weight_contact_training_samples']=stored.get('active_weight_optimizer_samples',stored.get('sampled_updates',{})).get(contact_key,0)
        report['contact_training_status']='SAMPLED_IN_ACTIVE_WEIGHT_ANCESTRY' if report['active_weight_contact_training_samples']>0 else 'UNTRAINED_CONTACT_TRANSFER'
        report['collision_queries']=solver.oracle.queries
        report['contact_damping_override']=example.applied_material_parameters.get('contact_damping_override')
        report['contact_safety']='Expanded collision detection with a bounded surface-motion envelope; finite material penetration remains allowed. No global nonpenetration or CCD certificate.'
    if a.demo in GENERAL_JOINT_DEMOS:
        label='conveyor_train_default' if a.demo=='basic.example_basic_conveyor' else 'conveyor_forces_train_default' if a.demo=='basic.example_basic_conveyor_forces' else 'newton_cradle_train_default' if a.demo=='contacts.example_newton_cradle' else 'mimic_default.npz_train_default' if a.demo=='basic.example_basic_mimic_joint' else 'basic_joints_train_default'
        report['joint_training_samples_in_selected_trial']=stored.get('sampled_updates',{}).get(label,0)
        report['active_weight_joint_training_samples']=stored.get('active_weight_optimizer_samples',stored.get('sampled_updates',{})).get(label,0)
        report['joint_training_status']='SAMPLED_IN_ACTIVE_WEIGHT_ANCESTRY' if report['active_weight_joint_training_samples'] else 'UNTRAINED_JOINT_TRANSFER'
        report['joint_history_contract']=solver.scene.metadata['joint_history_contract'];report['joint_angle_prev']=solver.history['joint_angle_prev'].cpu().tolist();report['collision_queries']=solver.oracle.queries
        if solver.scene.metadata.get('prescribed_pose_history'):
            report['prescribed_pose_history_contract']=solver.scene.metadata['prescribed_pose_history'];report['complete_history_keys']=sorted(solver.history)
            report['prescribed_task']={'belt_angular_speed_rad_per_s':example.belt_angular_speed,'physical_belt_angle_rad':float(solver.history['joint_angle_prev'][0,0]),'bag_radii_m':np.linalg.norm(example.state_0.body_q.numpy()[example.bag_bodies,:2],axis=-1).tolist()}
        if solver.scene.metadata.get('native_contact_export'):
            report['native_contact_export_contract']=solver.scene.metadata['native_contact_export'];report['contact_exports']=solver.contact_exports
            report['conveyor_task']={'minimum_maximum_travel_m':float(np.min(example.max_travel)),'maximum_travel_by_body_m':example.max_travel.tolist(),'mean_final_speed_m_per_s':float(np.linalg.norm(example.state_0.body_qd.numpy()[example.tracked_bodies,:3],axis=1).mean()),'final_contact_force_norm':float(np.linalg.norm(example.solver_observables.contact_f.numpy()))}
        if a.demo=='basic.example_basic_mimic_joint':
            report['ideal_mimic_contract']=solver.scene.metadata['ideal_mimic_contract']
            report['mimic_task']={name:getattr(example,name) for name in ('pitch','coupling_ratio','max_abs_screw_angle','max_abs_coupling_error','screw_angle','nut_travel')}
    report['response_mode']='physical_vector' if a.physical_vector else 'physical' if a.disable_learning else 'learned_without_hierarchy' if a.disable_hierarchy else 'learned'
    active_model=model.protected_response if model.config.get('protected_learning') else model
    report['active_neural_parameters']=sum(parameter.numel() for parameter in active_model.parameters()) if not (a.disable_learning or a.physical_vector) else 0
    if a.disable_hierarchy:report['active_neural_parameters']-=sum(p.numel() for module in (model.protected_response.hierarchy,model.protected_response.transport) for p in module.parameters())
    if stored['config']['model'].get('shared_hierarchy'):report['hierarchy_enabled']=not a.disable_hierarchy
    report['execution_mode']='cuda_graph' if getattr(solver,'graph',None) is not None else 'eager'
    if report['neural_graph_replay']['captures']:report['execution_mode']='eager_physics_with_attention_replay'
    if model.config.get('protected_learning'):report['stored_legacy_parameters_unused']=sum(parameter.numel() for parameter in model.parameters())-sum(parameter.numel() for parameter in active_model.parameters())
    report['implementation_sha256']=implementation
    report['checkpoint_selected_update']=stored.get('step')
    if stored.get('operator_probe_provenance'):report['operator_probe_provenance']=stored['operator_probe_provenance']
    if task_metrics is not None:report['original_task_metrics']=task_metrics
    report.update(architecture=model.architecture_id,learned_parameters=sum(p.numel() for p in model.parameters()),simulation_to_wall_ratio=report['physical_seconds']/report['seconds'],authored_frames_per_second=frames/report['seconds'],history_contract=solver.scene.metadata.get('history_contract'))
    print(json.dumps(report,indent=2))
    if a.report:a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    if report['status']=='FAIL':sys.exit(1)

if __name__=='__main__':main()
