"""Record independent shared-model cloth_twist states without rendering each substep."""
import argparse
import functools
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/shared_normal_648_v220/shared_strict.pt')
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--frames',type=int,default=360)
    parser.add_argument('--device',default='cuda:0');parser.add_argument('--gpu-memory-gib',type=float,default=12.)
    parser.add_argument('--physics-device',choices=('cpu',),help='Optional CPU physical operators with CUDA neural blocks')
    parser.add_argument('--resolution',type=int,help='Explicit lower resolution cloth grid preview')
    parser.add_argument('--state-interval',type=int,default=0,help='Save exact native/accepted-history states every N video frames')
    parser.add_argument('--path-subdivision-depth',type=int,default=0)
    parser.add_argument('--resume-state',type=Path);parser.add_argument('--resume-recording',type=Path)
    parser.add_argument('--resume-smaller-timestep',action='store_true',help='Explicit own-state continuation with smaller physical steps')
    parser.add_argument('--neural-graphs',action='store_true');parser.add_argument('--profile',action='store_true')
    parser.add_argument('--neural-precision',choices=('fp32','bf16'),default='fp32')
    parser.add_argument('--simulation-substeps',type=int,help='Explicit numerical preview; changes h while retaining original60Hz controls')
    parser.add_argument('--self-motion-budget',type=float,default=.005)
    parser.add_argument('--duty-cycle',type=float,default=.25);args=parser.parse_args()
    if args.frames<1 or not 0<args.duty_cycle<=1:parser.error('Positive frames and duty in(0,1] required')
    if args.simulation_substeps is not None and args.simulation_substeps<1:parser.error('Positive simulation substeps required')
    if args.state_interval<0:parser.error('Nonnegative saved-state interval required')
    if not 0<=args.path_subdivision_depth<=8:parser.error('Path subdivision depth in [0,8] required')
    if bool(args.resume_state)!=bool(args.resume_recording):parser.error('Both own-state resume arguments required')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(args.device,ROOT)
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.self_contact_scene import export_model,authored_full_surface
    from wp_vbd.self_contact_runtime import SelfContactExampleSolver
    from wp_vbd.gpu_budget import GpuBudget
    from wp_vbd.neural_cuda_graph import enable_attention_replay,attention_replay_report
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    budget=GpuBudget(args.device,args.gpu_memory_gib)
    physical_device=args.physics_device or args.device
    if args.physics_device and not args.device.startswith('cuda'):parser.error('CPU physics routing requires a CUDA neural device')
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(physical_device)
    if physical_device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    stored=torch.load(args.checkpoint,weights_only=True,map_location='cpu')
    config={**stored['config']['model'],'particle_self_contact_v58':True}
    network=MixedSolver(config).to(physical_device).eval();network.load_state_dict(stored['model'])
    if args.neural_graphs:enable_attention_replay(network)
    from wp_vbd.neural_precision import NeuralPrecision
    precision=NeuralPrecision(network.protected_response,args.neural_precision,args.device)
    bridge=None
    if args.physics_device:
        from wp_vbd.neural_device_bridge import NeuralDeviceBridge
        bridge=NeuralDeviceBridge(network.protected_response,args.device)
    viewer=newton.viewer.ViewerNull(num_frames=args.frames)
    example,configurations,_=make_example('cloth.example_cloth_twist',reference=False,viewer=viewer,preserve_defaults=True)
    assert len(configurations)==1
    preview_asset=None
    if args.resolution is not None:
        from wp_vbd.cloth_twist_preview import resize_example
        preview_asset=resize_example(example,args.resolution,wp,newton)
    original_dt=example.sim_dt
    if args.simulation_substeps is not None:
        example.sim_substeps=args.simulation_substeps;example.sim_dt=example.frame_dt/example.sim_substeps
    controller_clock={'kind':'native_source_gpu_clock'}
    if physical_device=='cpu':
        from wp_vbd.cloth_twist_cpu_clock import install
        controller_clock=install(example)
    calls={'construct':0,'step':0};constructor=newton.solvers.SolverVBD.__init__
    @functools.wraps(constructor)
    def forbidden(*unused,**kwargs):calls['construct']+=1;raise AssertionError('No full VBD during model recording')
    def forbidden_step(*unused,**kwargs):calls['step']+=1;raise AssertionError('No full VBD step during recording')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden_step
    scene=export_model(example.model,configurations[0][1],'cloth.example_cloth_twist',newton,example.sim_dt,physical_device,
        motion_budget=args.self_motion_budget,full_surface=authored_full_surface(example),broad_phase='kdtree',max_rows=1000000,
        path_subdivision_depth=args.path_subdivision_depth)
    solver=SelfContactExampleSolver(network,scene,example.sim_dt,wp,newton,example.model,example.state_0,learned=True)
    original_step=solver.step
    @functools.wraps(original_step)
    def physical_step(*a,**kw):
        precision.verify_physical_boundary();value=original_step(*a,**kw);precision.verify_physical_boundary();return value
    solver.step=physical_step
    example.solver=solver;neural={'calls':0,'seconds':0.};query={'calls':0,'seconds':0.};neural_begin=[]
    def sync():
        if args.device.startswith('cuda'):torch.cuda.synchronize()
    def before(*unused):
        if args.profile:sync();neural_begin.append(time.perf_counter())
    def after(*unused):
        neural['calls']+=1
        if args.profile:sync();neural['seconds']+=time.perf_counter()-neural_begin.pop()
    network.protected_response.register_forward_pre_hook(before);network.protected_response.register_forward_hook(after)
    original_query=solver.oracle.query
    def measured_query(*a,**kw):
        began_query=time.perf_counter();result=original_query(*a,**kw);query['calls']+=1
        if args.profile:sync()
        query['seconds']+=time.perf_counter()-began_query;return result
    solver.oracle.query=measured_query
    positions=np.lib.format.open_memmap(args.output/'particle_position.npy',mode='w+',dtype=np.float32,
        shape=(args.frames+1,example.model.particle_count,3))
    positions[0]=example.state_0.particle_q.numpy();positions.flush()
    np.savez_compressed(args.output/'static.npz',rest=example.model.particle_q.numpy(),faces=example.faces,
        pinned=scene.fixed_particle_ids.cpu().numpy())
    began=time.perf_counter();frames=0;failure=None;timings=[]
    result={'status':'RECORDING_INDEPENDENT_MODEL_STATES','demo':'cloth.example_cloth_twist','requested_frames':args.frames,
        'frames':0,'physical_seconds':0.,'physical_dt':example.sim_dt,'physical_substeps_per_frame':example.sim_substeps,
        'frame_dt':example.frame_dt,'source_fps':example.fps,'device':args.device,'physical_device':physical_device,'checkpoint_step':stored['step'],
        'original_physical_dt':original_dt,'explicit_numerical_preview':original_dt!=example.sim_dt,
        'preview_asset':preview_asset,'explicit_geometry_preview':preview_asset is not None,
        'rotation_angular_velocity_rad_s':example.rot_angular_velocity,'rotation_end_seconds':example.rot_end_time,
        'controller_clock':controller_clock,
        'self_motion_budget_m':args.self_motion_budget,
        'self_path_subdivision_depth':args.path_subdivision_depth,
        'time_step_schedule':[{'first_frame':0,'physical_dt':example.sim_dt,'substeps_per_frame':example.sim_substeps}],
        'checkpoint_sha256':digest(args.checkpoint),'effective_model_configuration':config,'failure':None,
        'teacher_trajectory_reads':0,'new_optimizer_updates':0,'duty_cycle':args.duty_cycle,
        'source_git_commit':__import__('subprocess').check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'script_sha256':digest(Path(__file__)),'module_sha256':{p.name:digest(p) for p in (ROOT/'src/wp_vbd').glob('*.py')},
        'scope':'Actual independent shared-model dynamics; source controller and four-round budget. Explicit geometry/time-step preview settings and any own-state continuation are recorded. No rendering during simulation, teacher input or default-asset quality certificate.'}
    if args.resume_state:
        saved=torch.load(args.resume_state,weights_only=True,map_location='cpu')
        parent_raw=(args.resume_recording/'recording.json').read_bytes();parent=json.loads(parent_raw)
        if saved['checkpoint_sha256']!=result['checkpoint_sha256'] or saved['effective_model_configuration']!=config:
            raise ValueError('Own-state continuation must retain the exact checkpoint/configuration')
        if any(saved[key]!=expected for key,expected in [('frame_dt',example.frame_dt),
            ('preview_asset',preview_asset),('self_motion_budget_m',args.self_motion_budget)]):
            raise ValueError('Own-state continuation must retain asset, timing and detection coverage')
        changed_dt=saved['physical_dt']!=example.sim_dt
        if changed_dt and (not args.resume_smaller_timestep or example.sim_dt>=saved['physical_dt']):
            raise ValueError('Explicitly declare a smaller continuation timestep')
        if not parent.get('all_model_tensor_values_unchanged') or parent['teacher_trajectory_reads'] or parent['full_vbd_prediction_calls']!={'construct':0,'step':0}:
            raise ValueError('Only an actual unchanged-weight independent model recording may resume')
        if parent['checkpoint_sha256']!=saved['checkpoint_sha256'] or parent['module_sha256']!=saved['module_sha256']:
            raise ValueError('Saved own state must bind its source recording')
        if digest(args.resume_recording/'static.npz')!=saved['static_sha256']:raise ValueError('Saved source static archive changed')
        with np.load(args.resume_recording/'static.npz') as previous,np.load(args.output/'static.npz') as current:
            if any(not np.array_equal(previous[k],current[k]) for k in previous.files):raise ValueError('Native preview asset changed')
        implementation_changes={k for k in result['module_sha256'] if result['module_sha256'][k]!=saved['module_sha256'].get(k)}
        allowed={'self_contact_geometry.py','self_contact_query.py','self_contact_runtime.py','self_contact_step.py','self_contact_scene.py'}
        if set(result['module_sha256'])!=set(saved['module_sha256']) or not implementation_changes<=allowed:
            raise ValueError('Only the explicitly measured self-path refinement may change on this continuation')
        if implementation_changes and not args.path_subdivision_depth:raise ValueError('Declare the measured path refinement')
        frames=saved['frames']
        schedule=saved.get('time_step_schedule',[{'first_frame':0,'physical_dt':saved['physical_dt'],
            'substeps_per_frame':round(saved['frame_dt']/saved['physical_dt'])}])
        expected_steps=sum((min(frames,schedule[i+1]['first_frame'] if i+1<len(schedule) else frames)-row['first_frame'])*row['substeps_per_frame']
            for i,row in enumerate(schedule) if row['first_frame']<frames)
        if not 0<frames<args.frames or frames>parent['frames'] or saved['physical_substeps']!=expected_steps:
            raise ValueError('A complete accepted source-frame boundary is required')
        previous=np.load(args.resume_recording/'particle_position.npy',mmap_mode='r')
        if not np.array_equal(previous[frames],saved['native']['0_particle_q'].numpy()):raise ValueError('Saved native pose differs from the source frame')
        positions[:frames+1]=previous[:frames+1];positions.flush()
        for index,native in enumerate((example.state_0,example.state_1)):
            for name in ('particle_q','particle_qd','particle_f','body_q','body_qd','body_f'):
                if f'{index}_{name}' in saved['native']:wp.to_torch(getattr(native,name)).copy_(saved['native'][f'{index}_{name}'])
        solver.history={k:v.to(physical_device) for k,v in saved['history'].items()};solver.steps=saved['physical_substeps']
        example.sim_time=saved['sim_time'];wp.to_torch(example.t).copy_(saved['controller_time'])
        if changed_dt:
            # Preserve the last accepted command phase and its angular rate:
            # next target = previous target + new h, rather than retaining a
            # large old-h target jump in a smaller physical interval.
            wp.to_torch(example.t).add_(example.sim_dt-saved['physical_dt'])
            schedule=[*schedule,{'first_frame':frames,'physical_dt':example.sim_dt,'substeps_per_frame':example.sim_substeps}]
        result['time_step_schedule']=schedule
        result['explicit_numerical_preview']=any(row['physical_dt']!=original_dt for row in schedule)
        result['resume']={'own_state':str(args.resume_state.resolve()),'own_state_sha256':digest(args.resume_state),
            'source_recording':str(args.resume_recording.resolve()),'source_recording_sha256':hashlib.sha256(parent_raw).hexdigest(),
            'starting_frame':frames,'starting_physical_seconds':frames*example.frame_dt,
            'source_module_changes':{k:{'before':saved['module_sha256'][k],'after':result['module_sha256'][k]} for k in sorted(implementation_changes)},
            'weights_asset_history_and_native_buffers_preserved':True,'physical_dt_preserved':not changed_dt,
            'controller_phase_adjustment_seconds':example.sim_dt-saved['physical_dt'] if changed_dt else 0.,'teacher_state_reads':0}
    session_start_frame=frames
    session_start_steps=solver.steps
    def persist():
        result.update(frames=frames,physical_substeps=solver.steps,physical_seconds=frames*example.frame_dt,
            accepted_physical_seconds=session_start_frame*example.frame_dt+(solver.steps-session_start_steps)*example.sim_dt,
            session_start_frame=session_start_frame,
            wall_seconds=time.perf_counter()-began,neural_response=neural,geometry_query=query,
            collision_queries=solver.oracle.queries,full_vbd_prediction_calls=calls,
            gpu_resource_budget=budget.report(),neural_graph_replay=attention_replay_report(network),neural_precision=precision.report())
        if bridge is not None:result['neural_device_bridge']=bridge.report()
        temporary=args.output/'recording.tmp';temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(args.output/'recording.json')
    def save_state():
        source={}
        for index,native in enumerate((example.state_0,example.state_1)):
            for name in ('particle_q','particle_qd','particle_f','body_q','body_qd','body_f'):
                value=getattr(native,name,None)
                if value is not None:source[f'{index}_{name}']=wp.to_torch(value).detach().cpu().clone()
        saved={'frames':frames,'physical_substeps':solver.steps,'frame_dt':example.frame_dt,'physical_dt':example.sim_dt,
            'sim_time':example.sim_time,'controller_time':wp.to_torch(example.t).detach().cpu().clone(),'native':source,
            'history':{k:v.detach().cpu().clone() for k,v in solver.history.items()},
            'checkpoint_sha256':result['checkpoint_sha256'],'script_sha256':result['script_sha256'],
            'module_sha256':result['module_sha256'],'effective_model_configuration':config,'preview_asset':preview_asset,
            'self_motion_budget_m':args.self_motion_budget,'static_sha256':digest(args.output/'static.npz'),
            'time_step_schedule':result['time_step_schedule'],
            'scope':'Own independent model accepted state at a complete video-frame boundary, including both native buffers and causal history. No teacher state or model weights included.'}
        target=args.output/f'state_frame_{frames:06d}.pt';temporary=target.with_suffix('.tmp')
        torch.save(saved,temporary);temporary.replace(target)
    persist()
    try:
        while frames<args.frames:
            start=time.perf_counter();example.step();sync();timings.append(time.perf_counter()-start)
            frames+=1;positions[frames]=example.state_0.particle_q.numpy();positions.flush();budget.observe()
            persist();print(json.dumps({'frames':frames,'physical_seconds':result['physical_seconds'],'frame_compute_seconds':timings[-1],'wall_seconds':result['wall_seconds']}),flush=True)
            if args.state_interval and frames%args.state_interval==0:save_state()
            if args.device.startswith('cuda'):
                remaining=(time.perf_counter()-start)*(1/args.duty_cycle-1)
                while remaining>0:delay=min(.25,remaining);time.sleep(delay);remaining-=delay
    except (RuntimeError,ValueError,AssertionError) as error:failure=repr(error)
    finally:
        viewer.close();precision.close()
        if bridge is not None:bridge.close()
    unchanged=all(torch.equal(v,network.state_dict()[k].detach().cpu()) for k,v in stored['model'].items())
    assert unchanged and calls=={'construct':0,'step':0}
    result.update(status='MODEL_RECORDING_COMPLETE' if frames==args.frames and failure is None else 'MODEL_RECORDING_INCOMPLETE',
        failure=failure,all_model_tensor_values_unchanged=unchanged,frame_compute_seconds=timings,
        static_sha256=digest(args.output/'static.npz'))
    if failure and solver.last_diagnostic is not None:
        def serial(value):
            if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
            if isinstance(value,dict):return {k:serial(v) for k,v in value.items()}
            if isinstance(value,(tuple,list)):return [serial(v) for v in value]
            return value
        diagnostic_path=args.output/'terminal_solver_diagnostic.json'
        diagnostic_path.write_text(json.dumps(serial(solver.last_diagnostic),indent=2)+'\n')
        result['terminal_solver_diagnostic']=str(diagnostic_path)
    persist();print(json.dumps({k:result[k] for k in ['status','frames','physical_seconds','failure','wall_seconds']},indent=2),flush=True)
    if failure or frames!=args.frames:raise SystemExit(1)


if __name__=='__main__':main()
