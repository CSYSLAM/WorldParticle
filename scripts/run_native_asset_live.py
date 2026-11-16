"""Load native geometry/physical parameters and render shared-model inference."""
import argparse
import datetime
import functools
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset',type=Path,default=ROOT/'assets/live/native_five_families.json')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/shared_normal_648_v161/shared_step_000832.pt')
    parser.add_argument('--set',action='append',default=[],dest='overrides',help='Override an existing asset field, e.g. contact.friction=0.6')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--headless',action='store_true');parser.add_argument('--software-rendering',action='store_true')
    parser.add_argument('--frames',type=int,default=300);parser.add_argument('--substeps',type=int)
    parser.add_argument('--physics-only',action='store_true');parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--neural-precision',choices=('fp32','bf16'),default='fp32',help='Optional neural inference precision; physical operators keep their original dtype')
    parser.add_argument('--protect-complete-response',action='store_true',help='Experimental bounded neural gains on complete physical coarse spaces')
    parser.add_argument('--max-query-motion',type=float,help='Experimental C222 complete query envelopes up to this explicit maximum, in metres')
    parser.add_argument('--gpu-memory-gib',type=float,default=12.);parser.add_argument('--duty-cycle',type=float,default=.25)
    parser.add_argument('--report',type=Path);args=parser.parse_args()
    if args.frames<1 or args.substeps is not None and args.substeps<1:parser.error('Positive frame/substep budgets required')
    if not math.isfinite(args.duty_cycle) or not 0<args.duty_cycle<=1:parser.error('Finite duty cycle in (0,1] required')
    if not math.isfinite(args.gpu_memory_gib) or args.gpu_memory_gib<=2:parser.error('Reserve two GiB for native allocations')
    if args.software_rendering and args.device!='cpu':parser.error('Software rendering requires CPU inference')
    if args.screenshot is not None and (args.headless or args.screenshot.exists()):parser.error('A fresh screenshot path and a native window are required')
    if args.report is None:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        args.report=ROOT/'reports/native_assets'/f'live_{stamp}.json'
    if any(p.exists() for p in (args.report,args.report.with_suffix('.npz'),args.report.with_suffix('.asset.npz'))):raise FileExistsError(args.report)
    from wp_vbd.native_assets import read,build
    spec=read(args.asset,args.overrides)
    if args.software_rendering:
        os.environ.update(CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',__GLX_VENDOR_LIBRARY_NAME='mesa')
    elif args.headless and args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    from wp_vbd.gpu_ownership import claim_cuda
    # Hardware rendering also owns a GPU slot when simulation runs on CPU.
    owned_device='cuda:0' if not args.headless and not args.software_rendering else args.device
    ownership=claim_cuda(owned_device,ROOT)
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.native_asset_runtime import model_configuration,prepare
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.gpu_budget import GpuBudget
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(args.device)
    if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    # build() inspects the public constructor signature for physical defaults;
    # preserve that signature while forbidding every actual full solver call.
    calls={'construct':0,'step':0};original=newton.solvers.SolverVBD.__init__
    @functools.wraps(original)
    def forbidden(*unused,**kwargs):calls['construct']+=1;raise AssertionError('Full VBD forbidden during asset inference')
    def forbidden_step(*unused,**kwargs):calls['step']+=1;raise AssertionError('Full VBD step forbidden during asset inference')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden_step
    budget=GpuBudget(args.device,args.gpu_memory_gib)
    model,configuration,provenance=build(spec,newton,wp,device=args.device)
    stored=torch.load(args.checkpoint,weights_only=True,map_location=args.device)
    effective=model_configuration(stored['config']['model'],model,provenance['motion_budget'],newton)
    if args.protect_complete_response:
        if not effective.get('shared_constitutive_coarse'):parser.error('Complete-response policy requires the shared constitutive coarse head')
        if effective.get('relaxation')!=.8:parser.error('Complete-response bound currently requires relaxation0.8')
        effective['protected_complete_response']=True
    network=MixedSolver(effective).to(args.device).eval();network.load_state_dict(stored['model'])
    from wp_vbd.neural_precision import NeuralPrecision
    precision=NeuralPrecision(network.protected_response,args.neural_precision,args.device)
    neural_calls={'count':0}
    def observe(module,inputs,output):
        neural_calls['count']+=1
        if args.physics_only:raise AssertionError('Neural response executed in a physical control')
    hook=network.protected_response.register_forward_hook(observe)
    scene,solver=prepare(network,model,configuration,provenance,wp,newton,args.device,learned=not args.physics_only)
    if args.max_query_motion is not None:
        from wp_vbd.query_budget import QueryBudgetBank
        solver.query_budget_policy=QueryBudgetBank(solver,args.max_query_motion,observe_memory=lambda:budget.observe(force=True))
    from wp_vbd.reference_tape import arrays
    static=arrays(model);descriptor=args.report.with_suffix('.asset.npz');descriptor.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(descriptor,metadata=json.dumps({'schema':'native_shared_asset_static_v250',
        'scene_contract':scene.metadata['contract'],'provenance':provenance}),**{'source__'+k:v for k,v in static.items()})
    native=model.state();outgoing=model.state();control=model.control();substeps=args.substeps or provenance['substeps'];dt=provenance['dt']
    options={'enable_cuda_interop':newton.viewer.ViewerGL.CudaInterop.NONE} if args.software_rendering else {}
    viewer=newton.viewer.ViewerNull(num_frames=args.frames) if args.headless else newton.viewer.ViewerGL(width=1600,height=900,vsync=False,**options)
    viewer.set_model(model);renderer=None
    if not args.headless:
        from pyglet import gl
        renderer=gl.gl_info.get_renderer()
        if args.software_rendering and not any(s in renderer.lower() for s in ('llvmpipe','softpipe','swrast')):
            viewer.close();raise RuntimeError('Requested software renderer was not activated')
        viewer.renderer.set_title('WorldParticle | asset inference | '+provenance['name'])
        initial=np.concatenate(([model.particle_q.numpy()] if model.particle_count else [])+
                               ([model.body_q.numpy()[:,:3]] if model.body_count else []),axis=0)
        center=(initial.min(0)+initial.max(0))/2;extent=max(float(np.ptp(initial,axis=0).max()),.25)
        viewer.set_camera(pos=wp.vec3(*(center+extent*np.array([1.1,-1.9,1.2]))));viewer.camera.look_at(tuple(center))
        viewer.camera.pan(-.3*extent,0.)
    trajectory=[];timings=[];frames=0;failure=None;began=time.perf_counter()
    def record():
        particles=native.particle_q.numpy().copy() if native.particle_q is not None else np.empty((0,3),np.float32)
        bodies=native.body_q.numpy().copy() if native.body_q is not None else np.empty((0,7),np.float32)
        trajectory.append((particles,bodies))
    record()
    try:
        while viewer.is_running() and frames<args.frames:
            if viewer.should_step():
                for _ in range(substeps):
                    start=time.perf_counter();native.clear_forces();viewer.apply_forces(native)
                    precision.verify_physical_boundary()
                    solver.step(native,outgoing,control,None,dt);native,outgoing=outgoing,native
                    precision.verify_physical_boundary()
                    if args.device.startswith('cuda'):torch.cuda.synchronize()
                    timings.append(time.perf_counter()-start);budget.observe();record()
                    if args.device.startswith('cuda'):
                        delay=(time.perf_counter()-start)*(1/args.duty_cycle-1)
                        while delay>0:duration=min(delay,.25);time.sleep(duration);delay-=duration
                frames+=1
                if frames%10==0:print(f'frames={frames} substeps={solver.steps} physical_time={solver.steps*dt:.3f}',flush=True)
            viewer.begin_frame(solver.steps*dt);viewer.log_state(native);viewer.end_frame()
        if args.screenshot is not None:
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True,exist_ok=True);Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(args.screenshot)
    except (ValueError,RuntimeError,AssertionError) as error:failure=str(error)
    finally:viewer.close();precision.close()
    budget.observe(force=True);complete=frames==args.frames and failure is None
    all_weights_unchanged=all(torch.equal(value,network.state_dict()[key]) for key,value in stored['model'].items())
    if not all_weights_unchanged:raise AssertionError('Asset inference changed shared model weights')
    if not args.physics_only and solver.steps and neural_calls['count']==0:raise AssertionError('No shared neural response executed')
    def portable(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
        if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
        if isinstance(value,(list,tuple)):return [portable(v) for v in value]
        return value
    failure_gates={}
    if failure is not None:
        keys=('accepted','finite','min_alpha','contact_motion_domain_valid','contact_motion_ray_scales',
            'geometric_step_domain_valid','geometric_step_caps','contact_activation_ray_scales',
            'predictor_motion_valid','predictor_physics_domain_valid','predictor_maximum_surface_travel',
            'final_physics_domain_valid')
        failure_gates={key:portable(solver.last_diagnostic[key]) for key in keys if key in (solver.last_diagnostic or {})}
    result={'status':'NATIVE_ASSET_INFERENCE_EXECUTED' if complete else 'NATIVE_ASSET_INFERENCE_INCOMPLETE',
        'asset':provenance,'scene_contract':scene.metadata['contract'],'frames':frames,'requested_frames':args.frames,
        'physical_substeps':solver.steps,'physical_dt':dt,'substeps_per_frame':substeps,'physical_seconds':solver.steps*dt,
        'seconds':time.perf_counter()-began,'device':args.device,'headless':args.headless,'renderer':renderer,
        'software_rendering':args.software_rendering,'failure':failure,'learned_corrections_enabled':not args.physics_only,
        'neural_precision':precision.report(),
        'failure_acceptance_diagnostic':failure_gates,
        'geometry_query_budget_policy':solver.query_budget_policy.report() if hasattr(solver,'query_budget_policy') else {'enabled':False},
        'neural_response_calls':neural_calls['count'],'active_neural_parameters':sum(p.numel() for p in network.protected_response.parameters()),
        'all_model_tensor_values_unchanged':all_weights_unchanged,
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'checkpoint_selected_step':stored.get('step'),
        'static_asset_descriptor':str(descriptor),'static_asset_descriptor_sha256':hashlib.sha256(descriptor.read_bytes()).hexdigest(),
        'effective_model_configuration':effective,'collision_queries':solver.oracle.queries,
        'last_diagnostic':{k:v for k,v in (solver.last_diagnostic or {}).items() if k.startswith('active_') or k.endswith('nonzero_wrench') or k=='minimum_contact_normal_physical_separation'},
        'solver_step_seconds_p50':float(np.median(timings)) if timings else None,
        'solver_step_seconds_p95':float(np.percentile(timings,95)) if timings else None,
        'solver_step_seconds_total':float(sum(timings)),
        'physical_seconds_per_solver_second':solver.steps*dt/sum(timings) if timings and sum(timings)>0 else None,
        'duty_cycle':args.duty_cycle,'gpu_resource_budget':budget.report(),'full_vbd_prediction_calls':calls,
        'teacher_trajectory_reads':0,'new_optimizer_updates':0,'new_training_admissions':0,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'implementation_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/wp_vbd').glob('*.py')},
        'scope':'Actual asset/physical-parameter input and independent native shared-model inference. Backend selected by physical contents. Execution does not certify trained asset/material generalization, nonpenetration, visual quality or realtime; those require independent measurement.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.report.with_suffix('.npz'),particle_q=np.stack([x[0] for x in trajectory]),body_q=np.stack([x[1] for x in trajectory]),physical_substep=np.arange(len(trajectory)))
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','scene_contract','physical_substeps','physical_seconds','failure','neural_response_calls','full_vbd_prediction_calls')},indent=2))
    if not complete:raise SystemExit(1)


if __name__=='__main__':main()
