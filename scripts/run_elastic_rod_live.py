"""Native five-family window driven by one shared checkpoint, without VBD."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters',type=Path,default=ROOT/'configs/native_five_families.json')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/shared_normal_648_v161/shared_step_000832.pt')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--software-rendering',action='store_true',help='Use CPU OpenGL when the GPU is occupied')
    parser.add_argument('--screenshot',type=Path)
    parser.add_argument('--frames',type=int,default=300)
    parser.add_argument('--substeps',type=int,default=10)
    parser.add_argument('--dt',type=float,default=1/600)
    parser.add_argument('--motion-budget',type=float,default=.01)
    parser.add_argument('--refresh-self-geometry',action=argparse.BooleanOptionalAction,default=None,
                        help='Explicit override; otherwise preserve the checkpoint geometry policy')
    parser.add_argument('--physics-only',action='store_true',help='Explicit fixed-budget ablation with learned corrections disabled')
    parser.add_argument('--friction',type=float)
    parser.add_argument('--elastic-scale',type=float)
    parser.add_argument('--mass-scale',type=float)
    parser.add_argument('--gpu-memory-gib',type=float,default=12.)
    parser.add_argument('--duty-cycle',type=float,default=.25)
    parser.add_argument('--report',type=Path)
    args=parser.parse_args()
    if args.frames<=0 or args.substeps<=0 or not math.isfinite(args.dt) or args.dt<=0:
        parser.error('Positive frame/substep counts and finite positive dt required')
    if not math.isfinite(args.duty_cycle) or not 0<args.duty_cycle<=1:
        parser.error('Duty cycle must lie in (0,1]')
    if not math.isfinite(args.motion_budget) or args.motion_budget<=0:parser.error('Finite positive motion budget required')
    parameters=json.loads(args.parameters.read_text())
    if not isinstance(parameters,dict) or set(parameters)-{'friction','elastic_scale','mass_scale'}:
        parser.error('The five-family asset accepts friction, elastic_scale and mass_scale')
    for key,default in [('friction',.35),('elastic_scale',1.),('mass_scale',1.)]:
        value=getattr(args,key)
        if value is not None:parameters[key]=value
        else:parameters.setdefault(key,default)
    if any(type(v) not in (int,float) or not math.isfinite(v) for v in parameters.values()) or parameters['friction']<0 or min(parameters['elastic_scale'],parameters['mass_scale'])<=0:
        parser.error('Finite nonnegative friction and positive elastic/mass scales required')
    if args.report is None:
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
        args.report=ROOT/'reports/native_five_families'/f'live_{stamp}.json'
    if args.report.exists():raise FileExistsError(args.report)
    if args.screenshot is not None and (args.headless or args.screenshot.exists()):
        parser.error('Screenshot needs a window and a fresh output path')
    if args.software_rendering:
        if args.device!='cpu':parser.error('Software rendering requires --device cpu')
        os.environ['CUDA_VISIBLE_DEVICES']=''
        os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
        os.environ['MESA_LOADER_DRIVER_OVERRIDE']='llvmpipe'
        os.environ['__GLX_VENDOR_LIBRARY_NAME']='mesa'
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(args.device,ROOT)
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.elastic_rod_demo import build_asset
    from wp_vbd.elastic_rod_scene import export_model
    from wp_vbd.elastic_rod_runtime import ElasticRodExampleSolver
    from wp_vbd.gpu_budget import GpuBudget
    from wp_vbd.mixed_solver import MixedSolver
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    budget=GpuBudget(args.device,args.gpu_memory_gib)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(args.device)
    if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    configuration={k:v.default for k,v in inspect.signature(newton.solvers.SolverVBD).parameters.items()
                   if k not in ('model','collision_pipeline')}
    configuration.update(particle_enable_self_contact=True,particle_self_contact_margin=.05,particle_self_contact_gap=.02)
    calls={'construct':0,'step':0};original=newton.solvers.SolverVBD.__init__
    @functools.wraps(original)
    def forbidden(*unused,**kwargs):calls['construct']+=1;raise AssertionError('Complete VBD forbidden in mixed inference')
    def forbidden_step(*unused,**kwargs):calls['step']+=1;raise AssertionError('Complete VBD step forbidden in mixed inference')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden_step
    stored=torch.load(args.checkpoint,weights_only=True,map_location=args.device)
    model_configuration={**stored['config']['model'],'elastic_rod_self_contact':True}
    if args.refresh_self_geometry is not None:
        model_configuration.update(self_contact_refresh_geometry=args.refresh_self_geometry,
                                   elastic_rod_refresh_self_geometry=args.refresh_self_geometry)
    network=MixedSolver(model_configuration).to(args.device).eval()
    network.load_state_dict(stored['model'])
    neural_calls={'protected_response':0}
    def observe_neural(module,inputs,output):
        neural_calls['protected_response']+=1
        if args.physics_only:raise AssertionError('Neural response executed in the physical ablation')
    neural_hook=network.protected_response.register_forward_hook(observe_neural)
    asset,description=build_asset(newton,wp,device=args.device,**parameters)
    scene=export_model(asset,configuration,'research.native_five_families',newton,args.dt,args.device,
                       motion_budget=args.motion_budget,full_surface=True)
    from wp_vbd.self_contact_refresh import enabled as refresh_self_geometry
    refreshed_geometry=refresh_self_geometry(network.config,scene.metadata.get('contract'))
    native=asset.state();output=asset.state();control=asset.control()
    solver=ElasticRodExampleSolver(network,scene,args.dt,wp,newton,asset,native,learned=not args.physics_only)
    options={'enable_cuda_interop':newton.viewer.ViewerGL.CudaInterop.NONE} if args.software_rendering else {}
    viewer=newton.viewer.ViewerNull(num_frames=args.frames) if args.headless else newton.viewer.ViewerGL(width=1600,height=900,vsync=False,**options)
    viewer.set_model(asset)
    renderer_name=None
    if not args.headless:
        from pyglet import gl
        renderer_name=gl.gl_info.get_renderer()
        if args.software_rendering and not any(name in renderer_name.lower() for name in ('llvmpipe','softpipe','swrast')):
            viewer.close();raise RuntimeError('Requested software OpenGL renderer is not active: '+renderer_name)
        viewer.renderer.set_title('WorldParticle | mixed physics inference')
        viewer.set_camera(pos=wp.vec3(.55,-.7,.5))
        viewer.camera.look_at((.1,.07,.05))
        viewer.camera.pan(-.12,0.)
    trajectory=[]
    def record():
        trajectory.append((native.particle_q.numpy().copy(),native.body_q.numpy().copy()))
    record();frames=0;failure=None;begin=time.perf_counter()
    try:
        while viewer.is_running() and frames<args.frames:
            if viewer.should_step():
                for _ in range(args.substeps):
                    started=time.perf_counter()
                    native.clear_forces();viewer.apply_forces(native)
                    solver.step(native,output,control,None,args.dt);native,output=output,native
                    budget.observe();record()
                    if args.device.startswith('cuda'):
                        torch.cuda.synchronize();delay=(time.perf_counter()-started)*(1/args.duty_cycle-1)
                        while delay>0:
                            part=min(delay,.25);time.sleep(part);delay-=part
                frames+=1
                if frames%10==0:print(f'frames={frames} steps={solver.steps} t={solver.steps*args.dt:.3f}',flush=True)
            viewer.begin_frame(solver.steps*args.dt);viewer.log_state(native);viewer.end_frame()
        if args.screenshot is not None:
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(args.screenshot)
    except (ValueError,RuntimeError,AssertionError) as error:failure=str(error)
    finally:viewer.close()
    budget.observe(force=True)
    complete=frames==args.frames and failure is None
    result={'status':'NATIVE_MIXED_INFERENCE_EXECUTED' if complete else 'NATIVE_MIXED_INFERENCE_INCOMPLETE',
        'frames':frames,'requested_frames':args.frames,'physical_substeps':solver.steps,
        'physical_seconds':solver.steps*args.dt,'seconds':time.perf_counter()-begin,'failure':failure,
        'device':args.device,'headless':args.headless,'asset':description,'scene_contract':scene.metadata['contract'],
        'opengl_renderer':renderer_name,'software_rendering':args.software_rendering,
        'screenshot':str(args.screenshot) if args.screenshot else None,
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        'checkpoint_model_configuration':stored['config']['model'],'composition_enabled':True,
        'physical_dt':args.dt,'substeps_per_frame':args.substeps,'duty_cycle':args.duty_cycle,
        'motion_budget_m':args.motion_budget,
        'self_contact_refresh_geometry':refreshed_geometry,
        'learned_corrections_enabled':not args.physics_only,
        'neural_response_calls':neural_calls['protected_response'],
        'collision_queries':solver.oracle.queries,'history_keys':sorted(solver.history),
        'last_loaded_rows':{k:v for k,v in (solver.last_diagnostic or {}).items() if k.startswith('active_') or k.endswith('nonzero_wrench')},
        'gpu_resource_budget':budget.report(),'full_vbd_calls':calls,'teacher_trajectory_reads':0,
        'new_optimizer_updates':0,'new_training_admissions':0,
        'module_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/wp_vbd').glob('*.py')},
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':('Independent fixed-budget physical ablation with neural response hooks forbidden. ' if args.physics_only else
                 'Independent shared-model transfer on a fixed small native five-family asset with supplied physical parameters. ')+
                'Executed frames do not establish quality, arbitrary-asset support or realtime.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.report.with_suffix('.npz'),particle_q=np.stack([r[0] for r in trajectory]),
                        body_q=np.stack([r[1] for r in trajectory]),physical_substep=np.arange(len(trajectory)))
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('module_sha256','checkpoint_model_configuration')},indent=2))
    if not complete:raise SystemExit(1)


if __name__=='__main__':main()
