"""Research rigid-contact native viewer; geometry queries, no full VBD inference."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',default='vbd.example_vbd_rigid_rigid_contact')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/shared_hierarchy_v20/shared_strict.pt')
    parser.add_argument('--frames',type=int,default=300)
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--disable-learning',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--test-final',action='store_true')
    parser.add_argument('--contact-stiffness',type=float,help='Override all shape contact ke before constructing the research scene')
    parser.add_argument('--contact-damping',type=float,help='Override all shape contact kd')
    parser.add_argument('--friction',type=float,help='Override all shape mu; pair coefficients use the original geometric mean')
    parser.add_argument('--screenshot',type=Path)
    args=parser.parse_args()
    implementation={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/wp_vbd').glob('*.py')}
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.rigid_contact_scene import export_model,RigidContactScene
    from wp_vbd.rigid_contact_runtime import ContactExampleSolver
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache')
    wp.init();wp.set_device(args.device)
    raw=args.checkpoint.read_bytes();stored=torch.load(io.BytesIO(raw),weights_only=True,map_location=args.device)
    model=MixedSolver(stored['config']['model']).to(args.device).eval();model.load_state_dict(stored['model'])
    viewer=newton.viewer.ViewerNull(num_frames=args.frames) if args.headless else newton.viewer.ViewerGL(width=1600,height=900,vsync=False)
    example,configurations,authored_args=make_example(args.demo,reference=False,viewer=viewer)
    for value,name in ((args.contact_stiffness,'shape_material_ke'),(args.contact_damping,'shape_material_kd'),(args.friction,'shape_material_mu')):
        if value is not None:
            if not np.isfinite(value) or value<0:parser.error('Contact parameters must be finite and nonnegative')
            array=getattr(example.model,name);array.assign(np.full_like(array.numpy(),value))
    def forbidden(*unused,**kwargs):raise AssertionError('Full SolverVBD forbidden in contact inference')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden
    values,metadata=export_model(example.model,configurations[0][1],args.demo,newton,example.sim_dt)
    scene=RigidContactScene(values,metadata,device=args.device)
    solver=ContactExampleSolver(model,scene,scene.initial_state(example.model,wp),example.sim_dt,wp,newton,example.model,learned=not args.disable_learning)
    example.solver=solver
    if not args.headless:
        viewer.renderer.set_title('WorldParticle | shared rigid contact | research')
        center=scene.rest_x[:scene.world].mean(0).cpu().numpy()
        viewer.set_camera(wp.vec3(*(center+np.array([.8,-.8,.6]))));viewer.camera.look_at(center)
    begin=time.perf_counter();frames=0;failure=None;final_test='NOT_RUN';trace=[]
    trajectory=[solver.read(example.state_0).detach()]
    stream=wp.stream_from_torch(torch.cuda.current_stream()) if args.device.startswith('cuda') else None
    class Scope:
        def __enter__(self):
            self.scope=wp.ScopedStream(stream) if stream is not None else wp.ScopedDevice('cpu')
            return self.scope.__enter__()
        def __exit__(self,*values):return self.scope.__exit__(*values)
    def record(state):
        d=solver.last_diagnostic
        trace.append({'step':solver.steps,'contacts':d['contacts'],'minimum_separation':d['minimum_contact_normal_physical_separation'],'min_alpha':float(d['min_alpha'].min()),'graph_generation':scene.topology_generation})
        trajectory.append(solver.read(state).detach())
        if solver.steps%50==0:print(f'substep={solver.steps} contacts={d["contacts"]} alpha={float(d["min_alpha"].min()):.4g}',flush=True)
    solver.on_substep=record
    try:
        while viewer.is_running() and frames<args.frames:
            with Scope():
                if viewer.should_step():example.step();frames+=1
                example.render()
            if frames and frames%30==0:print(f'progress: frames={frames} steps={solver.steps} t={example.sim_time:.3f}',flush=True)
        if args.test_final:
            try:example.test_final();final_test='PASS'
            except Exception as error:final_test='FAIL: '+str(error)
        if args.screenshot and not args.headless:
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True,exist_ok=True)
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(args.screenshot)
    except Exception:
        failure=traceback.format_exc();print(failure,flush=True)
    finally:viewer.close()
    pose=wp.to_torch(example.state_0.body_q).detach().cpu().tolist()
    report={'status':'FAIL' if failure or final_test.startswith('FAIL') else 'INFERENCE_EXECUTED','failure':failure,'demo':args.demo,'frames':frames,'physical_substeps':solver.steps,'physical_seconds':solver.steps*example.sim_dt,'seconds':time.perf_counter()-begin,'dt':example.sim_dt,'checkpoint_sha256':hashlib.sha256(raw).hexdigest(),'full_vbd_calls':0,'trajectory_files_read':0,'training_status':'UNTRAINED_CONTACT_TRANSFER','learning_enabled':not args.disable_learning,'test_final':final_test,'collision_queries':solver.oracle.queries,'final_body_origin_poses':pose,'trace':trace,'solver_configuration':metadata['solver_configuration'],'arguments':authored_args}
    report['implementation_sha256']=implementation
    contact_key='rigid_contact_train_default' if args.demo=='vbd.example_vbd_rigid_rigid_contact' else args.demo.split('.')[-1].removeprefix('example_')+'_train_default'
    if stored.get('sampled_updates',{}).get(contact_key,0)>0:report['training_status']='SAMPLED_IN_SHARED_HEAD_CALIBRATION'
    report['applied_parameters']={'contact_stiffness':args.contact_stiffness,'contact_damping':args.contact_damping,'friction':args.friction}
    trajectory_path=args.report.with_suffix('.npz');trajectory_path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(trajectory_path,**{key:np.stack([getattr(s,key)[0,:scene.world].cpu().numpy() for s in trajectory]) for key in ('x','q','v','omega')})
    report['trajectory']={'path':str(trajectory_path),'sha256':hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),'physical_substeps':len(trajectory)-1}
    report['implementation_unchanged_during_run']=all(hashlib.sha256((ROOT/'src/wp_vbd'/name).read_bytes()).hexdigest()==digest for name,digest in implementation.items())
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k in ('status','failure','frames','physical_substeps','physical_seconds','seconds','test_final')},indent=2),flush=True)
    if report['status']=='FAIL':raise SystemExit(1)


if __name__=='__main__':main()
