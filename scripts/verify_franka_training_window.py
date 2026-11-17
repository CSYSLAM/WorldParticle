"""Check causal two-hop learning on qualified grasp data without optimizer steps."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import activate_newton_source, source_root
NEWTON_ROOT=source_root()
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/plateau_review_v303/shared_trial_2478.pt')
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--index',type=int,default=520)
    parser.add_argument('--backward',action='store_true')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    activate_newton_source()
    if args.report.exists():raise FileExistsError(args.report)
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(args.device,ROOT)
    import torch
    import warp as wp
    import newton
    sys.path.append(str(NEWTON_ROOT/'.venv/lib/python3.12/site-packages'))
    from wp_vbd.franka_reference_data import load_reference
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.mixed_experiment import select,loss,external_inputs
    from wp_vbd.general_joint_data import set_controls
    from wp_vbd.mixed_contact_data import query
    from newton._src.solvers.vbd import solver_vbd
    native_projection=solver_vbd.project_joint_mimics
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(args.device)
    if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    def forbidden(*unused,**kwargs):raise AssertionError('No complete reference solver in the learning window')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    scene,states=load_reference(args.dataset,wp,newton,args.device)
    if not 0<=args.index<len(states['x'])-2:raise ValueError('A complete two-hop window is required')
    stored=torch.load(args.checkpoint,map_location='cpu',weights_only=True)
    config=stored['model_config'] if 'model_config' in stored else stored['config']['model']
    model=MixedSolver(config).to(args.device).eval();model.load_state_dict(stored['model'])
    calls=0
    def count(*unused):
        nonlocal calls
        calls+=1
    model.protected_response.hierarchy.register_forward_hook(count)
    current=select(states,slice(args.index,args.index+1));rows=[];total=0.;start=time.perf_counter()
    for hop in range(2):
        index=args.index+hop
        set_controls(scene,states,index);query(scene,current,wp)
        predicted,diagnostic=model(scene,current,scene.metadata['dt'],**external_inputs(states,slice(index,index+1)))
        if not bool(diagnostic['accepted'].all()):raise AssertionError('Learning-window step rejected')
        target=select(states,slice(index+1,index+2));value=loss(scene,predicted,target);total=total+value
        n=scene.metadata['particle_count'];b=scene.metadata['body_count']
        rows.append({'hop':hop,'loss':float(value.detach()),
                     'material_position_rms_m':float((predicted.x[:,:n]-target.x[:,:n]).square().mean().sqrt().detach()),
                     'robot_com_position_rms_m':float((predicted.x[:,n:n+b]-target.x[:,n:n+b]).square().mean().sqrt().detach())})
        current=predicted
        print(json.dumps({'hop':hop,'seconds':time.perf_counter()-start,**rows[-1]}),flush=True)
    forward_calls=calls;gradient=None
    if args.backward:
        total.backward()
        values=[p.grad for p in model.parameters() if p.grad is not None]
        finite=all(bool(torch.isfinite(v).all()) for v in values)
        norm=float(torch.stack([v.double().square().sum() for v in values]).sum().sqrt())
        nonzero=sum(bool(v.abs().max()>0) for v in values)
        assert finite and norm>0 and nonzero>0
        gradient={'finite':finite,'norm':norm,'nonzero_parameter_tensors':nonzero}
    assert solver_vbd.project_joint_mimics is native_projection
    assert all(torch.equal(p.detach().cpu(),stored['model'][name]) for name,p in model.named_parameters())
    result={'status':'PASS','device':args.device,'index':args.index,'hops':rows,
            'total_two_hop_loss':float(total.detach()),'forward_encoder_calls':forward_calls,
            'backward_recomputation_encoder_calls':calls-forward_calls,'gradient':gradient,
            'seconds':time.perf_counter()-start,'new_optimizer_updates':0,'weights_changed':False,
            'full_vbd_calls':0,'featherstone_calls':0,'native_reference_adapter_not_installed':True,
            'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'scope':'Teacher-conditioned causal training/evaluation windows only; no autonomous model task claim.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(args.report.read_text())


if __name__=='__main__':main()
