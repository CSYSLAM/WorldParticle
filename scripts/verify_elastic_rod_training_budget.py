"""Measure complete mixed two-hop training graphs with resident Adam moments.

No optimizer step is executed. CUDA is claimed before device libraries load;
CPU results cannot be used as a CUDA or course-admission certificate.
"""
import argparse
import functools
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--indices',type=int,nargs='+',default=[0,1,600,1800,2800])
    parser.add_argument('--gpu-memory-gib',type=float,default=12.)
    parser.add_argument('--duty-cycle',type=float,default=.25)
    parser.add_argument('--host-encoder-inputs',action='store_true')
    parser.add_argument('--host-step-saved-tensors',action='store_true')
    parser.add_argument('--checkpoint-relation-ffn-rows',type=int,default=0)
    parser.add_argument('--checkpoint-attention-blocks',action='store_true')
    parser.add_argument('--contact-activation-ray-step',action='store_true')
    parser.add_argument('--contact-motion-ray-step',action='store_true')
    parser.add_argument('--contact-structural-support',action='store_true')
    parser.add_argument('--learned-contact-repair',action='store_true')
    parser.add_argument('--self-broad-phase',choices=('chunked','kdtree'))
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    if not 0<args.duty_cycle<=1 or args.checkpoint_relation_ffn_rows<0:
        parser.error('Finite positive duty cycle and nonnegative checkpoint bound required')
    if len(set(args.indices))!=len(args.indices):parser.error('Distinct physical windows required')
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(args.device,ROOT)
    import torch
    import warp as wp
    import newton
    from wp_vbd.elastic_rod_data import load_reference,step
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.mixed_experiment import select,loss
    from wp_vbd.training_window import transfer_window
    from wp_vbd.gpu_budget import GpuBudget
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    budget=GpuBudget(args.device,args.gpu_memory_gib)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(args.device)
    if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    calls={'construct':0,'step':0}
    original_constructor=newton.solvers.SolverVBD.__init__
    @functools.wraps(original_constructor)
    def forbidden(*unused,**kwargs):calls['construct']+=1;raise AssertionError('No VBD construction in training budget probe')
    def forbidden_step(*unused,**kwargs):calls['step']+=1;raise AssertionError('No VBD step in training budget probe')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden_step
    scene,reference=load_reference(args.data,wp,newton,args.device,state_device='cpu')
    if args.self_broad_phase is not None:
        from wp_vbd.elastic_rod_runtime import ElasticRodCollisionOracle
        scene.metadata['self_contact_broad_phase']=args.self_broad_phase
        scene.geometry_oracle=ElasticRodCollisionOracle(scene.geometry_oracle.source_model,scene,wp,newton)
    if not scene.metadata['eligible_normal_course']:raise ValueError('Complete source-scope qualified reference required')
    if any(not 0<=i<len(reference['x'])-2 for i in args.indices):raise ValueError('Complete two-hop windows required')
    stored=torch.load(args.checkpoint,weights_only=True,map_location='cpu')
    configuration={**stored['config']['model'],'elastic_rod_self_contact':True,'elastic_rod_refresh_self_geometry':True,
                   'elastic_rod_differentiated_residual_backward':True}
    for flag in ('checkpoint_attention_blocks','contact_activation_ray_step','contact_motion_ray_step','contact_structural_support','learned_contact_repair'):
        if getattr(args,flag):configuration[flag]=True
    if args.checkpoint_relation_ffn_rows:configuration['checkpoint_relation_ffn_rows']=args.checkpoint_relation_ffn_rows
    configuration['contact_row_checkpointing']=True
    network=MixedSolver(configuration).to(args.device).train();network.load_state_dict(stored['model'])
    if args.host_encoder_inputs and not configuration.get('neural_encoder_checkpointing'):
        raise ValueError('Host encoder inputs require encoder recomputation')
    network.protected_response.checkpoint_inputs_on_cpu=args.host_encoder_inputs
    network.step_saved_tensors_on_cpu=args.host_step_saved_tensors
    for name,p in network.named_parameters():p.requires_grad_(name.startswith('protected_response.'))
    trainable=[p for p in network.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable)==46882811
    optimizer=torch.optim.AdamW(trainable,lr=1e-7,eps=1e-12,weight_decay=0.,foreach=False)
    for p in trainable:
        optimizer.state[p]={'step':torch.tensor(0.),'exp_avg':torch.zeros_like(p),'exp_avg_sq':torch.zeros_like(p)}
    versions={k:p._version for k,p in network.named_parameters()}
    cases=[];failure=None
    try:
        for index in args.indices:
            if args.device.startswith('cuda'):torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            began=time.perf_counter();network.zero_grad(set_to_none=True)
            states=transfer_window(reference,index,2,args.device);current=select(states,slice(0,1))
            objective=current.x.new_zeros(());rows=[];queries=scene.geometry_oracle.queries
            for offset in range(2):
                current,diagnostic=step(network,scene,current,states,offset,wp)
                if not bool(diagnostic['accepted'].all()):raise RuntimeError(f'Prediction rejected at {index+offset}')
                objective=objective+(loss(scene,current,select(states,slice(offset+1,offset+2)))+diagnostic.get('contact_repair_supervision',0.))/2
                rows.append({k:diagnostic[k] for k in ('active_self_rows_by_kind','active_soft_cloth_self_rows',
                    'active_dynamic_particle_body_rows','rod_rows_with_nonzero_wrench','finite_joint_rows_with_nonzero_wrench')})
                if 'contact_repair_calls' in diagnostic:
                    rows[-1].update(contact_repair_calls=diagnostic['contact_repair_calls'],
                        contact_repair_supervision=float(diagnostic['contact_repair_supervision'].detach()))
            assert scene.geometry_oracle.queries==queries+2
            history_versions={k:v._version for k,v in current.history.items()}
            if not bool(torch.isfinite(objective)):raise RuntimeError('Nonfinite objective')
            objective.backward()
            assert scene.geometry_oracle.queries==queries+2
            assert history_versions=={k:v._version for k,v in current.history.items()}
            gradients=[p.grad for p in trainable if p.grad is not None]
            if not gradients or not all(bool(torch.isfinite(g).all()) for g in gradients):raise RuntimeError('Missing or nonfinite full gradients')
            gradient_norm=float(torch.stack([g.double().square().sum() for g in gradients]).sum().sqrt())
            if gradient_norm<=0:raise RuntimeError('Zero full gradient')
            modules={prefix:sum(float(p.grad.double().norm()) for name,p in network.named_parameters()
                if name.startswith('protected_response.'+prefix) and p.grad is not None)
                for prefix in ('hierarchy.encoder','hierarchy.decoder','hierarchy.material')}
            if not all(v>0 for v in modules.values()):raise RuntimeError('Missing shared encoder/decoder/material gradients')
            row={'index':index,'two_hop_loss':float(objective.detach()),'gradient_norm':gradient_norm,
                'parameter_elements_with_grad':sum(p.numel() for p in trainable if p.grad is not None),
                'backbone_gradient_norms':modules,'actual_evaluated_rows':rows,
                'own_geometry_queries':2,'backward_geometry_queries':0,
                'accepted_history_unchanged_by_backward':True,'optimizer_step_executed':False}
            if args.device.startswith('cuda'):
                torch.cuda.synchronize();row.update(observed_process_gpu_mib=budget.observe(force=True),
                    peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved())
            row['seconds']=time.perf_counter()-began;cases.append(row);print(json.dumps(row),flush=True)
            del current,objective,gradients,diagnostic,states
            network.zero_grad(set_to_none=True);gc.collect()
            if args.device.startswith('cuda'):
                torch.cuda.empty_cache();remaining=(time.perf_counter()-began)*(1/args.duty_cycle-1)
                while remaining>0:
                    duration=min(5.,remaining);time.sleep(duration);remaining-=duration
    except Exception:
        failure=traceback.format_exc()
    unchanged=versions=={k:p._version for k,p in network.named_parameters()}
    assert unchanged and all(torch.equal(v,network.state_dict()[k].cpu()) for k,v in stored['model'].items())
    assert all(float(optimizer.state[p]['step'])==0 and not bool(optimizer.state[p]['exp_avg'].any())
               and not bool(optimizer.state[p]['exp_avg_sq'].any()) for p in trainable)
    assert calls=={'construct':0,'step':0}
    result={'status':'PASS' if failure is None else 'FAIL','failure':failure,'device':args.device,'cases':cases,
        'effective_trainable_parameters':sum(p.numel() for p in trainable),
        'effective_model_config':configuration,'model_configuration_overrides':{k:v for k,v in configuration.items()
            if stored['config']['model'].get(k)!=v},'adam_moments_resident_during_forward_backward':True,
        'host_reference_trajectories':True,'reference_state_device':str(reference['x'].device),
        'host_encoder_checkpoint_inputs':args.host_encoder_inputs,'host_step_saved_tensors':args.host_step_saved_tensors,
        'all_model_weights_and_adam_unchanged':unchanged,'gpu_resource_budget':budget.report(),'duty_cycle':args.duty_cycle,
        'source_contact_definition':scene.metadata['source_self_contact_definition'],
        'device_acceleration_differences':scene.current_device_acceleration_differences,
        'self_contact_broad_phase':scene.geometry_oracle.self_query.broad_phase,
        'self_contact_motion_budget':scene.metadata['self_contact_motion_budget'],
        'data_sha256':digest(args.data),'checkpoint_sha256':digest(args.checkpoint),'script_sha256':digest(Path(__file__)),
        'implementation_sha256':{p.name:digest(p) for p in (ROOT/'src/wp_vbd').glob('*.py')},
        'new_optimizer_updates':0,'new_training_admissions':0,'full_vbd_prediction_calls':calls,
        'scope':'Independent complete mixed two-hop training graphs with resident zero-update Adam. CPU is not CUDA certification. Source scope qualification does not certify parallel-witness force equivalence, teacher convergence, native quality or realtime. Course admission and all inherited development limits remain separate.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    if failure:raise SystemExit(failure)


if __name__=='__main__':main()
