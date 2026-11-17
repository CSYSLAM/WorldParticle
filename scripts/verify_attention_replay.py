"""Check pure attention replay against actual trained eager block weights."""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/shared_normal_648_v84/shared_strict.pt')
    parser.add_argument('--neural-precision',choices=('fp32','bf16'),default='fp32')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    import torch
    from wp_vbd.gpu_budget import GpuBudget
    from wp_vbd.heterogeneous_backbone import AttentionBlock
    from wp_vbd.neural_cuda_graph import enable_attention_replay,attention_replay_report
    from wp_vbd.neural_precision import NeuralPrecision
    torch.set_num_threads(1);torch.manual_seed(89);torch.use_deterministic_algorithms(True)
    budget=GpuBudget(args.device)
    raw=args.checkpoint.read_bytes();checkpoint_sha=hashlib.sha256(raw).hexdigest()
    stored=torch.load(io.BytesIO(raw),weights_only=True,map_location='cpu');del raw
    config=stored['model_config'] if 'model_config' in stored else stored['config']['model']
    width=config['protected_width'];heads=config['hierarchy_heads']
    block=AttentionBlock(width,heads).to(args.device).eval()
    prefix='protected_response.hierarchy.encoder.0.'
    block.load_state_dict({name.removeprefix(prefix):value for name,value in stored['model'].items() if name.startswith(prefix)});del stored
    schema=list(block.state_dict());enable_attention_replay(block,maximum_entries=2)
    precision=NeuralPrecision(block,args.neural_precision,args.device)
    inference_precision=True
    maximum_difference=0.;snapshots=[];comparisons=0
    def compare(query,context,sizes):
        nonlocal maximum_difference,comparisons
        with torch.autocast(query.device.type,dtype=torch.bfloat16,
                            enabled=inference_precision and args.neural_precision=='bf16'):
            reference=block._eager_attention(query,context,sizes)
        result=block(query,context,sizes)
        precision.verify_physical_boundary()
        difference=float((reference-result).abs().max());maximum_difference=max(maximum_difference,difference)
        torch.testing.assert_close(result,reference,rtol=2e-6,atol=2e-6)
        comparisons+=1
        return result
    with torch.no_grad():
        for batch,queries,keys in ((2,6,4),(1,3,2),(1,9,7)):
            for repeat in range(6):
                query=torch.randn(batch,queries,width,device=args.device)
                context=torch.randn(batch,keys,width,device=args.device)
                sizes=torch.randint(1,5,(batch,keys,1),device=args.device).float()
                value=compare(query,context,sizes)
                if repeat==3:snapshots.append((value,value.clone()))
        # In-place loading updates captured weight addresses; old values must
        # never be baked into replay. Head-layout mutations invalidate keys.
        block.output.bias.add_(.125);compare(query,context,sizes)
        block.heads=heads//2
        for repeat in range(3):compare(query,context,sizes)
        for value,saved in snapshots:assert torch.equal(value,saved),'Replay mutated an earlier output'
    replay=attention_replay_report(block)
    if args.device.startswith('cuda'):
        assert replay['captures']>=4 and replay['hits']>0 and replay['evictions']>=2,replay
    precision.close();inference_precision=False
    # Reuse the same cache with FP32 after BF16. Policy changes must neither
    # replay stale casts nor change training's eager full-precision path.
    if args.neural_precision=='bf16':
        with torch.no_grad():
            for _ in range(4):compare(query,context,sizes)
    before=attention_replay_report(block)['captures']
    query=query.detach().requires_grad_();context=context.detach().requires_grad_()
    reference=block._eager_attention(query,context,sizes);result=block(query,context,sizes)
    parameters=tuple(block.parameters());inputs=(query,context,*parameters)
    left=torch.autograd.grad(reference.square().mean(),inputs)
    right=torch.autograd.grad(result.square().mean(),inputs)
    gradient_difference=max(float((a-b).abs().max()) for a,b in zip(left,right))
    assert all(torch.equal(a,b) for a,b in zip(left,right))
    block.train()
    with torch.no_grad():compare(query,context,sizes)
    assert attention_replay_report(block)['captures']==before
    assert list(block.state_dict())==schema
    budget.observe(force=True)
    report={'status':'CUDA_ATTENTION_REPLAY_CHECKS_PASS' if args.device.startswith('cuda') else 'CPU_ATTENTION_REPLAY_BYPASS_AND_GRADIENT_CHECKS_PASS',
            'device':args.device,'checkpoint_sha256':checkpoint_sha,'actual_trained_block':prefix.rstrip('.'),
            'latent_width':width,'comparisons':comparisons,'maximum_output_difference':maximum_difference,
            'autograd_bypass_maximum_gradient_difference':gradient_difference,
            'earlier_outputs_owned_and_unchanged':True,'state_dict_schema_unchanged':True,
            'replay':attention_replay_report(block),'gpu_resource_budget':budget.report(),
            'neural_precision':precision.report(),
            'new_optimizer_updates':0,'physics_steps':0,'full_vbd_prediction_calls':0,
            'implementation_sha256':{name:hashlib.sha256((ROOT/'src/wp_vbd'/name).read_bytes()).hexdigest() for name in ('neural_cuda_graph.py','heterogeneous_backbone.py')},
            'scope':'Actual selected648 encoder block, changing inputs/shapes/multiplicities, current weight/head layout, bounded cache and autograd/training bypass. CPU does not exercise CUDA capture. No complete physics, native task, speedup or realtime claim.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
