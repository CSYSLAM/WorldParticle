"""Budget a proposed sparse FFN extension; this does not enable a new model.

The per-token path proxy is not the number of unique weights touched by a
whole scene. New adapters/readouts and retired heads are deliberately excluded.
"""
import argparse
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=ROOT/'model_releases/shared_vbd_v32_a1')
    parser.add_argument('--blocks',type=int,default=4)
    parser.add_argument('--experts',type=int,nargs='+',default=[4,8])
    parser.add_argument('--top-k',type=int,nargs='+',default=[1,2])
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    manifest=json.loads((args.snapshot/'manifest.json').read_text())
    config=json.loads((args.snapshot/'model_config.json').read_text())
    depth=config['hierarchy_encoder_layers']+config['hierarchy_decoder_layers']
    if not 0<args.blocks<=depth:parser.error('Selected FFN count must fit the existing attention depth')
    if any(e<2 for e in args.experts) or any(k<1 for k in args.top_k):parser.error('Positive valid expert counts required')
    width=config['protected_width'];hidden=2*width
    base=manifest['active_neural_parameters']
    one_ffn=2*width*hidden+hidden+width
    rows=[]
    for experts in args.experts:
        router=args.blocks*(width+1)*experts
        total=base+args.blocks*(experts-1)*one_ffn+router
        for top_k in args.top_k:
            if top_k>experts:parser.error('top-k cannot exceed the number of experts')
            dense_ffn=2*width*(top_k*hidden)+top_k*hidden+width
            rows.append(dict(experts=experts,top_k=top_k,replaced_ffns=args.blocks,
                total_parameter_proxy=total,
                per_token_path_parameter_proxy=base+args.blocks*(top_k-1)*one_ffn+router,
                dense_matched_expert_matmul_proxy=base+args.blocks*(dense_ffn-one_ffn),
                fp32_parameter_bytes=4*total,
                fp32_adamw_weight_gradient_two_moments_bytes=16*total))
    result=dict(status='DESIGN_ESTIMATE_NOT_IMPLEMENTATION',base_active_parameters=base,
        width=width,expert_hidden=hidden,parameters_per_ffn=one_ffn,candidates=rows,
        exclusions=['New feature adapters/readouts are not added; retired response heads are not subtracted',
            'Path proxy is not whole-frame activated parameters, FLOPs or measured latency',
            'All experts remain GPU-resident; sparse activation does not remove their storage',
            'Activation, physical state, cache, allocator and routing workspace are not included',
            'Matched dense proxy ignores dispatch/router costs and does not assert equal wall time'])
    payload=json.dumps(result,indent=2)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(payload)
    else:print(payload,end='')


if __name__=='__main__':main()
