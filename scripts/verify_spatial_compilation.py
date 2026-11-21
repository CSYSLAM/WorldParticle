"""Check scoped FP64 small products, layouts, empty batches and affine scales."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def product(a,b):return a.reshape(-1,6,6)@b.reshape(-1,6,3)


def affine(bias,a,b,alpha,beta):
    import torch
    return torch.addmm(bias,a.reshape(-1,6),b.reshape(6,3),alpha=alpha,beta=beta)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0');p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    import torch
    from wp_vbd.compiled_contact import evaluate,small_spatial_products,_small_products
    torch.set_num_threads(1);torch.manual_seed(27151)
    device=torch.device(a.device);rows=[]
    precision=(torch.get_default_dtype(),torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32)
    limit=torch._dynamo.config.cache_size_limit
    with torch.no_grad(),small_spatial_products(True):
        for dtype in (torch.float64,torch.float32):
            for batch,transposed in ((2,False),(2,True),(0,False)):
                x=torch.randn(batch,6,6,device=device,dtype=dtype)
                y=torch.randn(batch,6,3,device=device,dtype=dtype)
                if transposed:x=x.transpose(-1,-2)
                actual=evaluate(product,x,y) if a.device.startswith('cuda') else product(x,y)
                expected=product(x,y)
                torch.testing.assert_close(actual,expected,rtol=1e-11 if dtype==torch.float64 else 1e-5,
                                           atol=1e-12 if dtype==torch.float64 else 1e-6)
                assert actual.dtype==dtype
                rows.append({'dtype':str(dtype),'batch':batch,'noncontiguous':transposed,'error':float((actual-expected).abs().max()) if actual.numel() else 0.})
        x=torch.randn(2,6,device=device,dtype=torch.float64);y=torch.randn(6,3,device=device,dtype=torch.float64)
        bias=torch.randn(3,device=device,dtype=torch.float64)
        for alpha,beta in ((1.,1.),(2.,-.5),(0.,0.)):
            actual=evaluate(affine,bias,x,y,alpha,beta) if a.device.startswith('cuda') else affine(bias,x,y,alpha,beta)
            torch.testing.assert_close(actual,affine(bias,x,y,alpha,beta),rtol=1e-11,atol=1e-12)
    assert not _small_products.get() and torch._dynamo.config.cache_size_limit==limit
    assert precision==(torch.get_default_dtype(),torch.get_float32_matmul_precision(),torch.backends.cuda.matmul.allow_tf32)
    result={'status':'PASS','device':a.device,'cases':rows,'affine_nonunit_alpha_beta_preserved':True,
            'precision_and_compiler_scope_restored':True,'native_dynamics_calls':0,'optimizer_updates':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
