"""Verify independent decomposition blocks, exact zeros and their gradients."""
import argparse
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']=''


def main():
    import torch
    from wp_vbd.math3d import eigh_zero_safe
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);torch.manual_seed(812)
    raw=torch.randn(2,43,3,3,dtype=torch.float64)
    blocks=raw@raw.transpose(-1,-2)+torch.eye(3,dtype=torch.float64)
    blocks[:,::9]=0.
    eigen,frame=eigh_zero_safe(blocks,maximum_batch=11)
    expected,_=eigh_zero_safe(blocks,maximum_batch=1000)
    torch.testing.assert_close(eigen,expected)
    reconstructed=frame@torch.diag_embed(eigen)@frame.transpose(-1,-2)
    torch.testing.assert_close(reconstructed,blocks)
    torch.testing.assert_close(frame.transpose(-1,-2)@frame,torch.eye(3,dtype=torch.float64).expand_as(frame))
    assert not bool(eigen[:,::9].any())
    torch.testing.assert_close(frame[:,::9],torch.eye(3,dtype=torch.float64).expand_as(frame[:,::9]))
    leaf=blocks.detach().clone().requires_grad_(True)
    spectrum,_=eigh_zero_safe(leaf,maximum_batch=11)
    spectrum.square().sum().backward()
    torch.testing.assert_close(leaf.grad,2*blocks)
    empty=torch.empty(1,0,3,3,dtype=torch.float64)
    assert eigh_zero_safe(empty)[0].shape==(1,0,3)
    result={'status':'PASS','all_blocks_retained':True,'spectrum_and_reconstruction_match':True,
            'exact_zero_spectrum_and_canonical_frame':True,'finite_correct_gradient':True,'empty_batch':True}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
