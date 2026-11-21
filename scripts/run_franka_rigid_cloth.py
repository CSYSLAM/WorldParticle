"""Infer a complete dynamic Franka pick/place onto a deformable cloth.

The payload is a free rigid body. IK supplies arm drive targets only.
The shared checkpoint and the physical response run on the chosen device.
Render the resulting recording with render_franka_model_video.py.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execution-policy',choices=('reference','assembly','fused','fused_search','fused_fast','fused_device','fused_graph'),default='reference')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'model_releases/shared_vbd_v24_cloth40/model.pt')
    p.add_argument('--device',choices=('cuda:0','cpu'),default='cuda:0')
    p.add_argument('--gpu-memory-gib',type=float,default=12.)
    p.add_argument('--frames',type=int,default=384)
    p.add_argument('--substeps',type=int,default=3)
    p.add_argument('--rigid-tangent-support-scale',type=float,default=64.)
    p.add_argument('--window',action='store_true')
    p.add_argument('--print-command',action='store_true')
    a=p.parse_args()
    command=[sys.executable,'-u',str(ROOT/'scripts/run_franka_vbd_shared.py'),
             '--output',str(a.output),'--checkpoint',str(a.checkpoint),'--device',a.device,
             '--gpu-memory-gib',str(a.gpu_memory_gib),'--frames',str(a.frames),'--substeps',str(a.substeps),
             '--soft-asset','rigid_cloth','--arm-drive-ke','20000','--grip-closed','.0225',
             '--articulation-modes','--separate-coarse-step','--tangent-support-scale','8',
             '--force-aware-anticipation','--compact-diagnostics','--motion-budget','.012',
             '--rigid-tangent-support-scale',str(a.rigid_tangent_support_scale),
             '--material-motion-modes','--surface-tangent-history','--contact-aware-warm-start']
    command += ['--execution-policy',a.execution_policy]
    if a.window:command.append('--window')
    if a.print_command:print(json.dumps(command,ensure_ascii=False,indent=2));return
    subprocess.run(command,check=True)


if __name__=='__main__':main()
