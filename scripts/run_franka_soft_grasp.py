"""Run the validated low-resolution Franka/soft-block model preview.

Defaults to an independently initialized, contact-free near-grasp pose.
Use --start-time 0 --frames 240 for the complete authored command timeline.
Computation is currently slower than real time.
No reference trajectory or native dynamics solver is used by this entry.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    parser.add_argument('--gpu-memory-gib',type=float,default=12.)
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/plateau_review_v303/shared_trial_2478.pt')
    parser.add_argument('--window',action='store_true')
    parser.add_argument('--frames',type=int,default=117)
    parser.add_argument('--substeps',type=int,default=10)
    parser.add_argument('--start-time',type=float,default=1.85)
    parser.add_argument('--neural-graphs',action='store_true')
    parser.add_argument('--neural-precision',choices=('fp32','bf16'),default='fp32')
    parser.add_argument('--compact-diagnostics',action='store_true')
    parser.add_argument('--material-mu',type=float,default=5000.)
    parser.add_argument('--material-lambda',type=float,default=5000.)
    parser.add_argument('--friction',type=float,default=1.5)
    parser.add_argument('--print-command',action='store_true',help='Print the exact argv without running inference')
    args=parser.parse_args()
    command=[sys.executable,'-u','-B',str(ROOT/'scripts/run_franka_vbd_shared.py'),
             '--device',args.device,'--gpu-memory-gib',str(args.gpu_memory_gib),
             '--checkpoint',str(args.checkpoint),'--frames',str(args.frames),
             '--substeps',str(args.substeps),'--start-time',str(args.start_time),'--material-mu',str(args.material_mu),
             '--material-lambda',str(args.material_lambda),'--friction',str(args.friction),
             '--arm-drive-ke','20000','--grasp-x','0','--grasp-yaw','-0.7853981633974483',
             '--grip-closed','.024','--articulation-modes','--separate-coarse-step',
             '--material-motion-modes','--tangent-support-scale','8','--force-aware-anticipation',
             '--output',str(args.output)]
    if args.window:command.append('--window')
    if args.neural_graphs:command.append('--neural-graphs')
    if args.neural_precision!='fp32':command+=['--neural-precision',args.neural_precision]
    if args.compact_diagnostics:command.append('--compact-diagnostics')
    if args.print_command:print(json.dumps(command,ensure_ascii=False,indent=2));return
    subprocess.run(command,check=True)


if __name__=='__main__':main()
