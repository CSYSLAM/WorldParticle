"""Repeat a robot scene with its recorded checkpoint and physical-policy recipe."""

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', choices=('franka', 'w1', 'cloth','piper','w1_dual','franka_rigid_cloth'), required=True)
    parser.add_argument('--recipe',type=Path,default=ROOT/'configs/robot_validation_cloth_ft.json')
    parser.add_argument('--device', choices=('cuda:0', 'cpu'), default='cuda:0')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--asset', type=Path, help='W1 URDF or asset directory')
    parser.add_argument('--frames', type=int, help='Shorten or extend the recording; omit for the full validated task')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--window',action='store_true',help='Display the live inference window')
    parser.add_argument('--dry-run', action='store_true', help='Print commands without loading the model or running physics')
    args = parser.parse_args()
    if args.asset and args.scene not in ('w1','w1_dual','piper'):
        parser.error('--asset applies to W1 and Piper scenes only')
    if args.frames is not None and args.frames < 1:
        parser.error('--frames must be positive')
    recipe = json.loads(args.recipe.expanduser().resolve().read_text())
    if args.scene not in recipe['scenes']:
        parser.error('The selected recipe has no scene named '+args.scene)
    scene = recipe['scenes'][args.scene]
    checkpoint = ROOT / recipe['checkpoint']
    newton_root = Path(os.environ.get('WP_VBD_NEWTON_ROOT',str(ROOT / recipe['newton_root']))).expanduser().resolve()
    output = (args.output or ROOT / 'outputs/robot_validation' /
              f'{args.scene}_{datetime.datetime.now():%Y%m%d_%H%M%S_%f}').expanduser().resolve()
    command = [sys.executable, '-u', scene['runner'], '--checkpoint', str(checkpoint),
               '--output', str(output), '--device', args.device, *scene['arguments']]
    if args.frames is not None:
        command[command.index('--frames') + 1] = str(args.frames)
    if args.asset:
        command += ['--asset', str(args.asset.expanduser().resolve())]
    if args.window:command.append('--window')
    render_command = [sys.executable, scene['renderer'], '--recording', str(output),
                      '--output', str(output / 'model.mp4')]
    env = dict(os.environ, WP_VBD_NEWTON_ROOT=str(newton_root))
    if args.device == 'cpu':
        env['CUDA_VISIBLE_DEVICES'] = ''
    print('Output:', output, flush=True)
    print('Recipe:',args.recipe.expanduser().resolve(),flush=True)
    print('WP_VBD_NEWTON_ROOT=' + shlex.quote(str(newton_root)), flush=True)
    print(shlex.join(command), flush=True)
    if args.render:
        print(shlex.join(render_command), flush=True)
    if args.dry_run:
        return
    if not checkpoint.is_file():
        raise RuntimeError('Restore the published checkpoint first: python scripts/fetch_model_snapshot.py')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != recipe['checkpoint_sha256']:
        raise ValueError('Checkpoint bytes do not match the selected validation recipe')
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=newton_root, text=True).strip()
    if sha != recipe['newton_sha']:
        raise ValueError('The validation recipe requires the pinned Newton revision')
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if args.render:
        subprocess.run(render_command, cwd=ROOT, env=dict(env, CUDA_VISIBLE_DEVICES=''), check=True)


if __name__ == '__main__':
    main()
