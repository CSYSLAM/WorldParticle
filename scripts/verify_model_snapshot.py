"""Verify the published shared-model checkpoint and strict CPU model loading."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot',type=Path,default=ROOT/'model_releases/shared_vbd_step156')
    parser.add_argument('--report',type=Path)
    args=parser.parse_args();folder=args.snapshot.expanduser().resolve()
    manifest=json.loads((folder/'manifest.json').read_text())
    path=folder/manifest['checkpoint']['file']
    if not path.is_file():raise RuntimeError('Download the weights first: python scripts/fetch_model_snapshot.py')
    if path.stat().st_size<1024 and path.read_bytes().startswith(b'version https://git-lfs.github.com/spec/v1'):
        raise RuntimeError('Only a pointer is present. Obtain the complete checkpoint with scripts/fetch_model_snapshot.py')
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    if path.stat().st_size!=manifest['checkpoint']['bytes'] or digest!=manifest['checkpoint']['sha256']:
        raise ValueError('Checkpoint size or SHA-256 mismatch')
    import torch
    from wp_vbd.mixed_solver import MixedSolver
    torch.set_num_threads(1)
    stored=torch.load(path,map_location='cpu',weights_only=True)
    config=json.loads((folder/'model_config.json').read_text())
    if stored['model_config']!=config:raise ValueError('Model configuration mismatch')
    if stored['step']!=manifest['checkpoint']['step'] or stored['architecture']!=manifest['architecture']:
        raise ValueError('Model identity mismatch')
    network=MixedSolver(config).eval();network.load_state_dict(stored['model'],strict=True)
    count=sum(p.numel() for p in network.protected_response.parameters())
    if count!=manifest['active_neural_parameters']:raise ValueError('Active parameter count mismatch')
    if not all(bool(torch.isfinite(value).all()) for value in stored['model'].values()):
        raise ValueError('Nonfinite model weights')
    if not stored.get('optimizer',{}).get('state') or 'rng_state' not in stored:
        raise ValueError('Expected optimizer and RNG state for this full checkpoint')
    result=dict(status='PASS',checkpoint_sha256=digest,checkpoint_bytes=path.stat().st_size,
                checkpoint_step=stored['step'],active_neural_parameters=count,
                strict_model_load=True,all_model_weights_finite=True,optimizer_and_rng_present=True,
                optimizer_state_entries=len(stored['optimizer']['state']),
                scope='Artifact integrity and strict CPU loading; no new rollout, training or convergence certification.')
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
