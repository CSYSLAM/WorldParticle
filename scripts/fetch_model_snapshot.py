"""Download or join the one published checkpoint, verifying every part and SHA."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from urllib.request import urlopen

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot',type=Path,default=ROOT/'model_releases/shared_vbd_step156')
    p.add_argument('--parts-dir',type=Path,help='Join already downloaded parts instead of fetching HTTP')
    args=p.parse_args();folder=args.snapshot.expanduser().resolve()
    manifest=json.loads((folder/'manifest.json').read_text());spec=manifest['checkpoint']
    if Path(spec['file']).name!=spec['file']:raise ValueError('Checkpoint must be a local filename')
    target=folder/spec['file'];parts=manifest['download']['parts']
    if sum(row['bytes'] for row in parts)!=spec['bytes']:raise ValueError('Inconsistent part sizes')
    if target.exists():
        with target.open('rb') as stream:existing=hashlib.file_digest(stream,'sha256').hexdigest()
        if existing!=spec['sha256'] or target.stat().st_size!=spec['bytes']:
            raise ValueError('Existing checkpoint differs; preserve or remove it explicitly before downloading')
        print(json.dumps(dict(status='ALREADY_VERIFIED',file=str(target),sha256=existing)));return
    descriptor,temporary_name=tempfile.mkstemp(prefix=target.name+'.',suffix='.tmp',dir=folder)
    temporary=Path(temporary_name);combined=hashlib.sha256()
    try:
        with os.fdopen(descriptor,'wb') as output:
            for index,row in enumerate(parts):
                name=row['file']
                if Path(name).name!=name or not 0<row['bytes']<=32*1024**2:raise ValueError('Invalid part descriptor')
                for attempt in range(3):
                    try:
                        if args.parts_dir:data=(args.parts_dir/name).read_bytes()
                        else:
                            with urlopen(manifest['download']['base_url']+'/'+name,timeout=45) as response:
                                data=response.read(row['bytes']+1)
                        if len(data)!=row['bytes'] or hashlib.sha256(data).hexdigest()!=row['sha256']:
                            raise ValueError('Checkpoint part integrity mismatch: '+name)
                        break
                    except (OSError,ValueError):
                        if attempt==2:raise
                        time.sleep(2)
                output.write(data);combined.update(data)
                print(json.dumps(dict(part=index+1,total=len(parts),bytes=row['bytes'],verified=True)),flush=True)
        if temporary.stat().st_size!=spec['bytes'] or combined.hexdigest()!=spec['sha256']:
            raise ValueError('Combined checkpoint integrity mismatch')
        temporary.replace(target)
    except BaseException:
        temporary.unlink(missing_ok=True);raise
    print(json.dumps(dict(status='COMPLETE',file=str(target),sha256=combined.hexdigest())),flush=True)


if __name__=='__main__':main()
