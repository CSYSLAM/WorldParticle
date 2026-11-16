"""Claim one canonical repository CUDA slot before importing device libraries."""
import fcntl
import os
from pathlib import Path
import subprocess


def claim_cuda(device,repository):
    if not str(device).startswith('cuda'):return None
    # Detached worktrees must share the original repository lock and process
    # scope; their own ROOT/.local paths would create independent slots.
    common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],
                                       cwd=repository,text=True).strip()).resolve()
    root=common.parent
    lock=root/'.local/owned_gpu_compute.lock';lock.parent.mkdir(parents=True,exist_ok=True)
    ownership=lock.open('a+')
    try:
        try:fcntl.flock(ownership,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Another WorldParticle CUDA job owns the device; use CPU or wait')
        output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        for line in output.splitlines():
            if not line.strip().isdigit():continue
            pid=int(line.strip())
            if pid==os.getpid():continue
            try:cwd=Path(f'/proc/{pid}/cwd').resolve(strict=True)
            except (FileNotFoundError,ProcessLookupError):continue
            if cwd.is_relative_to(root):
                raise RuntimeError(f'WorldParticle CUDA process {pid} is running; use CPU or wait')
        return ownership
    except BaseException:
        ownership.close();raise
