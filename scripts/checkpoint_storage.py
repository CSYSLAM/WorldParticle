"""Save complete checkpoints atomically while retaining the previous file."""
import errno
import os
from pathlib import Path
import shutil
import tempfile


def required_bytes(value):
    import torch
    storages={}
    def visit(item):
        if isinstance(item,torch.Tensor):
            storage=item.untyped_storage();storages[(str(item.device),storage.data_ptr())]=storage.nbytes()
        elif isinstance(item,dict):
            for child in item.values():visit(child)
        elif isinstance(item,(list,tuple)):
            for child in item:visit(child)
    visit(value)
    return int(sum(storages.values())*1.02)+64*1024**2


def atomic_save(value,target,*,reserve_bytes=2*1024**3):
    import torch
    target=Path(target);needed=required_bytes(value);free=shutil.disk_usage(target.parent).free
    if free<needed+reserve_bytes:
        raise OSError(errno.ENOSPC,f'Checkpoint needs {needed} bytes plus {reserve_bytes} reserve; available {free}',str(target))
    # Unique same-directory files cannot clobber an earlier interrupted save.
    with tempfile.NamedTemporaryFile(prefix=target.name+'.',suffix='.tmp',dir=target.parent,delete=False) as stream:
        temporary=Path(stream.name)
        torch.save(value,stream);stream.flush();os.fsync(stream.fileno())
    temporary.replace(target)


def verify_replayed_sample(actual,expected):
    import math
    for key in ('update','dataset','demo','index','gradient_normalization','actual_max_gradient_scalar_changed'):
        if actual.get(key)!=expected.get(key):raise ValueError('Recovery replay identity differs: '+key)
    errors={}
    for key in ('two_hop_loss_scaled','state_loss_scaled','repair_loss_scaled','gradient_norm_before_clip'):
        a,b=actual[key],expected[key]
        if not math.isclose(a,b,rel_tol=1e-6,abs_tol=1e-10):raise ValueError('Recovery replay value differs: '+key)
        errors[key]=abs(a-b)
    return {'update':actual['update'],'dataset':actual['dataset'],'index':actual['index'],
            'recorded_scalar_absolute_differences':errors,'recorded_scalars_bitexact':all(value==0 for value in errors.values()),
            'scope':'Recorded sample identity/scalars only. The lost original end-state weights cannot be certified bitexact.'}
