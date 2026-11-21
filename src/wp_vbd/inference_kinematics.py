"""Bounded, one-physical-step memoization of pure pose kinematics.

No residual, force, material response, contact dual or acceptance decision is
cached. Tensor identity AND mutation version bind every geometry result. The
scope owns strong input references and expires before the next native query.
Gradient-enabled and unversioned inference tensors use the reference path.
"""
from collections import Counter,OrderedDict
from contextvars import ContextVar
import torch

from .static_topology import tensor_versions

_current=ContextVar('wp_inference_kinematics',default=None)


def _bytes(value,seen=None):
    if seen is None:seen=set()
    if isinstance(value,torch.Tensor):
        # A small view retains its entire backing allocation. Count strong
        # input references as well as outputs; aliasing within an entry does
        # not add storage, while shared entries are conservatively overcounted.
        storage=value.untyped_storage();key=(value.device,storage._cdata)
        if key in seen:return 0
        seen.add(key)
        return storage.nbytes()
    if isinstance(value,(list,tuple)):return sum(_bytes(v,seen) for v in value)
    if isinstance(value,dict):return sum(_bytes(v,seen) for v in value.values())
    if hasattr(value,'__dict__'):return sum(_bytes(v,seen) for v in vars(value).values())
    return 0


class StepKinematics:
    def __init__(self,*,maximum_entries=256,maximum_bytes=32*1024**2):
        if maximum_entries<1 or maximum_bytes<1:raise ValueError('Positive cache limits required')
        self.entries=OrderedDict();self.hits=Counter();self.misses=Counter()
        self.maximum_entries=maximum_entries;self.maximum_bytes=maximum_bytes
        self.bytes=0;self.peak_bytes=0;self.evictions=0

    def __enter__(self):
        self.token=_current.set(self)
        return self

    def __exit__(self,*unused):
        _current.reset(self.token)
        self.entries.clear();self.bytes=0

    def evaluate(self,name,owner,sources,options,compute):
        versions=tensor_versions(sources)
        if versions is None:return compute()
        key=(name,id(owner),versions,options)
        if key in self.entries:
            self.hits[name]+=1
            self.entries.move_to_end(key)
            return self.entries[key][0]
        self.misses[name]+=1
        value=compute();size=_bytes((value,sources))
        if size>self.maximum_bytes:return value
        while self.entries and (len(self.entries)>=self.maximum_entries or self.bytes+size>self.maximum_bytes):
            _,entry=self.entries.popitem(last=False);self.bytes-=entry[3];self.evictions+=1
        # Strong owner/source references prevent Python identities being reused.
        self.entries[key]=(value,owner,sources,size);self.bytes+=size
        self.peak_bytes=max(self.bytes,self.peak_bytes)
        return value

    def report(self):
        return {'hits':dict(self.hits),'misses':dict(self.misses),'peak_bytes':self.peak_bytes,
                'evictions':self.evictions,'maximum_bytes':self.maximum_bytes}


def memoize(name,owner,sources,options,compute):
    scope=_current.get()
    if scope is None or torch.is_grad_enabled():return compute()
    return scope.evaluate(name,owner,tuple(s for s in sources if isinstance(s,torch.Tensor)),options,compute)


def spec_tensors(spec):
    return tuple(value for value in vars(spec).values() if isinstance(value,torch.Tensor))
