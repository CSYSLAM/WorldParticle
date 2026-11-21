"""Bounded versioned index maps for the protected response; never cache physics."""
from collections import OrderedDict
import sys

import numpy as np
import torch

from .static_topology import tensor_versions


def _bytes(value,seen=None):
    if seen is None:seen=set()
    if isinstance(value,torch.Tensor):
        storage=value.untyped_storage();key=('tensor',value.device,storage._cdata)
        if key in seen:return 0
        seen.add(key);return storage.nbytes()
    if id(value) in seen:return 0
    seen.add(id(value));size=sys.getsizeof(value)
    if isinstance(value,np.ndarray):
        base=value
        while isinstance(base.base,np.ndarray):base=base.base
        return size+base.nbytes
    if isinstance(value,dict):return size+sum(_bytes(k,seen)+_bytes(v,seen) for k,v in value.items())
    if isinstance(value,(tuple,list,set,frozenset)):return size+sum(_bytes(v,seen) for v in value)
    if hasattr(value,'__dict__'):return size+_bytes(vars(value),seen)
    return size


class IndexWorkspace:
    def __init__(self,*,maximum_entries=128,maximum_bytes=16*1024**2):
        if maximum_entries<1 or maximum_bytes<1:raise ValueError('Positive index cache limits required')
        self.maximum_entries=maximum_entries;self.maximum_bytes=maximum_bytes
        self.entries=OrderedDict();self.bytes=0;self.peak_bytes=0
        self.hits=0;self.misses=0;self.evictions=0;self.unversioned=0

    def get(self,name,sources,compute):
        versions=tensor_versions(sources)
        if versions is None:self.unversioned+=1;return compute()
        key=(name,versions)
        if key in self.entries:
            self.hits+=1;self.entries.move_to_end(key);return self.entries[key][0]
        self.misses+=1;value=compute();size=_bytes((key,sources,value))
        if size>self.maximum_bytes:return value
        while self.entries and (len(self.entries)>=self.maximum_entries or self.bytes+size>self.maximum_bytes):
            _,entry=self.entries.popitem(last=False);self.bytes-=entry[2];self.evictions+=1
        # Strong sources prevent identity reuse; include their full storage
        # (even for small views) in the conservative memory budget.
        self.entries[key]=(value,sources,size);self.bytes+=size;self.peak_bytes=max(self.peak_bytes,self.bytes)
        return value

    def report(self):
        return dict(hits=self.hits,misses=self.misses,evictions=self.evictions,unversioned=self.unversioned,
                    entries=len(self.entries),bytes=self.bytes,peak_bytes=self.peak_bytes,
                    maximum_bytes=self.maximum_bytes,maximum_entries=self.maximum_entries)


def workspace(scene):
    value=getattr(scene,'_response_index_workspace',None)
    if value is None:value=IndexWorkspace();scene._response_index_workspace=value
    return value


def _host(cache,source):
    return cache.get('host_indices',(source,),lambda:source.detach().cpu().numpy().astype(np.int64,copy=True))


def build(scene,terms,blocks):
    """Same component/row/slot order as the original Python lookup loops."""
    cache=workspace(scene)
    sources=(*scene.interaction_entities,*(p.ids for p,*_ in blocks),*(term.ids for term in terms))
    name=('complete',scene.n,len(scene.interaction_entities),len(blocks),len(terms),str(scene.device))
    return cache.get(name,sources,lambda:_build(scene,terms,blocks,cache))


def _build(scene,terms,blocks,cache):
    from .protected_response import ComponentTopology
    term_ids=[_host(cache,term.ids) for term in terms]
    repeated=set()
    for source,rows in zip((term.ids for term in terms),term_ids):
        def duplicates(rows=rows):
            ordered=np.sort(rows,axis=-1)
            return frozenset(ordered[:,1:][(ordered[:,1:]==ordered[:,:-1])&(ordered[:,1:]>=0)].tolist())
        repeated.update(cache.get('repeated_owners',(source,),duplicates))
    block_ids=[_host(cache,p.ids) for p,*_ in blocks];groups=[]
    def device(array,dtype=torch.long):return torch.from_numpy(np.array(array,copy=True)).to(device=scene.device,dtype=dtype)
    for entities in scene.interaction_entities:
        members=_host(cache,entities);member_key=tuple(members.tolist())
        inverse=np.full(scene.n,-1,dtype=np.int64);inverse[members]=np.arange(len(members))
        selected=[]
        for index,((plan,*_),ids) in enumerate(zip(blocks,block_ids)):
            def map_block(ids=ids):
                if not np.all((ids>=0)&(ids<scene.n)):return None
                mapped=inverse[ids]
                return device(mapped) if np.all(mapped>=0) else None
            # The immutable complete index contents also admit reuse when
            # native snapshots allocate a new tensor with identical owners.
            mapped=cache.get(('block_map',member_key,ids.shape,ids.tobytes()),(),map_block)
            if mapped is not None:selected.append((index,mapped))
        mapped=[]
        for index,(term,ids) in enumerate(zip(terms,term_ids)):
            def map_term(ids=ids):
                inside=(ids>=0)&(ids<scene.n)
                slots=np.where(inside,inverse[np.clip(ids,0,scene.n-1)],-1)
                retained=np.flatnonzero((slots>=0).any(-1))
                if not len(retained):return None
                slots=slots[retained]
                return device(retained),device(np.maximum(slots,0)),device(slots>=0,torch.bool)
            row_map=cache.get(('term_map',member_key,ids.shape,ids.tobytes()),(),map_term)
            if row_map is not None:mapped.append((index,*row_map))
        groups.append(ComponentTopology(entities,selected,mapped,not bool(set(member_key)&repeated)))
    return groups


def mobile_components(scene):
    """Cache only static member indices, including mobility mutation versions."""
    cache=workspace(scene);sources=(scene.movable,*scene.interaction_entities)
    return cache.get(('mobile_components',len(scene.interaction_entities)),sources,
                     lambda:[ids[scene.movable[ids]] for ids in scene.interaction_entities])


def bending_ids(scene):
    return workspace(scene).get('bending_ids',(scene.bend_edges,scene.bending_indices),
                                lambda:scene.bend_edges[scene.bending_indices])
