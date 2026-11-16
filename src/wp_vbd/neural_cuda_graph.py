"""Optional inference-only CUDA replay of pure attention blocks.

Every input is copied on each call. Physics, contact queries, topology plans,
acceptance and history commit stay outside these graphs.
"""
from collections import OrderedDict

import torch


class AttentionReplay:
    def __init__(self, module, maximum_entries=2):
        if maximum_entries<1:raise ValueError('Positive neural graph cache capacity required')
        self.module=module
        self.maximum_entries=maximum_entries
        self.entries=OrderedDict()
        self.seen=OrderedDict()
        self.calls=self.hits=self.captures=self.evictions=self.warmup_evaluations=0

    def __call__(self, query, context, sizes):
        self.calls+=1
        if self.module.training or torch.is_grad_enabled() or query.device.type!='cuda':
            return self.module._eager_attention(query,context,sizes)
        if torch.cuda.is_current_stream_capturing():
            return self.module._eager_attention(query,context,sizes)
        if any(t.device!=query.device for t in (context,sizes)):
            raise ValueError('A neural replay requires inputs on one CUDA device')
        autocast_enabled=torch.is_autocast_enabled(query.device.type)
        autocast_dtype=torch.get_autocast_dtype(query.device.type) if autocast_enabled else None
        key=(query.device,self.module.width,self.module.heads,tuple((tuple(t.shape),t.dtype) for t in (query,context,sizes)),
             autocast_enabled,autocast_dtype,
             tuple(p.data_ptr() for p in self.module.parameters()))
        if key not in self.entries:
            self.seen[key]=self.seen.get(key,0)+1
            self.seen.move_to_end(key)
            while len(self.seen)>32:self.seen.popitem(last=False)
            # Avoid compiling one-off topology shapes. These calls execute the
            # same pure NN; they do not advance physics or consume a teacher.
            if self.seen[key]<3:
                return self.module._eager_attention(query,context,sizes)
            if len(self.entries)>=self.maximum_entries:
                # A graph's private pool must outlive all queued users. The
                # barrier is paid only at eviction, never on a cache hit.
                torch.cuda.synchronize(query.device)
                self.entries.popitem(last=False);self.evictions+=1
            current=torch.cuda.current_stream(query.device)
            stream=torch.cuda.Stream(device=query.device);stream.wait_stream(current)
            buffers=tuple(t.clone(memory_format=torch.contiguous_format) for t in (query,context,sizes))
            # Clones were enqueued on current after the initial dependency.
            stream.wait_stream(current)
            # Capture casts from the persistent FP32 parameters too. Autocast
            # weight-cache buffers may disappear when the caller exits its
            # precision context, and must not become graph inputs.
            with torch.cuda.stream(stream),torch.no_grad(),torch.autocast(
                    query.device.type,enabled=autocast_enabled,dtype=autocast_dtype,cache_enabled=False):
                for _ in range(3):self.module._eager_attention(*buffers)
            current.wait_stream(stream)
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream),torch.no_grad(),torch.autocast(
                    query.device.type,enabled=autocast_enabled,dtype=autocast_dtype,cache_enabled=False):
                output=self.module._eager_attention(*buffers)
            current.wait_stream(stream)
            self.entries[key]=(buffers,output,graph)
            self.captures+=1;self.warmup_evaluations+=4
        else:
            self.hits+=1
            self.entries.move_to_end(key)
        buffers,output,graph=self.entries[key]
        for target,value in zip(buffers,(query,context,sizes)):target.copy_(value)
        graph.replay()
        current=torch.cuda.current_stream(query.device)
        for tensor in (*buffers,output):tensor.record_stream(current)
        # Owned output: later replay cannot mutate an earlier NN result.
        return output.clone()


def enable_attention_replay(model,maximum_entries=2):
    from .heterogeneous_backbone import AttentionBlock
    if model.training:raise ValueError('Neural replay is an inference-only option')
    layers=[module for module in model.modules() if isinstance(module,AttentionBlock)]
    if not layers:raise ValueError('Attention hierarchy required')
    for module in layers:
        if getattr(module,'_attention_replay',None) is not None:
            raise ValueError('Neural replay is already configured')
        module._attention_replay=AttentionReplay(module,maximum_entries)


def attention_replay_report(model):
    caches=[getattr(module,'_attention_replay',None) for module in model.modules()]
    caches=[cache for cache in caches if cache is not None]
    return {'enabled':bool(caches),'attention_blocks':len(caches),
            'autocast_policy_in_cache_key':True,'autocast_weight_cache_during_capture':False,
            **{name:sum(getattr(c,name) for c in caches) for name in ('calls','hits','captures','evictions','warmup_evaluations')},
            'resident_graphs':sum(len(c.entries) for c in caches),
            'resident_input_output_bytes':sum(t.numel()*t.element_size() for c in caches for buffers,output,graph in c.entries.values() for t in (*buffers,output)),
            'scope':'Pure attention inference only; all query/context/multiplicity values refreshed. Training or grad-enabled calls use eager. Physics, collision/topology, fixed physical budget and acceptance/history logic are unchanged. Warmup/capture NN evaluations are counted and included in wall time. Logical buffer bytes exclude graph temporary pools.'}
