"""Optional bounded CUDA Graph compilation of reviewed pure physical operators.

Compiled outputs are cloned before returning: later graph replays must never
modify an accepted state, live metric, history or diagnostic. This module does
not capture collisions, neural control flow, or physical acceptance decisions.
"""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
import types

import torch
from torch.utils._pytree import tree_map

_enabled=ContextVar('wp_cuda_graph_enabled',default=False)
MAX_SPECIALIZATIONS=8
MAX_FUNCTION_MODES=64
_allowed={
    'wp_vbd.mixed_metric':{'_apply_tensor','_diagonal_tensor'},
    'wp_vbd.articulation_coarse':{'_raw_modes_tensor','_project_modes_tensor'},
    'wp_vbd.protected_response':{'_orthogonalize_tensor','_apply_coarse_tensor'},
    'wp_vbd.surface_joint_system':{'_candidate_merit_tensor'},
    'wp_vbd.rigid_contact_rows':{'_evaluate_reference'},
    'wp_vbd.mimic_constraint':{'_rows_tensor','_tangent_group_tensor','_primal_tensor',
                              '_dual_tensor','_correction_tensor','_retract_tensor'},
    'wp_vbd.general_joint_rows':{'coordinates'},
    'wp_vbd.mixed_contact_runtime':{'_motion_tensor'},
    'wp_vbd.contact_motion_step':{'_cap_tensor'},
    'wp_vbd.contact_ports':{'_state_validity_tensor'},
    'wp_vbd.contact_activation_bound':{'_curvature_tensor'},
}


@dataclass
class _Entry:
    compiled:object
    name:str
    disabled:bool=False
    calls:int=0


_entries={}
_capacity_fallbacks=0


def enabled():return _enabled.get() and not torch.is_grad_enabled()


@contextmanager
def capture(enabled):
    token=_enabled.set(bool(enabled) and not torch.is_grad_enabled())
    try:yield
    finally:_enabled.reset(token)


def statistics():
    """JSON-safe counters; no CUDA synchronization or retained physical data."""
    return {'function_modes':len(_entries),'active_function_modes':sum(not e.disabled for e in _entries.values()),
            'disabled_function_modes':sum(e.disabled for e in _entries.values()),
            'maximum_function_modes':MAX_FUNCTION_MODES,'maximum_specializations_per_function':MAX_SPECIALIZATIONS,
            'capacity_fallbacks':_capacity_fallbacks,
            'specialization_fallbacks':[{'function':e.name,'reason':'graph specialization budget reached'}
                                        for e in _entries.values() if e.disabled]}


def evaluate(function,small,fallback,*args,**kwargs):
    """Use a graph when eligible; fallback evaluates the SAME current operator.

    Only the Dynamo specialization-limit exception is handled. Numerical,
    geometry, physical guard, CUDA, and other compilation errors propagate.
    The reviewed operators do not mutate their physical inputs, so retrying
    after a pre-execution specialization guard is safe.
    """
    global _capacity_fallbacks
    cuda_input=any(isinstance(a,torch.Tensor) and a.is_cuda for a in args)
    if not enabled() or not cuda_input or function.__name__ not in _allowed.get(function.__module__,()):
        return fallback()
    key=(function,bool(small))
    entry=_entries.get(key)
    if entry is None:
        if len(_entries)>=MAX_FUNCTION_MODES:
            _capacity_fallbacks+=1
            return fallback()
        options={'triton.cudagraphs':True,'triton.cudagraph_trees':False,
                 'triton.cudagraph_support_input_mutation':False}
        if small:
            from .spatial_compilation import SMALL_SPATIAL_PRODUCTS
            options['post_grad_custom_pre_pass']=SMALL_SPATIAL_PRODUCTS
        # Isolate graph specializations from the original dynamic compiler.
        # Its cache must remain usable when this optional graph hits its cap.
        code=function.__code__.replace(co_name=function.__name__+'_bounded_graph')
        isolated=types.FunctionType(code,function.__globals__,function.__name__,function.__defaults__,function.__closure__)
        isolated.__kwdefaults__=function.__kwdefaults__
        with torch._dynamo.config.patch(cache_size_limit=MAX_SPECIALIZATIONS):
            compiled=torch.compile(isolated,fullgraph=True,dynamic=True,options=options)
        entry=_Entry(compiled,function.__module__+'.'+function.__name__)
        _entries[key]=entry
    if entry.disabled:return fallback()
    from torch._dynamo.exc import FailOnRecompileLimitHit
    try:
        with torch._dynamo.config.patch(cache_size_limit=MAX_SPECIALIZATIONS):
            result=entry.compiled(*args,**kwargs)
        entry.calls+=1
    except FailOnRecompileLimitHit:
        entry.disabled=True
        entry.compiled=None
        return fallback()
    def copy(value):
        if isinstance(value,torch.Tensor):return value.clone()
        if value is None or isinstance(value,(bool,int,float,str)):return value
        raise TypeError('Unreviewed CUDA Graph output type: '+str(type(value)))
    return tree_map(copy,result)
