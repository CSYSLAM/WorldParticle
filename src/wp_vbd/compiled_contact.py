"""Optional fusion of the existing tensor-only contact constitutive operator.

All physical formulas, FP64 input/output types and validity masks are retained.
Compilation is lazy; its cold cost must be reported separately from warm timing.
No native dynamics or approximation is introduced.
"""
from contextlib import contextmanager
from contextvars import ContextVar
import torch

_compiled={}
_small_products=ContextVar('wp_small_spatial_products',default=False)


@contextmanager
def small_spatial_products(enabled):
    """Scope a compiler option and a bounded specialization budget."""
    enabled=bool(enabled)
    previous=_small_products.get()
    if previous==enabled and (not enabled or torch._dynamo.config.cache_size_limit>=32):
        yield;return
    token=_small_products.set(enabled)
    try:
        if enabled:
            # Coarse column counts vary within the declared 6..18 budget.
            # Keep these finite specializations instead of changing the basis.
            with torch._dynamo.config.patch(cache_size_limit=max(32,torch._dynamo.config.cache_size_limit)):
                yield
        else:yield
    finally:_small_products.reset(token)


def evaluate(function,*args,**kwargs):
    from . import cuda_graph
    if cuda_graph.enabled():
        small=_small_products.get() and not torch.is_grad_enabled()
        return cuda_graph.evaluate(function,small,lambda:_evaluate_plain(function,*args,**kwargs),*args,**kwargs)
    return _evaluate_plain(function,*args,**kwargs)


def _evaluate_plain(function,*args,**kwargs):
    small=_small_products.get() and not torch.is_grad_enabled()
    key=(function,small)
    if key not in _compiled:
        options={'triton.cudagraphs':False}
        if small:
            from .spatial_compilation import SMALL_SPATIAL_PRODUCTS
            options['post_grad_custom_pre_pass']=SMALL_SPATIAL_PRODUCTS
        _compiled[key]=torch.compile(function,fullgraph=True,dynamic=True,options=options)
    return _compiled[key](*args,**kwargs)
