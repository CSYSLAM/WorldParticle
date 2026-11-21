"""A per-compilation FP64 small-matrix pass, without global library patches."""
import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch._inductor.custom_graph_pass import CustomGraphPass,get_hash_for_files
from torch._inductor.pattern_matcher import Arg,CallFunction,PatternMatcherPass,register_graph_pattern


def _eligible(left,right,rank):
    a=left.meta.get('val');b=right.meta.get('val')
    if not isinstance(a,torch.Tensor) or not isinstance(b,torch.Tensor):return False
    if a.ndim!=rank or b.ndim!=rank or a.device.type!='cuda' or b.device!=a.device:return False
    if a.dtype!=torch.float64 or b.dtype!=torch.float64:return False
    if rank==3:
        return sum(statically_known_true(dimension<32) for dimension in (a.shape[1],a.shape[2],b.shape[2]))>=2
    return statically_known_true(a.shape[1]<32) and statically_known_true(b.shape[1]<32)


class SmallSpatialProducts(CustomGraphPass):
    """Decompose only proven small CUDA FP64 MM/BMM; retain unknown sizes."""
    def __init__(self):
        self.matcher=PatternMatcherPass(pass_name='wp_small_spatial_fp64')
        aten=torch.ops.aten

        def realize(nodes):
            for node in nodes:node.meta['inductor_realize_to_strides']=True

        @register_graph_pattern(CallFunction(aten.bmm,Arg(),Arg()),pass_dict=self.matcher)
        def bmm(match,a,b):
            if not _eligible(a,b,3):return
            def replacement(a,b):return (a[:,:,:,None]*b[:,None,:,:]).sum(-2).to(a.dtype)
            match.replace_by_example(replacement,[a,b]);realize((a,b))

        @register_graph_pattern(CallFunction(aten.mm,Arg(),Arg()),pass_dict=self.matcher)
        def mm(match,a,b):
            if not _eligible(a,b,2):return
            def replacement(a,b):return (a[:,:,None]*b[None,:,:]).sum(-2).to(a.dtype)
            match.replace_by_example(replacement,[a,b]);realize((a,b))

        @register_graph_pattern(CallFunction(aten.addmm,Arg(),Arg(),Arg()),pass_dict=self.matcher)
        def addmm(match,bias,a,b):
            if not _eligible(a,b,2):return
            if match.output_node().kwargs.get('alpha',1)!=1 or match.output_node().kwargs.get('beta',1)!=1:return
            def replacement(bias,a,b):return (a[:,:,None]*b[None,:,:]).sum(-2).to(a.dtype)+bias
            match.replace_by_example(replacement,[bias,a,b]);realize((a,b))

    def __call__(self,graph):self.matcher.apply(graph)

    def uuid(self):return get_hash_for_files((__file__,),extra='cuda-fp64-small32-v1')


SMALL_SPATIAL_PRODUCTS=SmallSpatialProducts()
