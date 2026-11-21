"""Verify bounded graph inference on actual sparse physical operators."""
from pathlib import Path
import argparse,sys,json,importlib.util
ROOT=Path(__file__).resolve().parents[1]
if not (ROOT/'src/wp_vbd').is_dir():ROOT=Path.cwd()
sys.path.insert(0,str(ROOT/'src'))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--report',type=Path,required=True)
 p.add_argument('--candidate',type=Path,help='Isolated graph module before installation')
 a=p.parse_args()
 from wp_vbd.gpu_ownership import claim_cuda
 ownership=claim_cuda('cuda:0',ROOT)
 import torch
 from wp_vbd import compiled_contact
 if a.candidate:
  import wp_vbd
  spec=importlib.util.spec_from_file_location('wp_vbd.cuda_graph',a.candidate)
  graph=importlib.util.module_from_spec(spec);sys.modules[spec.name]=graph;setattr(wp_vbd,'cuda_graph',graph);spec.loader.exec_module(graph)
  original=getattr(compiled_contact,'_evaluate_plain',compiled_contact.evaluate)
  def evaluate(function,*args,**kwargs):
   return graph.evaluate(function,compiled_contact._small_products.get(),lambda:original(function,*args,**kwargs),*args,**kwargs)
  compiled_contact.evaluate=evaluate
 else:
  from wp_vbd import cuda_graph as graph
 from wp_vbd.mixed_metric import _apply_tensor,_diagonal_tensor,MetricTerm
 from wp_vbd.contact_ports import _state_validity_tensor
 from wp_vbd.protected_response import _orthogonalize_tensor
 torch.set_num_threads(1);torch.manual_seed(19);torch.use_deterministic_algorithms(False)
 initial_limit=torch._dynamo.config.cache_size_limit
 retained=[];rows=[]
 assert not graph.enabled()
 with torch.no_grad(),graph.capture(True),compiled_contact.small_spatial_products(True):
  assert graph.enabled()
  with graph.capture(False):assert not graph.enabled()
  assert graph.enabled()
  with torch.enable_grad():assert not graph.enabled()
  for n,count,columns in [(12,4,3),(12,6,3),(12,4,6),(12,4,3),(15,5,3),(12,0,3),(12,4,3)]:
   local=torch.diag_embed(1+torch.rand(1,n,6,device='cuda',dtype=torch.float64))
   ids=torch.randint(n,(count,3),device='cuda');ids[:,2]=ids[:,1]
   jacobian=torch.randn(1,count,2,3,6,device='cuda',dtype=torch.float64)
   weight=torch.rand(1,count,2,device='cuda',dtype=torch.float64)
   dof=torch.ones(n,6,device='cuda',dtype=torch.bool);dof[::3]=False
   value=torch.randn(1,n,6,columns,device='cuda',dtype=torch.float64)
   terms=[MetricTerm(ids,jacobian,weight)]
   expected=_apply_tensor(local,terms,dof,value)
   actual=compiled_contact.evaluate(_apply_tensor,local,terms,dof,value)
   error=float((expected-actual).abs().max());assert error<1e-10
   retained.append((actual,actual.clone()))
   ids.copy_(ids.flip(0));dof.logical_not_();value.mul_(1.3);weight.mul_(2)
   expected=_apply_tensor(local,terms,dof,value)
   second=compiled_contact.evaluate(_apply_tensor,local,terms,dof,value)
   assert float((expected-second).abs().max())<1e-10
   assert all(torch.equal(x,y) for x,y in retained)
   rows.append({'entities':n,'contact_rows':count,'columns':columns,'max_error':error})
  for n in [7,11,7]:
   x=torch.zeros(1,n,3,device='cuda',dtype=torch.float64)
   q=torch.zeros(1,n,4,device='cuda',dtype=torch.float64);q[...,3]=1
   com=torch.zeros(n,3,device='cuda',dtype=torch.float64);mask=torch.ones(n,device='cuda',dtype=torch.bool)
   good=compiled_contact.evaluate(_state_validity_tensor,x,q,com,mask);assert bool(good.all())
   x[0,0,0]=float('nan')
   bad=compiled_contact.evaluate(_state_validity_tensor,x,q,com,mask)
   assert torch.equal(bad,_state_validity_tensor(x,q,com,mask)) and not bool(bad.all()) and bool(good.all())
  maximum=graph.MAX_SPECIALIZATIONS
  try:
   graph.MAX_SPECIALIZATIONS=2
   for columns in [3,6,12,6]:
    z=torch.randn(1,9,6,columns,device='cuda',dtype=torch.float64)
    energy=torch.randn(1,71,columns,device='cuda',dtype=torch.float64)
    z[...,-1]=z[...,0];energy[...,-1]=energy[...,0]
    expected=_orthogonalize_tensor(z,energy,1e-12)
    actual=compiled_contact.evaluate(_orthogonalize_tensor,z,energy,1e-12)
    assert torch.equal(expected[1],actual[1]) and float((expected[0]-actual[0]).abs().max())<1e-10
  finally:graph.MAX_SPECIALIZATIONS=maximum
  stats=graph.statistics()
  assert any(r['function']=='wp_vbd.protected_response._orthogonalize_tensor' for r in stats['specialization_fallbacks'])
  capacity=graph.MAX_FUNCTION_MODES
  try:
   graph.MAX_FUNCTION_MODES=len(graph._entries)
   expected=_diagonal_tensor(local,terms,dof)
   actual=compiled_contact.evaluate(_diagonal_tensor,local,terms,dof)
   assert float((expected-actual).abs().max())<1e-10
   assert graph.statistics()['capacity_fallbacks']>=1
  finally:graph.MAX_FUNCTION_MODES=capacity
 assert not graph.enabled() and torch._dynamo.config.cache_size_limit==initial_limit
 try:
  with torch.no_grad(),graph.capture(True):raise RuntimeError('Synthetic scope failure')
 except RuntimeError:pass
 assert not graph.enabled()
 with graph.capture(True):assert not graph.enabled()  # Grad-enabled training.
 result={'status':'PASS','candidate':str(a.candidate) if a.candidate else None,'cases':rows,
         'dynamic_index_mobility_and_output_lifetime':True,'finite_state_refresh':True,
         'coarse_rank_and_specialization_fallback':True,'function_capacity_fallback':True,
         'nested_exception_and_training_scope_restored':True,'statistics':graph.statistics(),
         'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'peak_reserved_bytes':torch.cuda.max_memory_reserved()}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
