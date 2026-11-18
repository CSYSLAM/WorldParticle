"""Independent quadratic checks of smooth material response columns."""
import argparse,json,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
 import torch
 from wp_vbd.elastic_shape_modes import polynomial_basis,response_columns
 from wp_vbd.coarse_response import rigid_component_modes
 from wp_vbd import math3d as so3
 torch.set_num_threads(1);torch.manual_seed(319)
 n=37;points=torch.randn(1,n,3,dtype=torch.float64);mass=torch.rand(n,dtype=torch.float64)+.5
 basis=polynomial_basis(points,mass);center=(points*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
 rigid=rigid_component_modes(points,center,torch.ones(n,6,dtype=torch.bool))[...,:3,:]
 orthogonal=torch.einsum('bnip,n,bniq->bpq',rigid,mass,basis)
 assert float(orthogonal.abs().max())<1e-10
 scene=SimpleNamespace(n=n,kind=torch.ones(n,dtype=torch.long),movable=torch.ones(n,dtype=torch.bool),object_id=torch.zeros(n,dtype=torch.long),mass=mass)
 state=SimpleNamespace(x=points);ids=torch.arange(n)
 class Metric:
  terms=[]
  def __call__(self,x):return x*mass[None,:,None,None] if x.ndim==4 else x*mass[None,:,None]
 metric=Metric();target=torch.zeros(1,n,6,dtype=torch.float64);target[...,:3]=basis[...,5]*.003+basis[...,17]*.002
 residual=metric(target);column=response_columns(scene,state,metric,residual,ids)[...,0]
 actual=column*((column*residual).sum((1,2))/(column*metric(column)).sum((1,2)))[:,None,None]
 torch.testing.assert_close(actual,target,rtol=1e-8,atol=1e-10)
 work=float((residual*actual).sum());assert work>0
 # Global rigid columns have no response to this nonrigid physical load.
 assert float(torch.einsum('bnip,bni->bp',rigid,residual[...,:3]).abs().max())<1e-10
 rotation=so3.matrix(so3.exp(torch.tensor([[.3,-.4,.2]],dtype=torch.float64)))[0]
 rotated_points=points@rotation.T+points.new_tensor([.5,-.1,.8]);rotated_target=target.clone();rotated_target[...,:3]=target[...,:3]@rotation.T
 rotated_residual=metric(rotated_target);rotated_state=SimpleNamespace(x=rotated_points)
 other=response_columns(scene,rotated_state,metric,rotated_residual,ids)[...,0]
 other=other*((other*rotated_residual).sum((1,2))/(other*metric(other)).sum((1,2)))[:,None,None]
 torch.testing.assert_close(other,rotated_target,rtol=1e-8,atol=1e-10)
 zero=response_columns(scene,state,metric,torch.zeros_like(residual),ids);assert zero.shape[-1]==0
 result={'status':'PASS','rigid_mass_orthogonality_max_error':float(orthogonal.abs().max()),'quadratic_material_response_max_error_m':float((actual-target).abs().max()),'rigid_motion_covariant':True,'zero_residual_no_column':True,'positive_descent_work':work,'neural_parameter_changes':0,'native_dynamics_calls':0}
 a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
