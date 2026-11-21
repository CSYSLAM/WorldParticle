"""Compare isolated contact wrenches/metrics to pinned local Newton kernels."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import activate_newton_source,newton_revision
activate_newton_source()
import numpy as np
import torch
import warp as wp
from newton._src.solvers.vbd import rigid_vbd_kernels as source
from wp_vbd import math3d as so3
from wp_vbd.rigid_contact_rows import ContactContext,evaluate,compliant_contact_dual,numerical_weight_policy


# COM is flattened alongside the body poses, matching the source's body IDs.
@wp.kernel
def pinned(poses:wp.array[wp.transform],previous:wp.array[wp.transform],com:wp.array[wp.vec3],points:wp.array2d[wp.vec3],offsets:wp.array2d[wp.vec3],normal:wp.array[wp.vec3],surface:wp.array[wp.vec3],dual:wp.array[wp.vec3],c0:wp.array[wp.vec3],tangent:wp.array[wp.vec3],parameters:wp.array2d[float],mode:int,dt:float,wrenches:wp.array3d[float],metrics:wp.array4d[float],new_dual:wp.array[wp.vec3]):
    i=wp.tid()
    fa,ta,la,aa,ha,fb,tb,lb,ab,hb=source.evaluate_rigid_contact_from_collision(2*i,2*i+1,poses,previous,com,points[i,0],points[i,1],surface[i],offsets[i,0],offsets[i,1],normal[i],parameters[i,6],parameters[i,1],parameters[i,0],parameters[i,2],parameters[i,3],dual[i],parameters[i,4],parameters[i,5],int(mode==1),int(mode==0),dt,c0[i])
    for axis in range(3):
        wrenches[i,0,axis]=fa[axis];wrenches[i,0,axis+3]=ta[axis]
        wrenches[i,1,axis]=fb[axis];wrenches[i,1,axis+3]=tb[axis]
        for col in range(3):
            metrics[i,0,axis,col]=la[axis,col];metrics[i,1,axis,col]=lb[axis,col]
            metrics[i,0,axis+3,col]=aa[axis,col];metrics[i,1,axis+3,col]=ab[axis,col]
            metrics[i,0,axis,col+3]=aa[col,axis];metrics[i,1,axis,col+3]=ab[col,axis]
            metrics[i,0,axis+3,col+3]=ha[axis,col];metrics[i,1,axis+3,col+3]=hb[axis,col]
    new_dual[i]=source._compliant_contact_dual_step(dual[i],normal[i],parameters[i,6],tangent[i],parameters[i,0],parameters[i,4],parameters[i,1],parameters[i,2])


@wp.kernel
def weights(parameters:wp.array2d[float],output:wp.array2d[float]):
    i=wp.tid()
    normal=source._contact_auto_normal_rho(parameters[i,1],parameters[i,0])
    output[i,0]=normal
    output[i,1]=source._contact_auto_tangent_rho(parameters[i,2],normal,parameters[i,3])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/rigid_contact_rows_v1/verification.json')
    args=parser.parse_args()
    source_sha=newton_revision()
    assert source_sha=='25298aea1691e060c86f95e8d2b773c898c006d3','Pinned source changed'
    torch.set_num_threads(2);torch.manual_seed(731)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init()
    count=240;dt=1/600
    position=.1*torch.randn(count,2,3);rotation=so3.exp(.5*torch.randn(count,2,3))
    previous_position=position+.001*torch.randn_like(position)
    previous_rotation=so3.multiply(so3.exp(.01*torch.randn_like(position)),rotation)
    com=.02*torch.randn_like(position);points=.05*torch.randn_like(position);offsets=.01*torch.randn_like(position)
    # Identity/zero world-side geometry, without silently dropping its reaction.
    position[:20,0]=0;previous_position[:20,0]=0;rotation[:20,0]=torch.tensor([0.,0.,0.,1.]);previous_rotation[:20,0]=rotation[:20,0];com[:20,0]=0
    normal=so3.normalize(torch.randn(count,3));surface=.1*torch.randn(count,3)
    dual=torch.randn(count,3);c0=.0005*torch.randn(count,3);tangent=.001*torch.randn(count,3)
    c0-=normal*(c0*normal).sum(-1,keepdim=True)
    tangent-=normal*(tangent*normal).sum(-1,keepdim=True)
    material=10**(2+3*torch.rand(count));rho=10**(2+3*torch.rand(count));rho_t=rho*.7
    damping=torch.rand(count)*10;mu=torch.rand(count);epsilon=torch.full((count,),.01);depth=.04*torch.rand(count)-.02
    material[::17]=0;rho[::19]=0;mu[::13]=0;epsilon[::23]=0
    parameters=torch.stack((material,rho,rho_t,damping,mu,epsilon,depth),-1)
    def array(value,dtype):return wp.array(value.contiguous().numpy(),dtype=dtype,device='cpu')
    output=wp.zeros((count,2,6),dtype=float,device='cpu');metric=wp.zeros((count,2,6,6),dtype=float,device='cpu');trial=wp.zeros(count,dtype=wp.vec3,device='cpu')
    pose=torch.cat((position,rotation),-1).flatten(0,1);previous=torch.cat((previous_position,previous_rotation),-1).flatten(0,1)
    common=[array(pose,wp.transform),array(previous,wp.transform),array(com.flatten(0,1),wp.vec3),array(points,wp.vec3),array(offsets,wp.vec3),array(normal,wp.vec3),array(surface,wp.vec3),array(dual,wp.vec3),array(c0,wp.vec3),array(tangent,wp.vec3),array(parameters,float)]
    expected_weights=wp.zeros((count,2),dtype=float,device='cpu')
    policy=torch.stack((material,rho,rho_t*(.001+2*torch.rand(count)),10**(1+5*torch.rand(count))),-1)
    wp.launch(weights,count,[array(policy,float),expected_weights],device='cpu')
    actual_weights=torch.stack(numerical_weight_policy(policy[:,1],policy[:,2],policy[:,3],policy[:,0]),-1)
    np.testing.assert_array_equal(actual_weights.numpy(),expected_weights.numpy())
    results=[]
    for index,mode in enumerate(('compliant','legacy_hard','legacy_soft')):
        print('checking',mode,flush=True)
        ctx=ContactContext(material,rho,rho_t,damping,mu,dual,c0,epsilon,dt,mode)
        actual=evaluate(position,rotation,previous_position,previous_rotation,com,points,offsets,normal,depth,surface,ctx)
        assert bool(actual['valid'].all())
        wp.launch(pinned,count,common+[index,dt,output,metric,trial],device='cpu')
        wrench=torch.cat((actual['force'],actual['torque']),-1).numpy()
        np.testing.assert_allclose(wrench,output.numpy(),rtol=5e-5,atol=5e-5)
        np.testing.assert_allclose(actual['self_metric'].numpy(),metric.numpy(),rtol=5e-5,atol=3e-3)
        torch.testing.assert_close(actual['force'].sum(-2),torch.zeros_like(normal),rtol=0,atol=0)
        bound=ctx.mu*actual['normal_load_before_damping']
        assert bool((actual['tangent_force'].norm(dim=-1)<=bound*(1+2e-6)+1e-7).all())
        double=lambda v:v.double() if torch.is_tensor(v) else v
        from dataclasses import fields
        precise=ContactContext(**{f.name:double(getattr(ctx,f.name)) for f in fields(ctx)})
        result64=evaluate(*(double(v) for v in (position,so3.normalize(rotation.double()),previous_position,so3.normalize(previous_rotation.double()),com,points,offsets,so3.normalize(normal.double()),depth,surface)),precise)
        eigen=torch.linalg.eigvalsh(result64['self_metric']);scale=result64['self_metric'].abs().amax((-1,-2)).clamp_min(1)
        assert float((eigen[...,0]/scale).min())>-1e-12
        # Relative-point rows expose the cross-body coupling without a global
        # solve. The assembled pair metric must contain the verified self blocks.
        pair=torch.zeros(count,12,12,dtype=torch.float64)
        for key,weight in (('relative_skeleton_jacobian','normal_solve_metric'),('relative_surface_jacobian','tangent_solve_metric')):
            j=result64[key].transpose(-3,-2).reshape(count,3,12)
            pair+=j.transpose(-1,-2)@result64[weight]@j
        torch.testing.assert_close(pair[:,:6,:6],result64['self_metric'][:,0],rtol=1e-12,atol=1e-9)
        torch.testing.assert_close(pair[:,6:,6:],result64['self_metric'][:,1],rtol=1e-12,atol=1e-9)
        assert float((torch.linalg.eigvalsh(pair)[:,0]/pair.abs().amax((-1,-2)).clamp_min(1)).min())>-1e-12
        if mode=='compliant':
            snapshot=dual.clone();updated=compliant_contact_dual(dual,normal,depth,tangent,material,rho,rho_t,mu)
            np.testing.assert_allclose(updated.numpy(),trial.numpy(),rtol=5e-5,atol=5e-5)
            assert torch.equal(dual,snapshot) and updated.data_ptr()!=dual.data_ptr()
        poisoned=position.clone();poisoned[1,0,0]=float('nan')
        isolated=evaluate(poisoned,rotation,previous_position,previous_rotation,com,points,offsets,normal,depth,surface,ctx)
        assert not bool(isolated['valid'][1]) and bool(isolated['valid'][0])
        for key in ('force','torque','self_metric'):assert torch.equal(isolated[key][0],actual[key][0])
        results.append({'mode':mode,'contacts':count,'maximum_wrench_difference':float(np.abs(wrench-output.numpy()).max()),'maximum_metric_difference':float(np.abs(actual['self_metric'].numpy()-metric.numpy()).max()),'equal_opposite_pair_force_exact':True,'fp64_self_metric_psd':True})
    # Coincident effective anchors: internal torques must cancel about any
    # origin, including for nonzero body COM and noncentral body origins.
    q=so3.normalize(rotation[:1].double());x=position[:1].double();cm=com[:1].double()
    cp=so3.rotate(so3.conjugate(q),-x);zero=torch.zeros_like(cp)
    nn=torch.tensor([[0.,0.,1.]],dtype=torch.float64)
    scalar=lambda v:torch.tensor([v],dtype=torch.float64)
    ctx=ContactContext(scalar(1e4),scalar(1e5),scalar(1e4),scalar(1),scalar(.5),torch.tensor([[.2,.1,1.]],dtype=torch.float64),torch.tensor([[.001,0.,0.]],dtype=torch.float64),scalar(.01),dt)
    coincident=evaluate(x,q,x-.001,q,cm,cp,zero,nn,scalar(.001),nn*0,ctx)
    centers=x+so3.rotate(q,cm)
    moment=coincident['torque'].sum(-2)+torch.linalg.cross(centers,coincident['force']).sum(-2)
    assert float(moment.abs().max())<2e-9
    report={'status':'PASS','cases':results,'device':'cpu','newton_sha':'25298aea1691e060c86f95e8d2b773c898c006d3','operator_sha256':hashlib.sha256((ROOT/'src/wp_vbd/rigid_contact_rows.py').read_bytes()).hexdigest(),'compliant_dual_matches_and_preserves_input':True,'contact_numerical_weight_policy_matches':True,'pair_metric_cross_blocks_and_psd':True,'coulomb_bounds_checked':True,'same_batch_poison_isolation':True,'coincident_surface_anchor_global_moment_error':float(moment.abs().max()),'new_demo_training_admissions':0,'scope':'Independent local contact wrenches, friction regimes and PSD solve metrics only. No collision generation, contact identity/history lifecycle, continuous feasibility, complete learned integration or demo acceptance is claimed.'}
    report['reference_kernel_sha256']=hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest()
    report['verification_script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report['comparison_tolerances']={'wrench':{'rtol':5e-5,'atol':5e-5},'self_metric':{'rtol':5e-5,'atol':3e-3}}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
