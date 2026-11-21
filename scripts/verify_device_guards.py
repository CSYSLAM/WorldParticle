"""Preserve motion envelopes, invalid-state rejection and diagnostic values."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--spatial-products',action='store_true');a=p.parse_args()
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    from wp_vbd.compiled_contact import small_spatial_products
    with small_spatial_products(a.spatial_products):verify(a)


def verify(a):
    import torch
    from wp_vbd import math3d as so3
    from wp_vbd.data import State
    from wp_vbd.mixed_contact_runtime import MixedCollisionOracle
    from wp_vbd.contact_motion_step import proposal_scale
    from wp_vbd.contact_ports import PointPorts
    from wp_vbd.diagnostic_io import to_plain
    torch.set_num_threads(1);torch.manual_seed(2719)
    device=torch.device(a.device);dtype=torch.float64;n=9;batch=2;budget=.1
    radius=torch.tensor([0.,.02,.3]*3,device=device,dtype=dtype)
    movable=torch.ones(n,device=device,dtype=torch.bool);movable[-1]=False
    dof=torch.cat((torch.ones(n,3,device=device,dtype=torch.bool),(radius>0)[:,None].expand(-1,3)),1)&movable[:,None]
    scene=SimpleNamespace(n=n,movable=movable,dof=dof,body_radius=radius,compiled_response_kernels=False)
    oracle=object.__new__(MixedCollisionOracle);oracle.scene=scene;oracle.motion_budget=budget
    scene.geometry_oracle=oracle
    x=torch.randn(batch,n,3,device=device,dtype=dtype);q=so3.exp(torch.randn_like(x)*.3)
    start=State(x,q,torch.zeros_like(x),torch.zeros_like(x),{'sentinel':x.new_tensor([2719.])})
    checks={};error=0.;cases=0
    with torch.no_grad():
        for fraction in (0.,.5,.9999,1.01):
            direction=torch.randn(batch,n,6,device=device,dtype=dtype)*dof
            rate=direction[...,:3].norm(dim=-1)+radius*direction[...,3:].norm(dim=-1)
            direction*=fraction*budget/rate.clamp_min(1e-30)[...,None]
            current=State(x+direction[...,:3],so3.multiply(so3.exp(direction[...,3:]),q),start.v,start.omega,start.history)
            for magnitude in (0.,.001,2.):
                update=torch.randn_like(direction)*dof*magnitude
                scene.compiled_response_kernels=False;old=proposal_scale(scene,start,current,update)
                scene.compiled_response_kernels=True;new=proposal_scale(scene,start,current,update)
                assert torch.equal(old[1],new[1]);torch.testing.assert_close(old[0],new[0],rtol=1e-12,atol=1e-13)
                plain=to_plain(new[2]);assert plain['valid']==old[2]['valid']
                for name in ('scale','maximum_current_surface_travel','maximum_bounded_surface_travel'):
                    torch.testing.assert_close(torch.tensor(plain[name]),torch.tensor(old[2][name]),rtol=1e-6,atol=1e-8)
                error=max(error,float((new[0]-old[0]).abs().max()));cases+=1
                if fraction>1:assert not bool(new[1].any()) and not bool(new[0].any())
                else:
                    for alpha in (0.,.0625,.125,.25,.5,1.):
                        scale=new[0][:,None,None]*alpha
                        trial=State(current.x+scale*update[...,:3],so3.multiply(so3.exp(scale*update[...,3:]),current.q),start.v,start.omega)
                        assert bool(oracle.motion(start,trial)[0].all())
        checks['same_caps_and_diagnostic_values']=True
        checks['all_original_ray_scales_stay_inside_motion_budget']=True
        bad=torch.zeros_like(direction);bad[0,0,0]=float('nan')
        cap,valid,_=proposal_scale(scene,start,start,bad)
        assert not valid[0] and cap[0]==0 and valid[1]
        checks['nonfinite_proposal_and_invalid_current_rejected']=True
        original_radius=scene.body_radius
        for bad_radius in (radius*-1,torch.full_like(radius,float('nan'))):
            scene.body_radius=bad_radius
            try:proposal_scale(scene,start,start,torch.zeros_like(direction))
            except ValueError as exc:assert str(exc)=='Finite nonnegative physical surface radii required'
            else:raise AssertionError('Bad radius accepted')
        scene.body_radius=original_radius;checks['invalid_radius_rejected']=True
        # An empty movable mask retains the vacuous motion predicate and
        # zero diagnostic maxima; fixed orientations never become bodies.
        scene.movable=torch.zeros_like(movable);scene.dof=torch.zeros_like(dof)
        cap,valid,detail=proposal_scale(scene,start,start,torch.zeros_like(direction))
        assert bool(valid.all()) and torch.equal(cap,torch.ones_like(cap))
        assert to_plain(detail)['maximum_current_surface_travel']==[0.,0.]
        scene.movable=movable;scene.dof=dof;checks['all_fixed_scene_supported']=True
        ports=PointPorts.__new__(PointPorts);ports.node_count=n;ports.body_mask=radius>0
        com=x.new_zeros(n,3)
        for failure in ('none','position_nan','body_q_zero','particle_q_zero','com_nan'):
            xx=x.clone();qq=q.clone();cc=com.clone()
            if failure=='position_nan':xx[0,0,0]=float('nan')
            if failure=='body_q_zero':qq[0,1]=0.
            if failure=='particle_q_zero':qq[0,0]=0.
            if failure=='com_nan':cc[0,0]=float('nan')
            results=[]
            for enabled in (False,True):
                ports.compiled_response_kernels=enabled
                try:ports._validate_state_uncached(State(xx,qq,None,None),cc)
                except ValueError as exc:results.append(str(exc))
                else:results.append('PASS')
            assert results[0]==results[1],(failure,results)
        checks['same_finite_and_orientation_rejections']=True
    source=torch.arange(12,device=device,dtype=dtype).reshape(3,4).requires_grad_(True)
    integer=torch.tensor([2**60+3,-2**60+5],device=device,dtype=torch.int64)
    boolean=torch.tensor([True,False],device=device)
    diagnostic={'a':source.T,'b':[integer,boolean,source[0,0]],'empty':source[:0], 'alias':source.T,'tuple':(integer,'text')}
    def legacy(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().tolist()
        if isinstance(value,dict):return {key:legacy(child) for key,child in value.items()}
        if isinstance(value,list):return [legacy(child) for child in value]
        if isinstance(value,tuple):return tuple(legacy(child) for child in value)
        return value
    expected=legacy(diagnostic)
    for maximum in (1,32,8*1024**2):
        actual=to_plain(diagnostic,maximum_pack_bytes=maximum)
        assert actual==expected and json.dumps(actual)==json.dumps(expected)
    assert source.grad is None and source.requires_grad
    checks['bounded_diagnostic_packs_preserve_dtype_shape_alias_and_gradient']=True
    result={'status':'PASS','device':a.device,'small_spatial_products':a.spatial_products,'checks':checks,'motion_cases':cases,'maximum_cap_difference':error,
            'native_dynamics_calls':0,'optimizer_updates':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
