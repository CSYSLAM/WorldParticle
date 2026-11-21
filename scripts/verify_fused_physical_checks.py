"""Compare fused metadata/activation checks to a frozen pre-change source."""
import argparse,sys,json,importlib.util
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    import torch
    from wp_vbd.unified_graph import build_topology
    from wp_vbd.contact_activation_bound import normal_activation_curvature
    from wp_vbd.cuda_graph import capture
    from wp_vbd.compiled_contact import small_spatial_products
    old_graph=load('wp_vbd._v29_topology',a.baseline/'src/wp_vbd/unified_graph.py')
    old_bound=load('wp_vbd._v29_bound',a.baseline/'src/wp_vbd/contact_activation_bound.py')
    torch.set_num_threads(1);torch.manual_seed(47)
    cases=[]
    for n,rows in [(0,0),(1,0),(1,9),(8,32),(31,123)]:
        for dtype in (torch.int32,torch.int64):
            kind=torch.randint(4,(n,),device=a.device,dtype=torch.long)
            movable=torch.rand(n,device=a.device)>.3
            obj=torch.full((n,),2**60+7,device=a.device,dtype=torch.long)
            ids=torch.randint(-1,max(1,n),(rows,4),device=a.device,dtype=dtype)
            if rows:ids[:,0]=torch.randint(n,(rows,),device=a.device,dtype=dtype)
            kinds=torch.randint(16,(rows,),device=a.device)
            active=torch.rand(rows,device=a.device)>.4
            kwargs=dict(potential_active=active)
            old=old_graph.build_topology(kind,movable,obj,ids,kinds,**kwargs)
            new=build_topology(kind,movable,obj,ids,kinds,make_slots=False,fused_host=True,**kwargs)
            assert torch.equal(old.interaction_id,new.interaction_id) and old.interaction_id.dtype==new.interaction_id.dtype
            assert new.layout.slots==() and new.layout.incidence is ids and new.potential_active is active
            cases.append({'nodes':n,'rows':rows,'incidence_dtype':str(dtype)})
    n=4;kind=torch.zeros(n,device=a.device,dtype=torch.long);mobile=torch.ones(n,device=a.device,dtype=torch.bool)
    obj=kind.clone();ids=torch.tensor([[0,1,-1]],device=a.device);rk=torch.zeros(1,device=a.device,dtype=torch.long)
    def error(function,*args,**kwargs):
        try:function(*args,**kwargs)
        except (ValueError,RuntimeError,IndexError) as e:return type(e).__name__,str(e)
        raise AssertionError('Expected rejection')
    bad_cases=[]
    for label,k,m,i,r,kw in [
        ('kind',kind-1,mobile,ids,rk,{}),
        ('owner',kind,mobile,ids+7,rk,{}),
        ('relation',kind,mobile,ids,rk+16,{}),
        ('mobility',kind,mobile.float(),ids,rk,{}),
        ('empty_enabled',kind,mobile,ids*0-1,rk,{}),
        ('mask',kind,mobile,ids,rk,{'potential_active':torch.ones(1,device=a.device)}),
        ('batch_id',kind,mobile,ids,rk,{'batch_id':torch.tensor([2**60,2**60+1,0,0],device=a.device)}),
    ]:
        expected=error(old_graph.build_topology,k,m,obj,i,r,**kw)
        actual=error(build_topology,k,m,obj,i,r,make_slots=False,fused_host=True,**kw)
        assert expected==actual,(label,expected,actual);bad_cases.append(label)
    bound_cases=[]
    with torch.no_grad(),capture(a.device.startswith('cuda')),small_spatial_products(True):
        for shape in [(0,),(41,),(2,79)]:
            raw=torch.randn(shape,device=a.device,dtype=torch.float64)*.1
            linear=torch.randn(shape,device=a.device,dtype=torch.float64)*.2
            remainder=torch.rand(shape,device=a.device,dtype=torch.float64)*.05
            shift=torch.randn(shape,device=a.device,dtype=torch.float64)*.03
            material=torch.full(shape,1e5,device=a.device,dtype=torch.float64)
            values=(raw,linear,remainder,shift,material)
            expected=old_bound.normal_activation_curvature(*values)
            actual=normal_activation_curvature(*values,compiled=True)
            torch.testing.assert_close(actual,expected,atol=1e-12,rtol=3e-13)
            bound_cases.append({'shape':list(shape),'max_error':float((expected-actual).abs().max()) if expected.numel() else 0.})
        tensor=lambda x:torch.tensor([x],device=a.device,dtype=torch.float64)
        for label,values in [
            ('nan',(tensor(float('nan')),tensor(.2),tensor(0),tensor(0),tensor(1))),
            ('negative',(tensor(-.1),tensor(.2),tensor(-.3),tensor(0),tensor(1))),
            ('overflow',(tensor(-.1),tensor(1e308),tensor(0),tensor(0),tensor(1))),
            ('priority',(tensor(float('inf')),tensor(.2),tensor(-1),tensor(0),tensor(-1))),
        ]:
            expected=error(old_bound.normal_activation_curvature,*values)
            actual=error(normal_activation_curvature,*values,compiled=True)
            assert expected==actual,(label,expected,actual);bad_cases.append('bound_'+label)
    result={'status':'PASS','device':a.device,'exact_component_labels':cases,'bound_cases':bound_cases,
            'unchanged_rejection_messages_and_priority':bad_cases,'int64_metadata_not_routed_through_float':True,
            'only_unused_layout_slots_omitted':True}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
