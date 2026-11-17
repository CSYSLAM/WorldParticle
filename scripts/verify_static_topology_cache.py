"""Compare cached topology against the original math on actual Franka states."""
import argparse
from contextlib import contextmanager
import importlib.util
import hashlib
import io
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import activate_newton_source
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-root',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--checkpoint',type=Path,default=ROOT/'checkpoints/plateau_review_v303/shared_trial_2478.pt')
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    activate_newton_source()
    if args.report.exists():raise FileExistsError(args.report)
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(args.device,ROOT)
    import torch
    import warp as wp
    import newton
    from wp_vbd import mimic_constraint as mimic,articulation_coarse as articulation
    from wp_vbd.franka_reference_data import load_reference
    from wp_vbd.mixed_experiment import select,loss,external_inputs
    from wp_vbd.mixed_contact_data import query
    from wp_vbd.general_joint_data import set_controls
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.static_topology import tensor_versions
    torch.set_num_threads(1);torch.manual_seed(81);torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(args.device)
    if args.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    def forbidden(*unused,**kwargs):raise AssertionError('No complete dynamics solver in topology verification')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    def baseline(name):
        spec=importlib.util.spec_from_file_location('wp_vbd._reference_'+name,args.baseline_root/'src/wp_vbd'/f'{name}.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
    old_mimic=baseline('mimic_constraint');old_articulation=baseline('articulation_coarse')
    @contextmanager
    def reference_math():
        saved=(mimic.Tangent,articulation.topology,articulation.raw_modes)
        mimic.Tangent=old_mimic.Tangent;articulation.topology=old_articulation.topology;articulation.raw_modes=old_articulation.raw_modes
        try:yield
        finally:mimic.Tangent,articulation.topology,articulation.raw_modes=saved
    scene,states=load_reference(ROOT/'.local/franka_v310/franka_grasp_causal_h600.npz',wp,newton,args.device)
    maximum=0.
    def equal(a,b):
        nonlocal maximum
        assert a.shape==b.shape
        if a.dtype==torch.bool or not a.is_floating_point():assert torch.equal(a,b);return
        maximum=max(maximum,float((a-b).abs().max()) if a.numel() else 0.)
        assert torch.allclose(a,b,atol=2e-12,rtol=2e-12),(a-b).abs().max()
    def compare(state):
        a=old_mimic.Tangent(scene,state);b=mimic.Tangent(scene,state)
        equal(a.valid,b.valid);assert len(a.groups)==len(b.groups)
        for ga,gb in zip(a.groups,b.groups):
            for va,vb in zip(ga,gb):equal(va,vb)
        value=torch.randn_like(state.x).repeat(1,1,2)*1e-5
        equal(a.primal(value),b.primal(value));equal(a.dual(value),b.dual(value))
        for va,vb in zip(a.correction(),b.correction()):equal(va,vb)
        try:
            with reference_math():reference=old_articulation.raw_modes(scene,state)
        except ValueError as e:
            try:articulation.raw_modes(scene,state)
            except ValueError as other:assert str(e)==str(other)
            else:raise AssertionError('Changed tree topology admission')
        else:equal(reference,articulation.raw_modes(scene,state))
    checked=[0,520,720,1350]
    for index in checked:compare(select(states,slice(index,index+1)))
    compare(select(states,torch.tensor([520,720],device=args.device)))
    current=select(states,slice(720,721));invalidations=[]
    def mutation(label,value,operation):
        mimic.Tangent(scene,current);before=scene._mimic_group_topology[0]
        saved=value.clone();operation(value)
        mimic.Tangent(scene,current);assert scene._mimic_group_topology[0]!=before
        compare(current);value.copy_(saved);compare(current);invalidations.append(label)
    mutation('zero_mimic_multiplier',scene.mimic_spec.coefficients,lambda v:v[:,1].zero_())
    mutation('disabled_follower',scene.joint_spec.enabled,lambda v:v[-1:].fill_(False))
    mutation('changed_parent_owner',scene.joint_spec.owners,lambda v:v[-1:,0].fill_(scene.world))
    owner=int(scene.joint_spec.owners[-1,1])
    mutation('changed_mobility',scene.movable,lambda v:v[owner:owner+1].fill_(False))
    original=scene.mimic_spec.coefficients;scene.mimic_spec.coefficients=original.clone();compare(current)
    scene.mimic_spec.coefficients=original;compare(current);invalidations.append('replaced_tensor')
    # Explicitly supplied rows may have arbitrary ownership; do not reuse a
    # source-asset topology for this separate public API.
    evaluated=mimic.rows(scene,current)
    from types import SimpleNamespace
    minimal=SimpleNamespace(movable=scene.movable,mass=scene.mass,inverse_inertia=scene.inverse_inertia,n=scene.n)
    for a,b in zip(old_mimic.Tangent(minimal,current,evaluated).correction(),mimic.Tangent(minimal,current,evaluated).correction()):equal(a,b)
    frozen=SimpleNamespace(movable=torch.zeros_like(scene.movable),mass=scene.mass,inverse_inertia=scene.inverse_inertia,n=scene.n)
    for a,b in zip(old_mimic.Tangent(frozen,current,evaluated).correction(),mimic.Tangent(frozen,current,evaluated).correction()):equal(a,b)
    repeated=SimpleNamespace(value=evaluated.value.repeat(1,2),jacobian=evaluated.jacobian.repeat(1,2,1,1),
        owners=evaluated.owners.repeat(2,1),enabled=evaluated.enabled.repeat(2),valid=evaluated.valid.repeat(1,2))
    for a,b in zip(old_mimic.Tangent(minimal,current,repeated).correction(),mimic.Tangent(minimal,current,repeated).correction()):equal(a,b)
    mimic.Tangent(scene,current);articulation.raw_modes(scene,current)
    calls=0;original_tolist=torch.Tensor.tolist
    def count(value):
        nonlocal calls
        calls+=1;return original_tolist(value)
    try:
        torch.Tensor.tolist=count
        mimic.Tangent(scene,current);articulation.raw_modes(scene,current)
    finally:torch.Tensor.tolist=original_tolist
    assert calls==0,calls
    with torch.inference_mode():unversioned=torch.tensor([1.])
    assert tensor_versions((unversioned,)) is None
    payload=args.checkpoint.read_bytes();checkpoint_sha=hashlib.sha256(payload).hexdigest()
    stored=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True);del payload
    config=stored['model_config'] if 'model_config' in stored else stored['config']['model']
    model=MixedSolver(config).to(args.device).eval();model.load_state_dict(stored['model'])
    results=[]
    from contextlib import nullcontext
    for reference in (True,False):
        scene.metadata['compact_numerical_diagnostics']=not reference
        model.zero_grad(set_to_none=True);set_controls(scene,states,720);query(scene,current,wp)
        with reference_math() if reference else nullcontext():
            predicted,diagnostic=model(scene,current,scene.metadata['dt'],**external_inputs(states,slice(720,721)))
            assert bool(diagnostic['accepted'].all())
            objective=loss(scene,predicted,select(states,slice(721,722)));objective.backward()
        results.append((predicted.x.detach().clone(),predicted.q.detach().clone(),float(objective.detach()),
            {name:p.grad.detach().clone() for name,p in model.named_parameters() if p.grad is not None}))
    for a,b in zip(results[0][:2],results[1][:2]):equal(a,b)
    assert results[0][3].keys()==results[1][3].keys()
    gradient_error=0.
    for name,a in results[0][3].items():
        b=results[1][3][name];gradient_error=max(gradient_error,float((a-b).abs().max()))
        assert torch.allclose(a,b,atol=1e-10,rtol=1e-6),name
    assert all(torch.equal(p.detach().cpu(),stored['model'][name]) for name,p in model.named_parameters())
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    reference_hashes={name:digest(args.baseline_root/'src/wp_vbd'/f'{name}.py') for name in ('mimic_constraint','articulation_coarse','metric_ray_step')}
    candidate_hashes={name:digest(ROOT/'src/wp_vbd'/f'{name}.py') for name in (*reference_hashes,'static_topology')}
    assert any(reference_hashes[key]!=candidate_hashes[key] for key in reference_hashes),'Reference must retain the original implementation'
    report={'status':'PASS','device':args.device,'recorded_native_states_checked':checked,'batch_two_checked':True,
        'reference_source_sha256':reference_hashes,'candidate_source_sha256':candidate_hashes,
        'checkpoint_sha256':checkpoint_sha,'checkpoint_step':stored['step'],
        'maximum_forward_absolute_difference':maximum,'maximum_parameter_gradient_absolute_difference':gradient_error,
        'parameter_gradient_tensors_compared':len(results[0][3]),'original_loss':results[0][2],'cached_loss':results[1][2],
        'invalidations_checked':invalidations,'explicit_rows_without_asset_metadata_checked':True,
        'zero_rank_scalar_and_repeated_rank_deficient_rows_checked':True,
        'cache_hit_tensor_tolist_calls':calls,'unversioned_inference_tensor_rebuild':True,
        'compact_numerical_diagnostics_forward_and_backward_checked':True,
        'full_vbd_calls':0,'featherstone_calls':0,'new_optimizer_updates':0,
        'scope':'Actual geometry, current-state constraint factors, neural forward and backward equivalence; this report alone does not establish CUDA performance.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
