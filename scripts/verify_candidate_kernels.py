"""Check candidate energy distribution, derivatives and material path guards."""
import argparse
import inspect
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--report',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    p.add_argument('--fast-inference',action='store_true',help='Exercise the actual scoped fast-inference settings')
    a=p.parse_args()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.franka_vbd_scene import build as build_tet
    from wp_vbd.franka_cloth_corner_scene import build as build_cloth
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd import surface_joint_system as physics,math3d as so3
    from wp_vbd.mixed_contact_system import path_feasible,_path_feasible_reference
    from wp_vbd.mixed_physics import predictor
    from wp_vbd.data import State
    from wp_vbd.compiled_contact import evaluate as compiled
    from wp_vbd import mimic_constraint as mimic
    from wp_vbd.inference_policy import configure,inference_settings
    torch.set_num_threads(1);torch.manual_seed(531);torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(a.device)
    if a.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    bound=inspect.signature(newton.solvers.SolverVBD).bind(None,
        integrate_with_external_rigid_solver=False,particle_enable_self_contact=False)
    bound.apply_defaults();cfg={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    def forbidden(*args,**kwargs):raise AssertionError('Native dynamics forbidden')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=forbidden;cls.step=forbidden
    results=[]
    for name,builder in [('tet',build_tet),('cloth',build_cloth)]:
        e=builder(wp,newton,device=a.device,initialize_controls=False)
        newton.eval_fk(e.model,e.model.joint_q,e.model.joint_qd,e.state_0)
        wp.copy(e.model.body_q,e.state_0.body_q)
        e.control=e.model.control()
        scene=export_model(e.model,cfg,'research.'+name,newton,1/180,a.device)
        scene.periodic_bending=True
        runtime=SurfaceJointExampleSolver(SimpleNamespace(config={}),scene,1/180,wp,newton,e.model,e.state_0)
        runtime.refresh_control(e.control)
        start=runtime.read(e.state_0);scene.candidates=runtime.oracle.query(start)
        with torch.no_grad():
            context=physics.begin(scene,start,scene.candidates,1/180)
            frozen=physics.rows(scene,start,start,context)
            tx,tq=predictor(scene,start,1/180)
        for perturb in (0.,.0002):
            delta=(perturb*torch.randn(len(start.x),scene.n,6,device=a.device,dtype=torch.float64)).requires_grad_()
            state=State(start.x+delta[...,:3],so3.normalize(so3.multiply(so3.exp(delta[...,3:]),start.q)),start.v,start.omega)
            joint=physics.joints.joint_energy(scene,state,context,frozen_bound=frozen['joint_rows']['bound_lower'])
            old=physics.merit(scene,state,start,tx,tq,context,frozen)
            new=physics._candidate_merit_tensor(scene,state,start,tx,tq,context,frozen,joint)
            assert torch.equal(old,new),'Energy owner distribution changed'
            go=torch.autograd.grad(old.sum(),delta,retain_graph=True)[0]
            gn=torch.autograd.grad(new.sum(),delta)[0]
            gradient_error=float((go-gn).abs().max())
            gradient_scale=float(go.abs().max())
            torch.testing.assert_close(go,gn,rtol=1e-12,atol=1e-10,
                msg=f'Potential derivative: max error={gradient_error}, scale={gradient_scale}')
            error=0.
            with torch.no_grad(),inference_settings(configure({},'fused_fast') if a.fast_inference else {}):
                value=state.detach()
                if a.device.startswith('cuda'):
                    joint=physics.joints.joint_energy(scene,value,context,frozen_bound=frozen['joint_rows']['bound_lower'])
                    fast=compiled(physics._candidate_merit_tensor,scene,value,start,tx,tq,context,frozen,joint)
                    error=float((old-fast).abs().max())
                    torch.testing.assert_close(fast,old,rtol=1e-11,atol=1e-10)
                scene.compiled_candidate_kernels=a.device.startswith('cuda')
                good=path_feasible(scene,start,value)
                assert torch.equal(good,_path_feasible_reference(scene,start,value))
                collapsed=value.clone();collapsed.x[:,:e.model.particle_count]=start.x[:,:e.model.particle_count].mean(1,keepdim=True)
                rejected=path_feasible(scene,start,collapsed)
                assert not bool(rejected.any())
                assert torch.equal(rejected,_path_feasible_reference(scene,start,collapsed))
                # Endpoints can be valid while the interpolated path collapses.
                turned=value.clone();center=start.x[:,:e.model.particle_count].mean(1,keepdim=True)
                turned.x[:,:e.model.particle_count,:2]=2*center[...,:2]-start.x[:,:e.model.particle_count,:2]
                turned.x[:,:e.model.particle_count,2]=start.x[:,:e.model.particle_count,2]
                assert bool(path_feasible(scene,turned,turned).all())
                assert not bool(path_feasible(scene,start,turned).any())
                assert torch.equal(path_feasible(scene,start,turned),_path_feasible_reference(scene,start,turned))
                scene.compiled_candidate_kernels=False
                topology=mimic.group_topology(scene,None,value.x.device)
                random=torch.randn(len(value.x),scene.n,6,6,device=a.device,dtype=torch.float64)
                positive=random@random.transpose(-1,-2)+torch.eye(6,device=a.device,dtype=torch.float64)
                pvalue=torch.randn_like(value.x).repeat_interleave(2,-1)
                dvalue=torch.randn_like(pvalue)
                for inverse in (None,positive):
                    eager=mimic.Tangent(scene,value,inverse_blocks=inverse)
                    primal=eager.primal(pvalue);dual=eager.dual(dvalue)
                    dot_a=(primal*dvalue).sum();dot_b=(pvalue*dual).sum()
                    torch.testing.assert_close(dot_a,dot_b,rtol=1e-11,atol=1e-10)
                    if a.device.startswith('cuda'):
                        scene.compiled_candidate_kernels=True
                        fast=mimic.Tangent(scene,value,inverse_blocks=inverse)
                        torch.testing.assert_close(fast.primal(pvalue),primal,rtol=1e-11,atol=1e-10)
                        torch.testing.assert_close(fast.dual(dvalue),dual,rtol=1e-11,atol=1e-10)
                        assert torch.equal(fast.valid,eager.valid)
                        scene.compiled_candidate_kernels=False
                reference,valid=mimic.retract(scene,value)
                x,q,good=mimic._retract_tensor(scene,value.x,value.q,topology,2)
                assert torch.equal(reference.x,x) and torch.equal(reference.q,q) and torch.equal(valid,good)
                if a.device.startswith('cuda'):
                    scene.compiled_candidate_kernels=True
                    accelerated,fast_valid=mimic.retract(scene,value)
                    torch.testing.assert_close(accelerated.x,reference.x,rtol=1e-11,atol=1e-10)
                    torch.testing.assert_close(accelerated.q,reference.q,rtol=1e-11,atol=1e-10)
                    assert torch.equal(fast_valid,valid)
                    scene.compiled_candidate_kernels=False
                if topology:
                    large=value.clone()
                    child=int(scene.joint_spec.owners[scene.mimic_spec.pairs[0,0],1])
                    large.x[:,child]+=large.x.new_tensor([1.,2.,3.])
                    reference,valid=mimic.retract(scene,large)
                    x,q,good=mimic._retract_tensor(scene,large.x,large.q,topology,2)
                    assert torch.equal(reference.x,x) and torch.equal(reference.q,q) and torch.equal(valid,good)
                    assert not bool(valid.any()),'Large mimic correction must remain invalid'
                    original_spec=scene.mimic_spec
                    scene.mimic_spec=SimpleNamespace(**{key:(tensor.repeat_interleave(2,0) if hasattr(tensor,'repeat_interleave') else tensor)
                        for key,tensor in vars(original_spec).items()})
                    duplicate_topology=mimic.group_topology(scene,None,value.x.device)
                    repeated,repeated_valid=mimic.retract(scene,value)
                    x,q,good=mimic._retract_tensor(scene,value.x,value.q,duplicate_topology,2)
                    torch.testing.assert_close(x,repeated.x,rtol=1e-12,atol=1e-10)
                    torch.testing.assert_close(q,repeated.q,rtol=1e-12,atol=1e-10)
                    assert torch.equal(good,repeated_valid)
                    if a.device.startswith('cuda'):
                        scene.compiled_candidate_kernels=True
                        fast,fast_valid=mimic.retract(scene,value)
                        torch.testing.assert_close(fast.x,repeated.x,rtol=1e-11,atol=1e-10)
                        torch.testing.assert_close(fast.q,repeated.q,rtol=1e-11,atol=1e-10)
                        assert torch.equal(fast_valid,repeated_valid)
                        scene.compiled_candidate_kernels=False
                    scene.mimic_spec=original_spec
            results.append(dict(material=name,perturbation_m=perturb,
                energy_distribution_exact=True,energy_gradient_max_error=gradient_error,
                energy_gradient_max_magnitude=gradient_scale,
                compiled_energy_max_abs_error=error,valid_and_collapsed_paths_match=True,
                two_pass_mimic_projection_and_large_motion_rejection_exact=True,
                valid_endpoints_with_interior_collapse_rejected=True,
                redundant_coupled_mimic_rows_checked=bool(topology)))
    result=dict(status='PASS',device=a.device,fast_inference=a.fast_inference,cases=results,native_dynamics_calls=0,optimizer_updates=0)
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
