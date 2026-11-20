"""Check metric-matched mimic projections on actual Franka contact states."""
import argparse
import functools
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    a=parser.parse_args()
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.multirate_data import load_reference
    from wp_vbd.mixed_experiment import select
    from wp_vbd.mixed_contact_data import query
    from wp_vbd.general_joint_data import set_controls
    from wp_vbd.multirate_window import inputs_for_span
    from wp_vbd.mimic_constraint import Tangent
    from wp_vbd.mixed_physics import predictor,mass_whitening
    from wp_vbd import surface_joint_system as physics
    torch.set_num_threads(1);torch.manual_seed(20261119)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    def block(original):
        @functools.wraps(original)
        def forbidden(*args,**kwargs):raise AssertionError('No native dynamics in projector verification')
        return forbidden
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=block(cls.__init__);cls.step=block(cls.step)
    row={'loader':'surface','path':str(ROOT/'.local/multirate_20261118_v2/data/franka_hold_release_v1.npz'),
         'detection_motion_budget':.012}
    scene,states=load_reference(row,wp,newton,'cpu');tests=[]
    for time in (2.1,3.3):
        native=scene.metadata['dt'];dt=3*native
        index=round((time-scene.metadata['source_run']['initial_command_time'])/native)
        state=select(states,slice(index,index+1)).clone()
        actions=inputs_for_span(states,index,3,durations=[native]*3)
        set_controls(scene,actions,0);query(scene,state,wp)
        scene.metadata.update(surface_tangent_memory=True,surface_tangent_history_capacity=8192)
        tx,tq=predictor(scene,state,dt);ctx=physics.begin(scene,state,scene.candidates,dt)
        residual,metric,_=physics.system(scene,state,state,tx,tq,ctx)
        white=mass_whitening(scene,state,dt)
        matrix=white@metric.diagonal@white+torch.diag_embed((~scene.dof).double())[None]+1e-6*torch.eye(6)
        inverse=white@torch.linalg.inv(matrix)@white
        op=Tangent(scene,state,inverse_blocks=inverse)
        assert bool(op.valid.all())
        u=torch.randn_like(residual)*scene.dof;v=torch.randn_like(u)*scene.dof
        projected=op.primal(u)
        transpose_error=abs(float((projected*v).sum()-(u*op.dual(v)).sum()))
        idempotence_error=float((op.primal(projected)-projected).abs().max())
        assert transpose_error<1e-10 and idempotence_error<1e-10
        constraint_error=0.;before=0.;after=0.
        for _,ids,jac,*_ in op.groups:
            constraint_error=max(constraint_error,float(torch.einsum('brni,bni->br',jac,projected[:,ids]).abs().max()))
            energy=torch.linalg.inv(inverse[:,ids])
            before+=float(torch.einsum('bni,bnij,bnj->',u[:,ids],energy,u[:,ids]))
            after+=float(torch.einsum('bni,bnij,bnj->',projected[:,ids],energy,projected[:,ids]))
        assert constraint_error<1e-9 and after<=before*(1+1e-12)
        def response(value):return op.primal((inverse@op.dual(value)[...,None])[...,0])
        ru,rv=response(u),response(v)
        symmetry_error=abs(float((u*rv).sum()-(v*ru).sum()))
        assert symmetry_error<1e-10 and float((u*ru).sum())>=0
        source=residual.detach().requires_grad_();result=response(source)
        result.square().sum().backward()
        assert bool(torch.isfinite(source.grad).all()) and float(source.grad.norm())>0
        count=scene.candidates.rigid_count;soft=scene.candidates.soft
        final=torch.ones_like(ctx.dual_t[:,count:])*.01
        # Self-contact contracts append their own multipliers after the
        # surface rows; they must never be stored in the surface cache.
        ctx.dual_t=torch.cat((ctx.dual_t[:,:count],final,torch.ones(1,2,3)),1)
        ctx.soft.dual_t=torch.zeros_like(final)
        committed=physics.finish(scene,state,state,ctx,torch.ones(1,dtype=torch.bool))
        expected=final-(final*soft.normal).sum(-1,keepdim=True)*soft.normal
        cache=committed.history['surface_contact_tangent'][:,:len(soft.shape)]
        assert torch.equal(cache,expected) and bool(cache.any())
        tests.append({'time':time,'constraint_error':constraint_error,
            'transpose_error':transpose_error,'idempotence_error':idempotence_error,
            'local_energy_before':before,'local_energy_after':after,'response_symmetry_error':symmetry_error,
            'residual_backward_norm':float(source.grad.norm())})
    report={'status':'PASS','cases':tests,'native_dynamics_calls':0,'optimizer_updates':0,
        'new_neural_parameters':0,'scope':'Actual frozen contact/drive inverse blocks; primal/dual transpose, feasible tangent, idempotence, local-energy contraction, SPD response and residual backward.'}
    report['committed_surface_dual_is_last_accepted_dual']=True
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')
    print(a.report.read_text())


if __name__=='__main__':main()
