"""Verify material directions, causal teacher windows and adaptive transactions on CPU."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    import torch
    from wp_vbd import math3d as so3
    from wp_vbd.adaptive_step import AdaptiveStepper,AdaptiveStepConfig,AdaptiveStepFailure
    from wp_vbd.data import State
    from wp_vbd.elastic_shape_modes import ElasticShapeDirectionHead,material_basis
    from wp_vbd.multirate_window import ReferenceGrid,reference_grid,long_step_window,command_events,inputs_for_span

    torch.set_num_threads(1);torch.manual_seed(327)
    with tempfile.TemporaryDirectory() as temporary:
        directory=Path(temporary)
        path=directory/'teacher.npz'
        count=121
        command=np.zeros((count-1,1));command[4:]=1
        arrays={'x':np.zeros((count,8,3)), 'input_joint_effort':command,
                'metadata':np.asarray(json.dumps({'dt':1/300}))}
        np.savez_compressed(path,**arrays)
        grid,_=reference_grid(path)
        assert command_events(path,count)==[4]
        assert command_events(path,count,hard_only=True)==[]
        plan=long_step_window(grid,0,[1/60,1/30],events=[4])
        assert plan['strides']==[4,1,10] and plan['extra_event_steps']==1
        np.testing.assert_allclose(plan['durations'],[4/300,1/300,1/30])
        assert long_step_window(grid,0,[1/60,1/30])['strides']==[5,10]
        caller=directory/'caller.npz'
        np.savez_compressed(caller,**arrays,caller__steps_per_sample=np.asarray(6))
        caller_events=command_events(caller,count)
        assert 6 in caller_events and 12 in caller_events
        assert long_step_window(grid,0,[1/30],events=caller_events)['strides']==[4,2,4]
        coarse=ReferenceGrid(np.arange(20)/100,((0,19),))
        nearest=long_step_window(coarse,0,[1/60])
        assert nearest['endpoint_indices']==[2] and nearest['durations']==[.02]
        try:long_step_window(coarse,0,[1/60],allow_nearest=False)
        except ValueError:pass
        else:raise AssertionError('Unaligned endpoints must not become interpolated teachers')
        segmented=ReferenceGrid(np.arange(20)/300,((0,5),(10,19)))
        try:long_step_window(segmented,4,[1/60])
        except ValueError:pass
        else:raise AssertionError('A long window crossed a reference discontinuity')
        irregular=directory/'irregular.npz'
        np.savez_compressed(irregular,x=np.zeros((4,8,3)),physical_time=[0.,.001,.004,.010],metadata=json.dumps({}))
        irregular_grid,_=reference_grid(irregular)
        np.testing.assert_allclose(long_step_window(irregular_grid,0,[.004])['teacher_durations'],[.001,.003])
        actions={'input_force':torch.tensor([[[1.]],[[3.]]]),
                 'input_joint_effort':torch.ones(2,1)}
        integrated=inputs_for_span(actions,0,2,durations=[.001,.003])
        torch.testing.assert_close(integrated['input_force'],torch.tensor([[[2.5]]]))
        drive=inputs_for_span({'input_joint_effort':torch.tensor([[1.],[2.]])},0,2)
        torch.testing.assert_close(drive['input_joint_effort'],torch.tensor([[1.5]]))
        sixty=ReferenceGrid(np.arange(5)/60,((0,4),))
        macro=long_step_window(sixty,0,[1/30])
        assert macro['strides']==[2] and macro['endpoint_indices']==[2]
        moving=inputs_for_span({'input_pin_x':torch.tensor([[[.01,0.,0.]],[[.02,0.,0.]]])},0,2)
        torch.testing.assert_close(moving['input_pin_x'],torch.tensor([[[.02,0.,0.]]]))
        manifest={'newton_sha':'fixture','scenes':[{
            'dataset_id':'timing_fixture','loader':'legacy','path':str(path),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'qualified_train_starts':[0,5,10],'validation_starts':[80]}]}
        source=directory/'source.json';source.write_text(json.dumps(manifest))
        subprocess.run([sys.executable,str(ROOT/'scripts/prepare_multirate_data.py'),
                        '--manifest',str(source),'--output',str(directory/'indexed')],check=True,capture_output=True,text=True)
        indexed=json.loads((directory/'indexed/data_manifest.json').read_text())
        assert indexed['status']=='INDEXED' and len(indexed['scenes'])==1
        splits=indexed['scenes'][0]['multirate']
        for train in splits['training']['windows']:
            for dev in splits['development']['windows']:
                assert train['endpoint_indices'][-1]<dev['start'] or dev['endpoint_indices'][-1]<train['start']

    # Failed proposals deliberately mutate their private history before rejection.
    state=State(torch.zeros(1,2,3,dtype=torch.float64),
                torch.tensor([0.,0.,0.,1.],dtype=torch.float64).expand(1,2,4).clone(),
                torch.zeros(1,2,3,dtype=torch.float64),torch.zeros(1,2,3,dtype=torch.float64),
                {'elapsed':torch.zeros(1,dtype=torch.float64)})
    def trial(current,h,time):
        current.history['elapsed'].add_(h)
        current.x.add_(h)
        return current,{'accepted':h<=.012,'iterations':[{}]*3}
    stepper=AdaptiveStepper(AdaptiveStepConfig())
    current=state
    clock=0.
    schedules=[]
    for duration in (1/60,1/30,1/300):
        current,report=stepper.advance(current,duration,trial,start_time=clock,events=(clock+.007,))
        clock+=duration
        torch.testing.assert_close(current.x,torch.full_like(current.x,clock))
        torch.testing.assert_close(current.history['elapsed'],torch.tensor([clock],dtype=torch.float64))
        assert abs(sum(r['dt'] for r in report['attempts'] if r['accepted'])-duration)<1e-12
        schedules.append({k:v for k,v in report.items() if k not in ('diagnostics','attempts')})
    assert not bool(state.x.any()) and not bool(state.history['elapsed'].any())
    def reject(current,h,time):
        current.history['elapsed'].add_(100)
        return current,{'accepted':False}
    try:stepper.advance(state,1/60,reject)
    except AdaptiveStepFailure:pass
    else:raise AssertionError('An entirely rejected interval committed')
    assert not bool(state.history['elapsed'].any())
    def depth(current,h,time):
        return current,{'accepted':True,'minimum_contact_normal_physical_separation':-.1}
    try:AdaptiveStepper(AdaptiveStepConfig(maximum_contact_depth=.01)).advance(state,1/60,depth)
    except AdaptiveStepFailure:pass
    else:raise AssertionError('Contact depth bound ignored scalar physical diagnostics')

    n=37
    points=torch.randn(1,n,3,dtype=torch.float64)
    mass=torch.rand(n,dtype=torch.float64)+.5
    scene=SimpleNamespace(n=n,kind=torch.ones(n,dtype=torch.long),movable=torch.ones(n,dtype=torch.bool),
                          object_id=torch.zeros(n,dtype=torch.long),mass=mass,length=1.,rest_x=points[0])
    head=ElasticShapeDirectionHead(feature_version=2)
    # Exercise trained coefficients; the zero initialization could hide a frame bug.
    torch.nn.init.normal_(head.network[-1].weight,std=.03)
    residual=torch.randn(1,n,6,dtype=torch.float64)
    proposal=head.propose(scene,SimpleNamespace(x=points),residual,1/30)
    assert float(proposal[...,:3].square().sum(-1).mean(-1).sqrt().max())<=head.maximum_fraction+1e-12
    assert not bool(proposal[...,3:].any())
    rotation=so3.matrix(so3.exp(torch.tensor([[.3,-.4,.2]],dtype=torch.float64)))[0]
    rotated_residual=residual.clone();rotated_residual[...,:3]=residual[...,:3]@rotation.T
    rotated=head.propose(scene,SimpleNamespace(x=points@rotation.T+1.),rotated_residual,1/30)
    covariance_error=float((rotated[...,:3]-proposal[...,:3]@rotation.T).abs().max().detach())
    assert covariance_error<1e-8,covariance_error
    center=(points*mass[None,:,None]).sum(1,keepdim=True)/mass.sum()
    net=(proposal[...,:3]*mass[None,:,None]).sum(1)
    torque=(torch.linalg.cross(points-center,proposal[...,:3])*mass[None,:,None]).sum(1)
    assert float(net.abs().max().detach())<1e-10 and float(torque.abs().max().detach())<1e-10
    proposal.square().sum().backward()
    gradient=float(head.network[-1].weight.grad.norm())
    assert gradient>0 and all(bool(torch.isfinite(p.grad).all()) for p in head.parameters() if p.grad is not None)
    # Rigid modes remain removed after arbitrary material-coordinate combinations.
    basis=material_basis(points,points[0],mass)
    assert basis.shape==(1,n,3,24)
    from wp_vbd.protected_response import protected_update
    from wp_vbd.coarse_response import rigid_component_modes
    scene.device=points.device
    scene.dof=torch.zeros(n,6,dtype=torch.bool);scene.dof[:,:3]=True
    scene.metadata={};scene.interaction_entities=[torch.arange(n)]
    diagonal=torch.rand(1,n,6,dtype=torch.float64)+1
    class Metric:
        terms=[]
        local=torch.diag_embed(diagonal)
        def __call__(self,value):return value*diagonal[...,None]
    metric=Metric()
    rigid=rigid_component_modes(points,center,scene.dof)
    coarse=SimpleNamespace(blocks=[(SimpleNamespace(ids=torch.arange(n)),rigid)])
    class LocalResponse:
        def __call__(self,scene,state,complement,*unused):
            return -.01*complement,torch.ones(1,dtype=torch.bool)
    model=SimpleNamespace(config={},protected_response=LocalResponse(),elastic_shape_head=head,training=True)
    identity=torch.eye(6,dtype=torch.float64).expand(1,n,6,6)
    residual=residual*scene.dof
    head.zero_grad(set_to_none=True)
    update,_,valid=protected_update(model,scene,SimpleNamespace(x=points),metric,coarse,
                                     residual,identity,identity,None,None,1/30,True)
    assert model._elastic_shape_supervision_terms
    direction_loss=torch.stack(model._elastic_shape_supervision_terms).mean()
    head.zero_grad(set_to_none=True);direction_loss.backward(retain_graph=True)
    distillation_gradient=float(head.network[-1].weight.grad.norm())
    assert bool(torch.isfinite(direction_loss)) and distillation_gradient>1e-10
    # Full modal teacher work belongs to training only. A deployed update
    # must be identical without evaluating the teacher or its reduced solve.
    from unittest.mock import patch
    model.training=False
    with torch.no_grad(),patch('wp_vbd.elastic_shape_objective.projected_modal_systems',
                              side_effect=AssertionError('Teacher called during inference')):
        inference,_,_=protected_update(model,scene,SimpleNamespace(x=points),metric,coarse,
                                         residual,identity,identity,None,None,1/30,True)
    torch.testing.assert_close(inference,update.detach())
    model.elastic_shape_head=None
    baseline,_,_=protected_update(model,scene,SimpleNamespace(x=points),metric,coarse,
                                  residual,identity,identity,None,None,1/30,True)
    def energy(value):return (residual*value+.5*diagonal*value.square()).sum()
    assert bool(valid.all()) and energy(update)<=energy(baseline)+1e-10
    local_head=ElasticShapeDirectionHead(feature_version=2,local_heads=2)
    torch.nn.init.normal_(local_head.network[-1].weight,std=.03)
    proposal,local_details,columns=local_head.propose(scene,SimpleNamespace(x=points),residual,1/30,return_details=True)
    assert columns.shape==(1,n,6,3) and local_details[0]['basis'].shape==(1,n,3,72)
    local_model=SimpleNamespace(config={},protected_response=LocalResponse(),elastic_shape_head=local_head,training=True)
    local_update,_,local_valid=protected_update(local_model,scene,SimpleNamespace(x=points),metric,coarse,
        residual,identity,identity,None,None,1/30,True)
    assert bool(local_valid.all()) and energy(local_update)<=energy(baseline)+1e-10
    local_loss=torch.stack(local_model._elastic_shape_supervision_terms).mean()
    local_loss.backward()
    assert local_head.local_network[-1].weight.grad is not None
    assert float(local_head.local_network[-1].weight.grad.norm())>1e-10
    update.square().sum().backward()
    protected_gradient=float(head.network[-1].weight.grad.norm())
    assert protected_gradient>0 and torch.isfinite(head.network[-1].weight.grad).all()
    with torch.no_grad():
        # Legacy state tensors are float32 while physical residuals are float64.
        mixed_precision=head.propose(scene,SimpleNamespace(x=points.float()),residual,1/30)
        assert mixed_precision.dtype==torch.float64 and torch.isfinite(mixed_precision).all()
        large_n=5003
        large_points=torch.randn(1,large_n,3,dtype=torch.float64)
        large_scene=SimpleNamespace(n=large_n,kind=torch.ones(large_n,dtype=torch.long),
            movable=torch.ones(large_n,dtype=torch.bool),object_id=torch.zeros(large_n,dtype=torch.long),
            mass=torch.ones(large_n,dtype=torch.float64),length=1.,rest_x=large_points[0])
        large_proposal=head.propose(large_scene,SimpleNamespace(x=large_points),
                                    torch.randn(1,large_n,6,dtype=torch.float64),1/30)
        assert torch.isfinite(large_proposal).all()
    result={'status':'PASS','original_grid_300_hz_and_irregular_timestamps':True,
            'unaligned_endpoints_no_interpolation':True,'discrete_events_split':True,'smooth_controls_do_not_split_macro_step':True,
            'wrenches_integrated_by_duration':True,'disjoint_train_development_windows':True,
            'adaptive_retry_history_rollback':True,'adaptive_requested_intervals':schedules,
            'learned_direction_rigid_mass_orthogonal':True,'direction_rotation_error':covariance_error,
            'direction_gradient_norm':gradient,'optimizer_updates':0,'native_dynamics_calls':0}
    result.update(protected_direction_gradient_norm=protected_gradient,
                  projected_direction_does_not_increase_quadratic_energy=True,
                  physical_direction_energy_loss=float(direction_loss.detach()),
                  physical_direction_energy_gradient_norm=distillation_gradient,
                  inference_skips_full_modal_teacher=True,
                  learned_local_elastic_columns=2,local_elastic_columns=3,
                  local_elastic_loss=float(local_loss.detach()),
                  local_elastic_gradient_norm=float(local_head.local_network[-1].weight.grad.norm()),
                  legacy_mixed_precision_and_large_elastic_objects=True)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
