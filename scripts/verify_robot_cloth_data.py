"""Verify source reconstruction, causal-input storage and shape-loss gradients."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.robot_cloth_data import load_reference
    from wp_vbd.multirate_data import load_reference as dispatch
    from wp_vbd.mixed_experiment import select
    from wp_vbd.cloth_rollout_objective import ClothRolloutObjective
    from wp_vbd.multirate_window import inputs_for_span,stride_pattern
    torch.set_num_threads(1);wp.init();wp.set_device('cpu')
    def fail(*args,**kwargs):raise AssertionError('Native solver forbidden in source reconstruction')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=fail;cls.step=fail
    scene,states=load_reference(a.dataset,wp,newton,'cpu')
    _,other=dispatch({'loader':'robot_cloth','path':str(a.dataset)},wp,newton,'cpu')
    assert all(torch.equal(v,other[k]) for k,v in states.items())
    assert stride_pattern(1/720,3)==(4,4,4)
    assert len(states['x'])==2017
    assert not len(scene.fixed_particle_ids)
    for start in (0,4,8,12,1200):
        result=inputs_for_span(states,start,4)
        assert torch.equal(result['input_joint_target'],states['input_joint_target'][start:start+1])
    target=select(states,slice(1200,1201));obj=ClothRolloutObjective(scene)
    identical,_,_=obj(target,target,{})
    assert abs(float(identical))<1e-10
    perturbed=target.clone();perturbed.x=perturbed.x.clone()
    ids=torch.arange(scene.metadata['particle_count'])
    perturbed.x[:,ids,2]+=.01*torch.sin(ids*.7)
    perturbed.x.requires_grad_()
    value,terms,metrics=obj(perturbed,target,{})
    assert float(terms['drape_normals'])>0 and float(terms['drape_nonlocal'])>0
    value.backward()
    assert torch.isfinite(perturbed.x.grad).all() and float(perturbed.x.grad.norm())>0
    report={'status':'PASS','teacher_states':len(states['x']),'native_solver_calls':0,
            'frozen_source_reconstruction':True,'loader_dispatch':True,
            'aligned_teacher_stride_per_student_step':4,'student_steps_per_frame':3,
            'no_cloth_pins':True,'target_identical_loss':float(identical),
            'shape_perturbation_loss':float(value.detach()),'finite_nonzero_shape_gradient':True}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
