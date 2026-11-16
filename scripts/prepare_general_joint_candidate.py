"""Give finite-joint components the same shared coarse readout and audit it."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']
import torch
import warp as wp
import newton
import newton.viewer
from wp_vbd.mixed_solver import MixedSolver
from wp_vbd.mixed_experiment import load_sample,select,external_inputs,controlled_start,loss
from wp_vbd.rigid_contact_data import load_reference as load_contact,query
from wp_vbd.general_joint_data import load_reference as load_joint,set_controls


def main():
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    parent=ROOT/'checkpoints/shared_legacy_suite_v25/shared_strict.pt';stored=torch.load(parent,weights_only=True,map_location='cpu')
    config=copy.deepcopy(stored['config']);config['model']['constitutive_coarse_scope']='active_contact_or_finite_joint'
    before=MixedSolver(stored['config']['model']).eval();before.load_state_dict(stored['model'])
    model=MixedSolver(config['model']).eval();model.load_state_dict(stored['model'])
    assert all(torch.equal(v,model.state_dict()[name]) for name,v in before.state_dict().items())
    for m in (before,model):
        for name,value in m.named_parameters():value.requires_grad_(name.startswith('protected_response.coarse_response.'))
    mixed=load_sample(ROOT/'.local/coupling_v3/data/mixed_00.npz',device='cpu')
    rod=load_sample(ROOT/'.local/rod_history_v9/vbd.example_cable_bend_stiffness_train.npz',device='cpu')
    contact=load_contact(ROOT/'.local/rigid_contact_v21/domino_spiral_authored.npz',wp,newton,'cpu')
    joint,states=load_joint(ROOT/'.local/general_joint_v26/basic_joints_authored.npz',wp,newton,'cpu')
    def forbidden(*args,**kwargs):raise AssertionError('Full VBD forbidden in candidate check')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden
    results=[]
    with torch.no_grad():
        for label,(scene,data),index,collision in [('five_type_mixed',mixed,20,False),('rod_history',rod,100,False),('rigid_coupled_contact',contact,600,True)]:
            current=controlled_start(scene,select(data,slice(index,index+1)),data,index)
            if collision:query(scene,current,wp)
            inputs=external_inputs(data,slice(index,index+1));a,ad=before(scene,current,scene.metadata['dt'],**inputs);b,bd=model(scene,current,scene.metadata['dt'],**inputs)
            assert bool(ad['accepted'].all()) and bool(bd['accepted'].all())
            assert all(torch.equal(getattr(a,k),getattr(b,k)) for k in ('x','q','v','omega'))
            if a.history:assert all(torch.equal(a.history[k],b.history[k]) for k in a.history)
            results.append({'case':label,'state_and_history_bitexact':True})
    gradients={};predictions={};index=400
    for name,m in [('contact_only_scope',before),('finite_joint_scope',model)]:
        m.zero_grad(set_to_none=True);current=select(states,slice(index,index+1));objective=current.x.new_zeros(())
        for offset in range(2):
            set_controls(joint,states,index+offset);query(joint,current,wp)
            current,d=m(joint,current,joint.metadata['dt'],**external_inputs(states,slice(index+offset,index+offset+1)))
            assert bool(d['accepted'].all())
            objective=objective+loss(joint,current,select(states,slice(index+offset+1,index+offset+2)))/2
        objective.backward();gradient=torch.cat([p.grad.flatten() for p in m.protected_response.coarse_response.parameters()]);assert bool(torch.isfinite(gradient).all())
        gradients[name]=float(gradient.double().norm());predictions[name]=current.detach()
    assert gradients['contact_only_scope']==0 and gradients['finite_joint_scope']>0,gradients
    start=select(states,slice(index,index+1));saved=start.clone();set_controls(joint,states,index);query(joint,start,wp)
    original_motion=joint.geometry_oracle.motion
    def reject(a,b):
        valid,travel=original_motion(a,b);return torch.zeros_like(valid),travel
    joint.geometry_oracle.motion=reject
    with torch.no_grad():rejected,d=model(joint,start,joint.metadata['dt'],**external_inputs(states,slice(index,index+1)))
    assert not bool(d['accepted'].any())
    assert all(torch.equal(getattr(rejected,k),getattr(start,k)) and torch.equal(getattr(start,k),getattr(saved,k)) for k in ('x','q','v','omega'))
    assert all(torch.equal(rejected.history[k],start.history[k]) for k in start.history)
    directory=ROOT/'checkpoints/shared_joints_v26';directory.mkdir(parents=True,exist_ok=True);target=directory/'architecture_probe.pt'
    if target.exists():raise FileExistsError('Candidate initialization is immutable')
    torch.save({**stored,'config':config,'model':model.state_dict(),'parent_checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'new_optimizer_updates':0,'operator_probe_provenance':'Same shared head extended by finite-joint physical contract, no new neural weights or training.'},target)
    report={'status':'ARCHITECTURE_AND_REPLAY_PROBE_PASS','checkpoint':str(target.relative_to(ROOT)),'checkpoint_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'parent_checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'new_neural_parameters':0,'new_optimizer_updates':0,'inherited_actual_training_demos':14,'new_joint_training_admissions':0,'old_replay_cases':results,'real_teacher_two_step_joint_head_gradient_norm':gradients,'full_state_and_winding_rejection_rollback_bitexact':True,'full_vbd_calls':0,'scope':'Contract-gated extension of the same 33-parameter shared coarse head; old physical contracts use identical activation. New joint has a real teacher-loss gradient but has not yet entered an optimizer or frozen trained checkpoint.'}
    out=ROOT/'reports/shared_joints_v26';out.mkdir(parents=True,exist_ok=True);(out/'.gitignore').write_text('*.log\n*.npz\n');(out/'architecture_initial.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
