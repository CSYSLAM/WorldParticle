"""Migrate the shared checkpoint with a zero-initialized coarse head and verify it."""
from dataclasses import fields,replace
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import torch
from wp_vbd.mixed_solver import MixedSolver
from wp_vbd.mixed_experiment import load_sample,select,controlled_start,external_inputs
from wp_vbd.protected_response import ProtectedCoarse
from wp_vbd.rigid_contact_system import begin,system,coarse
from verify_rigid_contact_integration import fixture


def main():
    torch.set_num_threads(1)
    parent=ROOT/'checkpoints/shared_hierarchy_v20/shared_strict.pt';stored=torch.load(parent,weights_only=True,map_location='cpu')
    old=MixedSolver(stored['config']['model']).eval();old.load_state_dict(stored['model'])
    cfg=copy.deepcopy(stored['config']);cfg['model']['shared_constitutive_coarse']=True;cfg['model']['constitutive_coarse_scope']='active_contact';cfg['name']='shared_constitutive_v21'
    cfg['report_dir']='reports/shared_constitutive_v21';cfg['checkpoint_dir']='checkpoints/shared_constitutive_v21'
    model=MixedSolver(cfg['model']).eval();missing=model.load_state_dict(stored['model'],strict=False)
    expected=['protected_response.coarse_response.readout.weight','protected_response.coarse_response.readout.bias']
    assert sorted(missing.missing_keys)==sorted(expected) and not missing.unexpected_keys,missing
    assert all(torch.equal(model.state_dict()[k],v) for k,v in stored['model'].items())
    checks=[]
    for name in ('.local/coupling_v3/data/mixed_08.npz','.local/shared_multitype_reference_v20/softbody_train_512.npz','.local/shared_multitype_reference_v20/cloth_train_2048.npz','.local/shared_multitype_reference_v20/rigid_train_8.npz','.local/refined_analytic_v16/analytic_cpu256_full.npz'):
        scene,states=load_sample(ROOT/name,'cpu');current=controlled_start(scene,select(states,slice(60,61)),states,60)
        with torch.no_grad():
            before,bd=old(scene,current,scene.metadata['dt'],**external_inputs(states,slice(60,61)))
            after,ad=model(scene,current,scene.metadata['dt'],**external_inputs(states,slice(60,61)))
        assert bool(bd['accepted'].all()) and bool(ad['accepted'].all()),name
        for key in ('x','q','v','omega'):assert torch.equal(getattr(before,key),getattr(after,key)),(name,key)
        if before.history is not None:
            assert all(torch.equal(before.history[k],after.history[k]) for k in before.history)
        checks.append({'dataset':name,'zero_head_bitexact_state_and_history':True})
    scene,start,c=fixture();subset={f.name:getattr(c,f.name)[1:] for f in fields(c) if torch.is_tensor(getattr(c,f.name))}
    c=replace(c,**subset,source_count=2,capacity=2)
    def motion(before,after):
        travel=(after.x-before.x).norm(dim=-1)+scene.body_radius*torch.zeros_like(after.x[...,0])
        return travel.amax(-1)<.05,travel
    from types import SimpleNamespace
    scene.geometry_oracle=SimpleNamespace(motion=motion);scene.candidates=c
    model.zero_grad(set_to_none=True)
    result,diagnostic=model(scene,start,1/300)
    assert bool(diagnostic['accepted'].all())
    (result.x.square().sum()+result.omega.square().sum()/300**2).backward()
    head=model.protected_response.coarse_response
    gradient=sum(float(p.grad.double().square().sum()) for p in head.parameters())**.5
    assert gradient>1e-12 and all(bool(torch.isfinite(p.grad).all()) for p in head.parameters())
    # Inspect the actual frozen contact metric, rather than assuming a random
    # dense SPD matrix has the required physical coarse projection.
    context=begin(scene,start,c,1/300);r,metric,contact=system(scene,start,start,start.x,start.q,context)
    scene.update_graph(c.owners,contact['active'][0]);physical=ProtectedCoarse(scene,metric,coarse(scene,start).blocks)
    n=scene.world*6;basis=torch.eye(scene.n*6,dtype=torch.float64).reshape(1,scene.n,6,scene.n*6)
    h=metric(basis).reshape(scene.n*6,scene.n*6)[:n,:n]
    cm=physical(basis).reshape(scene.n*6,scene.n*6)[:n,:n]
    b=torch.linalg.inv(metric.diagonal[0,:scene.world]).reshape(scene.world,6,6)
    b=torch.block_diag(*b);identity=torch.eye(n,dtype=h.dtype);eigenvalues=[]
    for gain in (.5001,1.,1.4999):
        operator=gain*cm+(identity-cm@h)@b@(identity-h@cm)
        torch.testing.assert_close(operator,operator.T,rtol=1e-12,atol=1e-12)
        eigenvalues.append(float(torch.linalg.eigvalsh(operator)[0]));assert eigenvalues[-1]>0
        assert torch.count_nonzero(operator@torch.zeros(n,dtype=h.dtype))==0
    directory=ROOT/cfg['checkpoint_dir'];directory.mkdir(parents=True,exist_ok=True)
    checkpoint=directory/'architecture_probe.pt'
    if checkpoint.exists():raise FileExistsError('Keep migration candidates immutable')
    torch.save({'architecture':model.architecture_id,'config':cfg,'model':model.state_dict(),'step':0,'parent_checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'parent_selected_step':stored['step'],'new_optimizer_updates':0},checkpoint)
    report={'status':'PASS','checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'parent_checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),'new_parameters':sum(p.numel() for p in head.parameters()),'new_parameter_names':expected,'inherited_parameter_tensors_bitexact':True,'single_body_world_contact_head_gradient_norm':gradient,'five_actual_topology_zero_head_checks':checks,'positive_coarse_gain_minimum_inverse_eigenvalues':eigenvalues,'new_training_admissions':0,'coarse_scope':'Only active rigid, particle-shape or self-contact relations (kinds 7:10); never an object/demo ID mask. Legacy contracts have gain exactly one for arbitrary head weights.','scope':'Architecture migration and CPU admission; no new training, trajectory improvement or runtime speedup claimed.'}
    out=ROOT/cfg['report_dir'];out.mkdir(parents=True,exist_ok=True);(out/'.gitignore').write_text('*.log\n*.npz\n')
    (out/'architecture_initial.json').write_text(json.dumps(report,indent=2)+'\n');(ROOT/'configs/shared_constitutive_candidate.json').write_text(json.dumps(cfg,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
