"""Freeze directional architecture, data, selection caps and held-out material."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from wp_vbd.mixed_solver import MixedSolver

def main():
    parent=ROOT/'checkpoints/shared_inverse_v18/shared_strict.pt'
    directory=ROOT/'reports/shared_vector_v19';probe=ROOT/'checkpoints/shared_vector_v19/architecture_probe.pt'
    path=ROOT/'configs/shared_vector_v19.json'
    if path.exists() or probe.exists():raise FileExistsError('Keep experiment initialization immutable')
    stored=torch.load(parent,weights_only=True,map_location='cpu');cfg=copy.deepcopy(stored['config'])
    cfg.update(name='shared_vector_v19',seed=20261107,report_dir='reports/shared_vector_v19',checkpoint_dir='checkpoints/shared_vector_v19',initialize_from=str(probe))
    cfg['model']['protected_vector_ports']=True
    cfg['training'].update(steps=128,batch_seed=20261107,validation_selection={'maximum_relative_increase':.05,'absolute_floor':1e-4})
    cfg['scope'].update(training_curriculum='Shared encoder/context warm-started from v18; directional shapes and scalar gate zero initialized. No old rigid shape head retained.',
        protected_response='Six directional proposal/propagation/velocity columns, per-column inverse-metric scaling, rank mask and shared projected inverse gate.',
        input_normalization='Material energy divided by local mobile mass times L^2/dt^2; inverse conditioning already dimensionless because W=h M^-1/2. Inactive numerical padding excluded. Encoding covariance is separate from whole-solver covariance.',
        validation_selection='Each object endpoint/window score may increase at most 5% of max(initial score,1e-4); select balanced normalized mean only among admissible candidates. Fixed before optimizer updates.',
        physical_vector_control='Same six vector columns with unit shapes and prescribed near-unit metric gate, no NN calls. Compare separately from zero-gate physical baseline.',
        test_isolation='Material1.17 is development after v18. Newly frozen material1.33 is excluded from training/selection and model evaluation until weights are selected.',
        acceptance='Prototype experiment, not the default or all-demo accepted model.')
    torch.manual_seed(cfg['seed']);model=MixedSolver(cfg['model'])
    inherited={k:v for k,v in stored['model'].items() if not k.startswith(('protected_response.shape.','protected_response.factor.'))}
    missing=model.load_state_dict(inherited,strict=False)
    assert not missing.unexpected_keys
    assert set(missing.missing_keys)=={'protected_response.factor.weight','protected_response.factor.bias','protected_response.ports.shape.weight','protected_response.ports.shape.bias'}
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in inherited.items())
    directory.mkdir(parents=True,exist_ok=True);probe.parent.mkdir(parents=True,exist_ok=True)
    (directory/'.gitignore').write_text('*.log\n*.npz\n')
    manifest=json.loads((ROOT/'reports/shared_inverse_v18/data_manifest.json').read_text())
    manifest.update(architecture_experiment='Directional shared inverse, unchanged admitted data and splits',inherited_actual_training_demo_count=1,
        active_architecture_coverage='One official transfer demo plus five-type custom mixed fixtures. No new default-budget or Michell labels admitted.')
    (directory/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');path.write_text(json.dumps(cfg,indent=2)+'\n')
    provenance={'status':'UNTRAINED_DIRECTIONAL_GATE','parent_checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
        'active_parameters':sum(p.numel() for p in model.protected_response.parameters()),'new_optimizer_updates':0,'zero_gate':True,
        'unused_stored_legacy_parameters':109332,'new_official_demo_count':0,'active_official_training_demo_count':1}
    torch.save({'architecture':model.architecture_id,'model':model.state_dict(),'config':cfg,'step':0,'operator_probe_provenance':provenance},probe)
    provenance['checkpoint_sha256']=hashlib.sha256(probe.read_bytes()).hexdigest();(directory/'initialization.json').write_text(json.dumps(provenance,indent=2)+'\n')
    reference={'demo':'vbd.example_cable_twist_transfer','newton_sha':cfg['newton_sha'],'frames':390,'budgets':[128,256],
        'data_dir':'.local/vector_holdout_v19','report_dir':'reports/vector_holdout_v19','position_refinement_limit_over_L':.002,'orientation_refinement_limit_rad':.002,
        'episodes':[{'name':'fresh133','factor':1.33,'split':'test'}],'frozen_before_v19_training':True,
        'scope':'New material test, no new geometry claim. Material1.17 is development. Reference refinement does not admit labels into training.'}
    (ROOT/'configs/vector_holdout_v19.json').write_text(json.dumps(reference,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
