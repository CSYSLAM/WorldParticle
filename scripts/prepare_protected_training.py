"""Create the shared protected-factor prototype without overwriting old evidence."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from wp_vbd.mixed_solver import MixedSolver


def main():
    directory=ROOT/'reports/shared_protected_v17'
    checkpoint=ROOT/'checkpoints/shared_protected_v17/architecture_probe.pt'
    config_path=ROOT/'configs/shared_protected_v17.json'
    if checkpoint.exists() or config_path.exists():raise FileExistsError('Keep experiment initialization immutable')
    parent=ROOT/'checkpoints/shared_refined_v16/shared_strict.pt'
    stored=torch.load(parent,weights_only=True,map_location='cpu')
    config=copy.deepcopy(stored['config'])
    config.update(name='shared_protected_v17',seed=20261105,report_dir='reports/shared_protected_v17',checkpoint_dir='checkpoints/shared_protected_v17',initialize_from=str(checkpoint))
    config['model'].update(protected_learning=True,protected_width=32,protected_factor_bound=.4)
    config['training'].update(steps=128,learning_rate=3e-4,validation_interval=64,trainable_prefixes=['protected_response.'],batch_seed=20261105)
    config['training']['validation_windows_by_demo']={'vbd.example_cable_twist_transfer':{'starts':[0,1080,2280,3780],'steps':120}}
    config['scope'].update(training_curriculum='New scalar graph encoder/shape/factor shared by every type. Old response/coarse-gain weights remain stored and frozen, and are unused by this architecture.',
        protected_response='Unlearned energy-orthogonal physical projection plus a bounded learned factor in its complement. Zero factor head exactly equals the matching new physical baseline.',
        validation_scope='Four predeclared 0.2s transfer windows cover startup, ramp end, hold and terminal phases; 12 single-step checks span the full validation episode. Full autoregressive tasks and held-out material are separate.',
        backward='Frozen physical H and physical coarse Q; scalar encoder and factor gradients are retained. Not an exact full-physics derivative.',
        teacher_target='Same refined transfer and legacy replay manifest as v16; no default-budget rod labels added.',
        factor_budget='6x6 local factors; six learned columns per component; at most 12 physical columns normalized with two scalar Gram-Schmidt passes. No global dynamics solve.')
    torch.manual_seed(config['seed'])
    model=MixedSolver(config['model'])
    incompatible=model.load_state_dict(stored['model'],strict=False)
    assert incompatible.missing_keys and all(name.startswith('protected_response.') for name in incompatible.missing_keys)
    assert not incompatible.unexpected_keys
    assert all(torch.equal(value,model.state_dict()[name]) for name,value in stored['model'].items())
    directory.mkdir(parents=True,exist_ok=True);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    (directory/'.gitignore').write_text('*.log\n*.npz\n')
    manifest=json.loads((ROOT/'reports/shared_refined_v16/data_manifest.json').read_text())
    manifest['architecture_experiment']='Shared protected-factor response; data/splits unchanged from v16'
    manifest['historical_original_demo_training_count']=10
    manifest['inherited_actual_training_demo_count']=0
    manifest['active_architecture_coverage']='New shared neural module; stored legacy weights are unused. This corpus contains one official demo plus five-type custom mixed fixtures.'
    config['scope']['official_coverage']='New active neural architecture must be requalified. This corpus trains one official demo plus mixed fixtures; historical 10-demo neural training is not inherited.'
    (directory/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    provenance={'status':'UNTRAINED_NEW_NEURAL_ARCHITECTURE','parent_checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
        'parent_selected_update':stored['step'],'new_network_parameters':sum(p.numel() for p in model.protected_response.parameters()),
        'old_neural_parameters_stored_but_unused':sum(p.numel() for name,p in model.named_parameters() if not name.startswith('protected_response.')),
        'new_stage_optimizer_updates':0,'zero_factor_head':True}
    torch.save({'architecture':model.architecture_id,'model':model.state_dict(),'config':config,'step':0,'operator_probe_provenance':provenance,
        'data_manifest_sha256':hashlib.sha256((directory/'data_manifest.json').read_bytes()).hexdigest()},checkpoint)
    (directory/'initialization.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps({**provenance,'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest()},indent=2))


if __name__=='__main__':main()
