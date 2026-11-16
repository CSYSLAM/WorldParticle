"""Freeze the projected-inverse experiment and its unchanged data protocol."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import torch
from wp_vbd.mixed_solver import MixedSolver

def main():
    parent=ROOT/'checkpoints/shared_protected_v17/shared_strict.pt'
    directory=ROOT/'reports/shared_inverse_v18';checkpoint=ROOT/'checkpoints/shared_inverse_v18/architecture_probe.pt'
    config_path=ROOT/'configs/shared_inverse_v18.json'
    if checkpoint.exists() or config_path.exists():raise FileExistsError('Preserve frozen initialization')
    stored=torch.load(parent,weights_only=True,map_location='cpu');config=copy.deepcopy(stored['config'])
    config.update(name='shared_inverse_v18',seed=20261106,report_dir='reports/shared_inverse_v18',checkpoint_dir='checkpoints/shared_inverse_v18',initialize_from=str(checkpoint))
    config['model'].update(protected_inverse_replacement=True,protected_inverse_ridge=1e-8)
    config['training'].update(steps=128,batch_seed=20261106)
    config['scope'].update(protected_response='Physical energy-orthogonal projection plus shared learned six-column inverse replacement in its complement; zero scalar gate equals the matching physical baseline exactly.',
        factor_budget='At most 6x6 projected inverse solves and local factors; physical Q has at most 12 columns. No global dynamics matrix or convergence loop.',
        training_curriculum='Warm-start shared encoder/shapes from v17 selected update 64; new invariant context and scalar inverse gate initialized to zero.',
        official_coverage='Active encoder has one official demo plus five-type custom mixed training, not historical ten-demo neural coverage.',
        experimental_comparison='Sequential architecture development with a warm start, not a matched-total-training-budget architecture comparison.')
    torch.manual_seed(config['seed']);model=MixedSolver(config['model'])
    inherited={k:v for k,v in stored['model'].items() if not k.startswith('protected_response.factor.')}
    incompatible=model.load_state_dict(inherited,strict=False)
    assert not incompatible.unexpected_keys
    assert set(incompatible.missing_keys)=={'protected_response.factor.weight','protected_response.factor.bias','protected_response.context.0.weight','protected_response.context.0.bias','protected_response.context.2.weight','protected_response.context.2.bias'}
    assert all(torch.equal(v,model.state_dict()[k]) for k,v in inherited.items())
    directory.mkdir(parents=True,exist_ok=True);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    (directory/'.gitignore').write_text('*.log\n*.npz\n')
    manifest=json.loads((ROOT/'reports/shared_protected_v17/data_manifest.json').read_text())
    manifest.update(architecture_experiment='Shared projected-inverse response; v17 data and splits unchanged',inherited_actual_training_demo_count=1,
        active_architecture_coverage='One official transfer demo plus five-type custom mixed fixtures, inherited encoder and shapes. Legacy response remains stored but unused.')
    (directory/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    config_path.write_text(json.dumps(config,indent=2)+'\n')
    provenance={'status':'UNTRAINED_INVERSE_GATE','parent_checkpoint_sha256':hashlib.sha256(parent.read_bytes()).hexdigest(),
        'parent_selected_update':stored['step'],'active_parameters':sum(p.numel() for p in model.protected_response.parameters()),
        'new_stage_optimizer_updates':0,'zero_gate_exact_physical_baseline':True,'warm_started_encoder_and_shapes':True,
        'active_official_demo_training_count':1,'old_neural_parameters_stored_but_unused':sum(p.numel() for n,p in model.named_parameters() if not n.startswith('protected_response.'))}
    torch.save({'architecture':model.architecture_id,'model':model.state_dict(),'config':config,'step':0,'operator_probe_provenance':provenance},checkpoint)
    provenance['checkpoint_sha256']=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    (directory/'initialization.json').write_text(json.dumps(provenance,indent=2)+'\n');print(json.dumps(provenance,indent=2))

if __name__=='__main__':main()
