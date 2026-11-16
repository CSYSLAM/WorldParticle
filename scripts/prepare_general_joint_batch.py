"""Freeze a configuration-only batching candidate with unchanged trained weights."""
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def main():
    import torch
    from wp_vbd.mixed_solver import MixedSolver
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    parent_dir = ROOT / 'reports/shared_prescribed_v30'
    parent_path = ROOT / 'checkpoints/shared_prescribed_v30/shared_strict.pt'
    parent = json.loads((parent_dir / 'checkpoint.json').read_text())
    assert sha(parent_path) == parent['checkpoint_sha256']
    verification_dir = ROOT / 'reports/general_joint_batch_v31'
    validations = ['verification_cpu_with_training.json', 'verification_cuda.json']
    for name in validations:
        report = json.loads((verification_dir / name).read_text())
        assert report['status'] == 'PASS' and report['parent_checkpoint_sha256'] == parent['checkpoint_sha256']
        assert report['new_optimizer_updates'] == report['new_training_admissions'] == 0
        assert len(report['inherited_legacy']) == 10 and len(report['two_step_training_probes']) == 5
        for key, value in report['implementation_sha256_at_start'].items(): assert sha(ROOT / 'src/wp_vbd' / key) == value
    directory = ROOT / 'reports/shared_joint_batch_v31'
    target = ROOT / 'checkpoints/shared_joint_batch_v31/shared_strict.pt'
    if target.exists() or (directory / 'checkpoint.json').exists(): raise FileExistsError('Keep frozen batching candidates immutable')
    before = torch.load(parent_path, weights_only=True, map_location='cpu')
    candidate = copy.deepcopy(before)
    candidate['config']['model']['general_joint_singleton_fast_path'] = True
    candidate['step'] = 0; candidate['sampled_updates'] = {}; candidate['new_optimizer_updates'] = 0
    candidate['parent_checkpoint_sha256'] = candidate['initial_checkpoint_sha256'] = parent['checkpoint_sha256']
    candidate['inherited_data_manifest_sha256'] = candidate.pop('data_manifest_sha256')
    candidate['active_weight_optimizer_samples'] = {row['dataset']: row['updates'] for row in parent['selected_checkpoint_datasets']}
    candidate['operator_probe_provenance'] = {'change': 'Enable optional general-joint singleton batching; every NN tensor is unchanged', 'parent_checkpoint_sha256': parent['checkpoint_sha256'], 'new_optimizer_updates': 0, 'new_training_admissions': 0, 'selected_parent_trial_step': before['step'], 'scope': 'Existing training coverage is inherited from the active unchanged weights, not from a new optimization trial. Native quality and speed are checked separately.'}
    model = MixedSolver(candidate['config']['model']); model.load_state_dict(candidate['model'])
    assert all(torch.equal(value, candidate['model'][key]) for key, value in before['model'].items())
    assert sum(p.numel() for p in model.protected_response.parameters()) == 84139
    target.parent.mkdir(parents=True, exist_ok=True); torch.save(candidate, target)
    reread = torch.load(target, weights_only=True, map_location='cpu')
    assert all(torch.equal(value, reread['model'][key]) for key, value in before['model'].items())
    result = {'status': 'UNCHANGED_WEIGHT_BATCHING_CANDIDATE_FROZEN', 'checkpoint': str(target.relative_to(ROOT)), 'checkpoint_sha256': sha(target), 'parent_checkpoint_sha256': parent['checkpoint_sha256'], 'selected_new_updates': 0, 'completed_current_trial_updates': 0, 'trained_official_demo_count': parent['trained_official_demo_count'], 'trained_official_demos': parent['trained_official_demos'], 'selected_checkpoint_datasets': parent['selected_checkpoint_datasets'], 'current_stage_optimizer_datasets': {}, 'current_stage_nonzero_gradient_updates': {}, 'count_scope': 'All 528 selected samples are inherited from the unchanged active parent weights. This operator batching trial has zero new optimizer updates and adds zero demos.', 'all_nn_tensors_bitexact': True, 'active_neural_parameters': 84139, 'model_config_additions': {'general_joint_singleton_fast_path': True}, 'inherited_legacy_behavior_audit': parent['inherited_legacy_behavior_audit'], 'verification_sha256': {name: sha(verification_dir / name) for name in validations}, 'parent_training_report_sha256': sha(parent_dir / 'training.json'), 'parent_ancestry_audit_sha256': sha(parent_dir / 'weight_ancestry_audit.json'), 'preparation_script_sha256': sha(Path(__file__)), 'scope': 'Configuration-only operator change. Same shared trained weights and fixed physical budget. Full native behavior, all 51 demos and realtime remain independent checks.'}
    directory.mkdir(parents=True, exist_ok=True)
    (directory / 'checkpoint.json').write_text(json.dumps(result, indent=2) + '\n')
    (directory / '.gitignore').write_text('*.npz\n*.log\n')
    (verification_dir / '.gitignore').write_text('*.npz\n*.log\n')
    print(json.dumps({key: result[key] for key in ('status', 'checkpoint_sha256', 'trained_official_demo_count', 'selected_new_updates')}, indent=2))


if __name__ == '__main__':
    main()
