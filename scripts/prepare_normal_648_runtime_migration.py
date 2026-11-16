"""Migrate a stopped course from its selected prefix without new coverage."""
import argparse
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):result.update(part)
    return result.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-stage',default='shared_normal_648_v96')
    parser.add_argument('--selected-export',type=Path,default=ROOT/'reports/shared_normal_648_v96/inference_checkpoint_step_000525.json')
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/shared_normal_648_v100/data_manifest.json')
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    import torch
    from wp_vbd.development_baseline import extend_policies,selected_parent_record
    parent=ROOT/'reports'/args.parent_stage;producer_path=parent/'training.json'
    producer=json.loads(producer_path.read_text());selected_parent_record(producer)
    evidence=json.loads(args.selected_export.read_text())
    assert evidence['selected_updates']==producer['selected_step']
    inference=ROOT/evidence['checkpoint'];assert digest(inference)==evidence['checkpoint_sha256']
    trial=ROOT/evidence['source_trial_checkpoint'];assert digest(trial)==evidence['source_trial_checkpoint_sha256']
    recovered=torch.load(trial,weights_only=True,map_location='cpu')
    assert recovered['step']==producer['selected_step'] and recovered['model_config']==producer['config']
    manifest_path=parent/'data_manifest.json';manifest=deepcopy(json.loads(manifest_path.read_text()))
    from wp_vbd.training_scope import load_scope,validate_rows
    manifest['training_scope']=validate_rows(manifest['scenes'],load_scope(ROOT))
    assert recovered['data_manifest_sha256']==digest(manifest_path)
    samples=[row for row in producer['optimization_samples'] if row['update']<=recovered['step']]
    assert [row['update'] for row in samples]==list(range(1,recovered['step']+1))
    assert dict(Counter(row['dataset'] for row in samples))=={key:value for key,value in recovered['sampled_updates'].items() if value}
    del recovered
    expected={**producer['config'],'elastic_prestress_metric':True}
    overrides={key:value for key,value in expected.items() if producer['config'].get(key)!=value}
    assert overrides=={'elastic_prestress_metric':True},'This preparation enables only the verified prestress policy'
    proof=json.loads(args.preflight.read_text())
    assert proof['status']=='PASS' and proof['device']=='cuda:0'
    assert proof['checkpoint_sha256']==evidence['checkpoint_sha256']
    assert proof['effective_model_config']==expected and proof['model_configuration_overrides']==overrides
    assert proof['effective_trainable_parameters']==46882811 and proof['full_shared_backbone_gradients_enabled']
    assert proof['adam_moments_resident_during_forward_backward'] and proof['encoder_recomputation_enabled']
    assert proof['host_self_trajectories'] and proof['reference_state_device']=='cpu' and proof['broad_phase']=='kdtree'
    assert proof['new_optimizer_updates']==proof['new_training_admissions']==proof['full_vbd_prediction_calls']==0
    assert proof['gpu_resource_budget']['process_budget_gib']==12 and proof['gpu_resource_budget']['torch_allocator_cap_gib']==10
    assert proof['gpu_resource_budget']['observed_process_max_mib']<=12*1024
    row=next(row for row in manifest['scenes'] if row['dataset_id']=='softbody_gift_train_default')
    assert proof['data_sha256']==row['sha256']==digest(ROOT/row['path'])
    assert sorted(case['index'] for case in proof['cases'])==sorted(row['qualified_train_starts']+row['validation_starts'])
    for case in proof['cases']:
        assert case['all_requested_steps_accepted'] and case['two_hop_accepted']
        assert all(value>0 for value in case['backbone_gradient_norms'].values())
        assert all(case['self_relation_embedding_gradient_norms'][str(kind)]>0 for kind in (9,10))
        assert case['active_soft_cloth_self_rows']>0
    demos=sorted({row['demo'] for row in manifest['scenes'] if row.get('demo')})
    assert demos==evidence['trained_official_demos']
    policies=extend_policies(producer,manifest['scenes'],expected)
    baseline_path=ROOT/'checkpoints/shared_particle_hanging_v63/shared_strict.pt'
    assert digest(baseline_path)==producer['development_comparison_policy']['old32_checkpoint_sha256']
    # Old rows, splits, policies and baseline caps remain byte-for-byte values.
    for row in manifest['scenes']:
        assert digest(ROOT/row['path'])==row['sha256']
        if row.get('validation_path'):assert digest(ROOT/row['validation_path'])==row['validation_sha256']
    manifest.update(status='SELECTED_PREFIX_RUNTIME_MIGRATION_PREPARED_NOT_TRAINED',
        effective_model_config=expected,model_configuration_overrides=overrides,
        development_baseline_policies=policies,development_baseline_checkpoint_sha256=digest(baseline_path),
        inherited_selected_official_demos=len(demos),old32_demo_coverage_inherited=0,
        selected_parent={'checkpoint':str(trial.relative_to(ROOT)),'checkpoint_sha256':digest(trial),
            'report':str(producer_path.relative_to(ROOT)),'report_sha256':digest(producer_path),
            'inference_checkpoint_sha256':evidence['checkpoint_sha256'],'selected_updates':producer['selected_step']},
        new_dataset_preflight_sha256={str(args.preflight.resolve().relative_to(ROOT)):digest(args.preflight)},
        runtime_migration={'parent_status':producer['status'],'parent_failure':producer.get('failure'),
            'selected_parent_updates':producer['selected_step'],'discarded_unselected_calls':producer['completed_updates']-producer['selected_step'],
            'new_datasets':0,'required_pretraining_all_demo_development':True},
        scope='Same selected648 model/Adam/RNG, exact selected ancestry and all replay rows. Enable verified positive prestress metric; preserve original per-demo comparators/caps. No added demo coverage; all-course development must pass before optimizer calls. Later unselected parent calls are excluded.')
    manifest['source_manifests_sha256'][str(manifest_path.relative_to(ROOT))]=digest(manifest_path)
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({key:manifest[key] for key in ('status','current_course_demo_count','current_course_dataset_count','model_configuration_overrides','runtime_migration')},indent=2))


if __name__=='__main__':main()
