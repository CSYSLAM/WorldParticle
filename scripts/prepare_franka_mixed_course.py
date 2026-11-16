"""Append qualified Franka data while retaining every existing pure-VBD row.

Materializes an explicit model-only research initializer with a fresh AdamW
origin. It does not resume an eligible producer, optimize weights, or claim
that newly scheduled datasets have already been trained.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-manifest',type=Path,default=ROOT/'reports/shared_normal_648_v302/data_manifest.json')
    parser.add_argument('--parent',type=Path,default=ROOT/'checkpoints/plateau_review_v303/shared_trial_2478.pt')
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    import numpy as np
    import torch
    from wp_vbd.training_scope import validate_rows,load_scope,CUSTOM_PURE_VBD
    from wp_vbd.development_cases import validation_rows,case_key
    from wp_vbd.development_baseline import PHYSICAL_KEYS,validate_policy
    proof_path=args.dataset.with_suffix('.qualification.json');proof=json.loads(proof_path.read_text())
    if proof['status']!='QUALIFIED_FRANKA_PHYSICAL_SUBSTEP_DATA' or proof['output_sha256']!=digest(args.dataset):
        raise ValueError('Exact qualified Franka dataset required')
    base=json.loads(args.base_manifest.read_text());manifest=copy.deepcopy(base)
    parent=torch.load(args.parent,map_location='cpu',weights_only=True)
    config=parent['config']['model']
    if config!=base['effective_model_config']:raise ValueError('Declare and validate runtime changes before mixing courses')
    with np.load(args.dataset,allow_pickle=False) as z:metadata=json.loads(str(z['metadata']))
    demo=metadata['source_run']['scene'];loader,contract=CUSTOM_PURE_VBD[demo]
    row={'dataset_id':'franka_block_mu5000_lambda5000_grasp_v310','demo':demo,
         'loader':loader,'custom_vbd_contract':contract,
         'path':str(args.dataset.resolve().relative_to(ROOT)),'sha256':proof['output_sha256'],
         'qualification':str(proof_path.resolve().relative_to(ROOT)),'qualification_sha256':digest(proof_path),
         'teacher_iterations':metadata['source_run']['solver_parameters']['offline_reference_iterations'],
         'teacher_substeps':proof['physical_steps'],
         'qualified_train_starts':[520,1050,390,800,1200,350,600,1150,250,900,450,680,480,560,0,20,50,100],
         'validation_starts':[430,720,1350],'development_windows':[430,720,1350],
         'split':'distinct_phase_windows_development_only_not_held_out_material',
         'admitted_file_stat':{'bytes':args.dataset.stat().st_size,'mtime_ns':args.dataset.stat().st_mtime_ns}}
    if any(r['dataset_id']==row['dataset_id'] for r in base['scenes']):raise ValueError('Dataset already present')
    manifest['scenes'].append(row)
    assert manifest['scenes'][:-1]==base['scenes']
    scope=load_scope(ROOT);coverage=validate_rows(manifest['scenes'],scope)
    assert coverage['scheduled_official_demos']==validate_rows(base['scenes'],scope)['scheduled_official_demos']
    args.output.mkdir(parents=True)
    initial=args.output/'research_initial.pt'
    torch.save({'model_config':config,'model':parent['model'],'step':0,
                'research_parent_sha256':digest(args.parent),'research_parent_step':parent['step'],
                'optimizer_origin':'fresh_AdamW_no_parent_optimizer_or_selected_status'},initial)
    manifest.update(status='RESEARCH_MIXED_COURSE_PREPARED_PENDING_GPU_PREFLIGHT',
        initial_checkpoint=str(initial.resolve().relative_to(ROOT)),initial_checkpoint_sha256=digest(initial),
        initial_probe_updates=0,inherited_selected_official_demos=0,
        current_course_dataset_count=len(manifest['scenes']),training_scope=coverage,
        current_course_custom_pure_vbd_demos=coverage['scheduled_custom_pure_vbd_demos'],
        current_course_custom_pure_vbd_demo_count=coverage['scheduled_custom_pure_vbd_demo_count'])
    for name in ('selected_parent','selected_export','joint_course_policy'):manifest.pop(name,None)
    manifest['research_initialization']={'parent_checkpoint':str(args.parent.resolve().relative_to(ROOT)),
        'parent_sha256':digest(args.parent),'parent_step':parent['step'],'parent_was_selected_eligible':False,
        'weights_copied_exactly':True,'optimizer':'fresh_AdamW','new_optimizer_updates':0}
    manifest['research_sampling']={'schema':'fresh_research_replay_v1',
        'new_dataset_ids':[row['dataset_id']],'repeats':{row['dataset_id']:9}}
    from wp_vbd.research_course import sampling_rows
    updates_per_pass=len(sampling_rows(manifest['scenes'],manifest['research_sampling']))
    manifest['source_manifests_sha256'][str(args.base_manifest.resolve().relative_to(ROOT))]=digest(args.base_manifest)
    policies=manifest['development_baseline_policies']
    matched=validate_policy({key:True for key in PHYSICAL_KEYS if config.get(key,False)})
    old_keys={case_key(r) for r in validation_rows(base['scenes'])}
    for development in validation_rows(manifest['scenes']):
        key=case_key(development)
        if key not in old_keys:policies[key]=matched
    assert set(policies)=={case_key(r) for r in validation_rows(manifest['scenes'])}
    manifest['resource_policy'].update(gpu_process_budget_gib=16,torch_allocator_budget_gib=14,maximum_duty_cycle=1.)
    manifest_path=args.output/'data_manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    command=['.venv/bin/python','-u','-B','scripts/train_normal_shared_648.py','--manifest',str(manifest_path),
             '--device','cuda:0','--updates',str(2*updates_per_pass),'--validation-interval',str(updates_per_pass),
             '--loss-monitor-interval',str(updates_per_pass),'--learning-rate','1e-6',
             '--gpu-memory-gib','16','--duty-cycle','1','--checkpoint-neural-encoder',
             '--host-self-trajectories','--host-rod-trajectories','--host-encoder-inputs',
             '--host-step-saved-tensors','--compact-host-reference-constants','--trim-host-allocator',
             '--checkpoint-relation-ffn-rows','8192','--checkpoint-attention-blocks']
    plan={'status':'PREPARED_NOT_LAUNCHED','argv':command,'new_optimizer_updates':0,
          'existing_rows_unchanged':len(base['scenes']),'total_datasets':len(manifest['scenes']),
          'official_demo_count':coverage['scheduled_official_demo_count'],
          'custom_demo_count':coverage['scheduled_custom_pure_vbd_demo_count'],
          'updates_per_pass':updates_per_pass,'new_dataset_updates_per_pass':9,
          'planned_added_dataset_windows':row['qualified_train_starts'],
          'pending':'GPU admission of new phase windows and initial mixed-course development check before optimization.',
          'scope':'One shared 648-wide network; all existing standalone VBD replay retained. Research initialization does not inherit selected status or new optimizer counts.'}
    (args.output/'launch_plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps({key:value for key,value in plan.items() if key!='argv'},indent=2))


if __name__=='__main__':main()
