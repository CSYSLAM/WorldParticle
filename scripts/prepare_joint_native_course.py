"""Combine all official data, loaded mixed asset and both authored materials."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-stage',required=True);parser.add_argument('--selected-export',type=Path,required=True)
    parser.add_argument('--mixed-data',type=Path,required=True);parser.add_argument('--mixed-qualification',type=Path,required=True)
    parser.add_argument('--candidates',type=Path,required=True);parser.add_argument('--preflight-directory',type=Path,required=True)
    parser.add_argument('--epochs',type=int,default=4);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--learned-contact-repair',action='store_true');args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    from wp_vbd.joint_course import digest
    from wp_vbd.native_mixed_course import selected_official_parent,MIXED_POLICY
    from wp_vbd.native_parameter_course import qualified_candidate
    from wp_vbd.training_scope import load_scope,validate_rows
    from wp_vbd.development_baseline import extend_policies,validate_runtime_overrides
    from wp_vbd.development_cases import validation_rows
    from prepare_native_five_family_course import qualified_asset
    parent=ROOT/'reports'/args.parent_stage;producer=json.loads((parent/'training.json').read_text())
    manifest=copy.deepcopy(json.loads((parent/'data_manifest.json').read_text()));old_rows=manifest['scenes']
    selected,coverage=selected_official_parent(producer,load_scope(ROOT));export=json.loads(args.selected_export.read_text())
    if export['selected_updates']!=producer['selected_step'] or export['trained_official_demos']!=coverage['official_demos']:
        raise ValueError('Exact stopped selected all42 parent required')
    for key,sha in [('checkpoint','checkpoint_sha256'),('source_trial_checkpoint','source_trial_checkpoint_sha256'),('source_training_report','source_training_report_sha256')]:
        if digest(ROOT/export[key])!=export[sha]:raise ValueError('Selected parent artifact changed: '+key)
    configuration={**producer['config'],**MIXED_POLICY}
    if args.learned_contact_repair:configuration['learned_contact_repair']=True
    overrides=validate_runtime_overrides({k:v for k,v in configuration.items() if producer['config'].get(k)!=v})
    metadata,data_sha=qualified_asset(args.mixed_data,args.mixed_qualification,manifest['newton_sha'])
    def relative(path):
        path=Path(path)
        if not path.is_absolute():return str(path)
        try:return str(path.relative_to(ROOT))
        except ValueError:
            import subprocess
            common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],cwd=ROOT,text=True).strip()).parent
            return str(path.resolve().relative_to(common))
    def stat(path):
        value=Path(path).stat();return {'bytes':value.st_size,'mtime_ns':value.st_mtime_ns}
    mixed={'dataset_id':'native_five_families_train_default','demo':'research.native_five_families','loader':'elastic_rod',
        'custom_vbd_contract':metadata['contract'],'path':relative(args.mixed_data),'sha256':data_sha,
        'qualification':relative(args.mixed_qualification),'qualification_sha256':digest(args.mixed_qualification),
        'qualified_train_starts':[0,1,600,1800],'validation_starts':[2800],'development_windows':[2800],
        'self_contact_motion_budget':.02,'self_contact_max_rows':metadata['self_contact_max_rows'],
        'self_contact_responsive_pairs_only':False,'teacher_substeps':3000,'admitted_file_stat':stat(args.mixed_data)}
    additions=[mixed]
    for candidate in json.loads(args.candidates.read_text())['candidate_rows']:
        qualified_candidate(candidate,ROOT,manifest['newton_sha'])
        row={k:copy.deepcopy(candidate[k]) for k in ['dataset_id','demo','loader','custom_vbd_contract','path','sha256','qualification','qualification_sha256','qualified_train_starts','validation_starts','self_contact_motion_budget','self_contact_max_rows','teacher_substeps']}
        row.update(development_windows=row['validation_starts'],self_contact_responsive_pairs_only=False,admitted_file_stat=stat(ROOT/row['path']))
        additions.append(row)
    if len(additions)!=3:raise ValueError('Exactly three bound new datasets required')
    rows=[*additions,*old_rows];policies=extend_policies(producer,rows,configuration)
    preflights={r['dataset_id']:args.preflight_directory/(r['dataset_id']+'_cuda.json') for r in additions}
    policy={'kind':'joint_authored_course_deferred_full_development_v282','user_requested_combined_training':True,
        'reason':'User requests all mixed/material data trained together and fixed-window loss convergence checks; repeated full-suite prechecks and interstage CPU rollouts are deferred.',
        'epochs':args.epochs,'full_development_at_end':True,'pretraining_full_suite_skipped':True,
        'early_stop':'Observed fixed-validation plateau after at least four complete courses; full development at the actual end before promotion.',
        'inherited_manifest':str(Path('reports')/args.parent_stage/'data_manifest.json'),'inherited_manifest_sha256':digest(parent/'data_manifest.json'),
        'parent_inference_sha256':export['checkpoint_sha256'],'cuda_admissions':{key:{'path':str(path.resolve()),'sha256':digest(path)} for key,path in preflights.items()},
        'loss_monitor_scope':'Fixed authored windows plus five official object-family development windows; equal-dataset means. State and repair losses reported separately.'}
    manifest.update(status='JOINT_ALL_OFFICIAL_AND_THREE_AUTHORED_DATASETS_PREPARED_NOT_TRAINED',scenes=rows,
        training_scope=validate_rows(rows,load_scope(ROOT)),current_course_official_demos=coverage['official_demos'],current_course_demo_count=42,
        current_course_dataset_count=len(rows),current_course_custom_pure_vbd_demos=['research.native_five_families'],current_course_custom_pure_vbd_demo_count=1,
        inherited_selected_official_demos=42,old32_demo_coverage_inherited=0,effective_model_config=configuration,model_configuration_overrides=overrides,
        development_baseline_policies=policies,selected_parent={'checkpoint':export['source_trial_checkpoint'],
            'checkpoint_sha256':export['source_trial_checkpoint_sha256'],'report':str(Path('reports')/args.parent_stage/'training.json'),
            'report_sha256':digest(parent/'training.json'),'inference_checkpoint_sha256':export['checkpoint_sha256'],'selected_updates':producer['selected_step']},
        joint_course_policy=policy)
    manifest['source_manifests_sha256'][policy['inherited_manifest']]=policy['inherited_manifest_sha256']
    manifest.pop('optimizer_learning_rate_migration',None)
    from wp_vbd.joint_course import verify_joint_training_policy
    verify_joint_training_policy(manifest,ROOT,configuration,{'valid_updates_recovered':producer['selected_step']},
        producer['selected_step']+args.epochs*len(rows),args.epochs*len(rows),len(rows))
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'status':manifest['status'],'datasets':len(rows),'official_demos':42,
        'development_cases_at_end':len(validation_rows(rows)),'new_optimizer_updates':0,'epochs':args.epochs},indent=2))


if __name__=='__main__':main()
