"""Append qualified authored material episodes to the selected mixed course."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path):
    path=Path(path)
    if not path.is_absolute():return str(path)
    try:return str(path.relative_to(ROOT))
    except ValueError:
        common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],cwd=ROOT,text=True).strip()).parent
        return str(path.relative_to(common))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--parent-stage',required=True)
    parser.add_argument('--selected-export',type=Path,required=True);parser.add_argument('--candidates',type=Path,required=True)
    parser.add_argument('--preflight',action='append',default=[],help='dataset_id=actual_CUDA_report.json')
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    from wp_vbd.native_parameter_course import selected_mixed_parent,qualified_candidate,verify_parameter_preflight
    from wp_vbd.native_mixed_course import MIXED_POLICY
    from wp_vbd.training_scope import load_scope,validate_rows,CUSTOM_PURE_VBD
    from wp_vbd.development_baseline import extend_policies,validate_runtime_overrides
    from wp_vbd.development_cases import validation_rows
    parent=ROOT/'reports'/args.parent_stage;producer=json.loads((parent/'training.json').read_text())
    manifest=copy.deepcopy(json.loads((parent/'data_manifest.json').read_text()));old_rows=manifest['scenes'];scope=load_scope(ROOT)
    selected,coverage=selected_mixed_parent(producer,old_rows,scope)
    evidence=json.loads(args.selected_export.read_text());snapshot=ROOT/evidence['source_training_report']
    if digest(snapshot)!=evidence['source_training_report_sha256']:raise ValueError('Actual parent export snapshot changed')
    exported=json.loads(snapshot.read_text());limit=producer['selected_step']
    if evidence['selected_updates']!=limit or evidence['trained_official_demos']!=coverage['official_demos'] or evidence['trained_custom_pure_vbd_datasets']!=coverage['custom_pure_vbd_datasets']:
        raise ValueError('Export must bind actual selected official and authored optimizer coverage')
    if exported['source_git_commit']!=producer['source_git_commit'] or exported['config']!=producer['config']:
        raise ValueError('Parent producer and export identities differ')
    if [r for r in exported['optimization_samples'] if r['update']<=limit]!=[r for r in producer['optimization_samples'] if r['update']<=limit]:
        raise ValueError('The selected optimizer prefix changed')
    if [r for r in exported['validation_history'] if r['update']==limit and r['eligible']]!=[selected]:
        raise ValueError('The original eligible parent comparison must be preserved')
    trial=ROOT/evidence['source_trial_checkpoint'];checkpoint=ROOT/evidence['checkpoint']
    if digest(trial)!=evidence['source_trial_checkpoint_sha256'] or digest(checkpoint)!=evidence['checkpoint_sha256']:
        raise ValueError('Selected inference weights or raw Adam/RNG checkpoint changed')
    newton=Path('/home/oem/code/engine/newton')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=newton,text=True).strip()!=manifest['newton_sha'] or subprocess.run(['git','diff','--quiet'],cwd=newton).returncode:
        raise ValueError('Pinned clean tracked Newton source required')
    candidates=json.loads(args.candidates.read_text());additions=candidates['candidate_rows']
    if not additions:raise ValueError('Explicit new authored parameter episodes required')
    proofs={}
    for value in args.preflight:
        key,separator,path=value.partition('=')
        if not separator or key in proofs:raise ValueError('Unique dataset_id=CUDA_report pairs required')
        proofs[key]=Path(path)
    if set(proofs)!={row['dataset_id'] for row in additions}:raise ValueError('Actual CUDA proof required for every exact new dataset')
    configuration={**producer['config'],**MIXED_POLICY}
    overrides=validate_runtime_overrides({k:v for k,v in configuration.items() if producer['config'].get(k)!=v})
    implementation={p.name:digest(p) for p in (ROOT/'src/wp_vbd').glob('*.py')};rows=[];qualifications=[]
    for candidate in additions:
        metadata=qualified_candidate(candidate,ROOT,manifest['newton_sha'])
        proof_path=proofs[candidate['dataset_id']];proof=json.loads(proof_path.read_text())
        verify_parameter_preflight(proof,configuration=configuration,checkpoint_sha256=evidence['checkpoint_sha256'],
            data_sha256=candidate['sha256'],implementation_sha256=implementation)
        if proof['script_sha256']!=digest(ROOT/'scripts/verify_elastic_rod_training_budget.py'):
            raise ValueError('CUDA preflight implementation changed')
        path=ROOT/candidate['path'];stat=path.stat()
        row={k:copy.deepcopy(candidate[k]) for k in ('dataset_id','demo','loader','custom_vbd_contract','path','sha256',
             'qualification','qualification_sha256','qualified_train_starts','validation_starts','self_contact_motion_budget','self_contact_max_rows','teacher_substeps')}
        row.update(development_windows=row['validation_starts'],self_contact_responsive_pairs_only=False,
            admitted_file_stat={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        rows.append(row);qualifications.append({'dataset_id':row['dataset_id'],'preflight':relative(proof_path),'preflight_sha256':digest(proof_path),
            'physical_parameters':candidate['physical_parameters'],'native_definition_schema':metadata['asset_definition']['schema']})
    rows+=old_rows
    if len({row['dataset_id'] for row in rows})!=len(rows):raise ValueError('Every inherited and added dataset must retain its unique identity')
    for row in old_rows:
        if digest(ROOT/row['path'])!=row['sha256']:raise ValueError('Inherited data changed: '+row['dataset_id'])
        if row.get('validation_path') and digest(ROOT/row['validation_path'])!=row['validation_sha256']:
            raise ValueError('Inherited development data changed: '+row['dataset_id'])
    baseline=digest(ROOT/'checkpoints/shared_particle_hanging_v63/shared_strict.pt')
    if baseline!=producer['development_comparison_policy']['old32_checkpoint_sha256']:raise ValueError('Inherited32 comparator changed')
    custom=sorted({r['demo'] for r in rows if r.get('demo') in CUSTOM_PURE_VBD})
    manifest.update(status='NORMAL_AUTHORED_PARAMETER_EXTENSION_PREPARED_NOT_TRAINED',scenes=rows,
        training_scope=validate_rows(rows,scope),current_course_official_demos=coverage['official_demos'],
        current_course_demo_count=len(coverage['official_demos']),current_course_dataset_count=len(rows),
        current_course_custom_pure_vbd_demos=custom,current_course_custom_pure_vbd_demo_count=len(custom),
        inherited_selected_official_demos=len(coverage['official_demos']),old32_demo_coverage_inherited=0,
        effective_model_config=configuration,model_configuration_overrides=overrides,
        development_baseline_policies=extend_policies(producer,rows,configuration),development_baseline_checkpoint_sha256=baseline,
        selected_parent={'checkpoint':relative(trial),'checkpoint_sha256':digest(trial),'report':relative(parent/'training.json'),
            'report_sha256':digest(parent/'training.json'),'inference_checkpoint_sha256':evidence['checkpoint_sha256'],'selected_updates':limit},
        selected_export={'evidence':relative(args.selected_export),'evidence_sha256':digest(args.selected_export),
            'snapshot':relative(snapshot),'snapshot_sha256':digest(snapshot)},authored_parameter_qualifications=qualifications,
        authored_parameter_candidate_sha256=digest(args.candidates))
    manifest['source_manifests_sha256'][relative(parent/'data_manifest.json')]=digest(parent/'data_manifest.json')
    manifest['new_dataset_preflight_sha256']={**manifest.get('new_dataset_preflight_sha256',{}),
        **{relative(path):digest(path) for path in proofs.values()}}
    manifest.pop('optimizer_learning_rate_migration',None)
    manifest['scope']='Preserve every selected parent dataset and official42 coverage; append independently qualified authored material episodes to the same648 weights/Adam/RNG. Every new material/window gets a separate development case, while scoring averages windows within each added dataset. Preparation is not training or quality acceptance.'
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as stream:json.dump(manifest,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':manifest['status'],'datasets':len(rows),'development_cases':len(validation_rows(rows)),
        'new_optimizer_updates':0,'new_training_admissions':0},indent=2))


if __name__=='__main__':main()
