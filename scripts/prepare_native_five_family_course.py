"""Extend the selected full pure-VBD course with one qualified authored asset."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def relative(path):
    path=Path(path)
    if not path.is_absolute():return str(path)
    try:return str(path.relative_to(ROOT))
    except ValueError:
        # Reference/checkpoint mirrors are shared by detached worktrees.
        common=Path(subprocess.check_output(['git','rev-parse','--path-format=absolute','--git-common-dir'],
                                           cwd=ROOT,text=True).strip()).parent
        return str(path.relative_to(common))


def qualified_asset(data_path,qualification_path,expected_newton):
    """Check the real qualified file without constructing a model or CUDA context."""
    import numpy as np
    qualification=json.loads(qualification_path.read_text())
    data_sha=digest(data_path)
    if qualification.get('status')!='NATIVE_MIXED_CAUSAL_HISTORY_AND_ALL_REFERENCE_PATHS_QUALIFIED':
        raise ValueError('Actual complete authored-data qualification required')
    if qualification.get('output_sha256')!=data_sha:
        raise ValueError('Qualification must bind the exact authored data')
    if not all(qualification.get(key) is True for key in
               ('eligible_normal_course','eligible_normal_course_length',
                'complete_reference_material_and_VF_EE_paths_verified')):
        raise ValueError('Complete qualified histories, material and VF/EE paths required')
    if qualification.get('teacher_substeps')!=3000 or qualification.get('source_numeric_fields_verified')!=200:
        raise ValueError('The complete first five-family asset is required')
    if qualification.get('full_solver_or_shape_collision_calls')!={'construct':0,'step':0,'shape_collision':0}:
        raise ValueError('Qualification may not invoke a replacement full solver')
    with np.load(data_path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        if archive['x'].shape[0]!=3001:raise ValueError('Complete five-second reference required')
    if metadata.get('contract')!='elastic_rod_surface_joint_self_contact_v222' or metadata.get('training_schema')!='accepted_elastic_rod_inputs_and_history_v226':
        raise ValueError('Explicit causal native mixed contract required')
    if metadata.get('official_demo')!='research.native_five_families' or metadata.get('newton_sha')!=expected_newton:
        raise ValueError('Authored asset and pinned Newton identity required')
    if not all(metadata.get(k) is True for k in ('eligible_normal_course','reference_format_qualified')):
        raise ValueError('Qualified normal-course metadata required')
    if metadata.get('teacher_substeps')!=3000 or metadata.get('dt')!=1/600 or metadata.get('self_contact_motion_budget')!=.02:
        raise ValueError('Measured five-second reference and2cm envelope required')
    definition=metadata.get('source_self_contact_definition',{})
    if definition!=qualification.get('source_self_contact_definition') or definition.get('compatible') is not True:
        raise ValueError('Qualified reciprocal topology scope required')
    if digest(ROOT/'src/wp_vbd/elastic_rod_demo.py')!=metadata['factory_sha256']:
        raise ValueError('The qualified asset factory changed')
    if digest(Path(metadata['source_trace_path'])/'manifest.json')!=metadata['source_trace_manifest_sha256']:
        raise ValueError('The original offline reference trace changed')
    return metadata,data_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-stage',required=True)
    parser.add_argument('--selected-export',type=Path,required=True)
    parser.add_argument('--preflight',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--qualification',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError('Preserve existing course evidence')
    from wp_vbd.training_scope import load_scope,validate_rows
    from wp_vbd.native_mixed_course import selected_official_parent,verify_preflight,MIXED_POLICY
    from wp_vbd.development_baseline import extend_policies,validate_runtime_overrides
    parent=ROOT/'reports'/args.parent_stage
    producer=json.loads((parent/'training.json').read_text())
    scope=load_scope(ROOT)
    selected,coverage=selected_official_parent(producer,scope)
    manifest=copy.deepcopy(json.loads((parent/'data_manifest.json').read_text()))
    old_rows=manifest['scenes']
    scheduled=validate_rows(old_rows,scope)
    if scheduled['scheduled_official_demos']!=coverage['official_demos'] or len(old_rows)!=66:
        raise ValueError('Preserve the complete66-dataset official course')
    evidence=json.loads(args.selected_export.read_text())
    if evidence['selected_updates']!=producer['selected_step'] or evidence['trained_official_demos']!=coverage['official_demos']:
        raise ValueError('Export must bind the actual eligible complete official prefix')
    snapshot=ROOT/evidence['source_training_report']
    if digest(snapshot)!=evidence['source_training_report_sha256']:raise ValueError('Selected export snapshot changed')
    exported=json.loads(snapshot.read_text());limit=producer['selected_step']
    if exported['source_git_commit']!=producer['source_git_commit'] or exported['config']!=producer['config']:
        raise ValueError('Export and terminal producer identities differ')
    if [r for r in exported['optimization_samples'] if r['update']<=limit]!=[r for r in producer['optimization_samples'] if r['update']<=limit]:
        raise ValueError('Export and terminal producer selected optimizer prefixes differ')
    if [r for r in exported['validation_history'] if r['update']==limit and r['eligible']]!=[selected]:
        raise ValueError('Export must preserve the eligible original development record')
    trial=ROOT/evidence['source_trial_checkpoint'];checkpoint=ROOT/evidence['checkpoint']
    if digest(trial)!=evidence['source_trial_checkpoint_sha256'] or digest(checkpoint)!=evidence['checkpoint_sha256']:
        raise ValueError('Selected training or inference weights changed')
    newton=Path('/home/oem/code/engine/newton')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=newton,text=True).strip()!=manifest['newton_sha']:
        raise ValueError('Pinned Newton source changed')
    if subprocess.run(['git','diff','--quiet'],cwd=newton).returncode:
        raise ValueError('Pinned Newton tracked source has local changes')
    metadata,data_sha=qualified_asset(args.data,args.qualification,manifest['newton_sha'])
    configuration={**producer['config'],**MIXED_POLICY}
    overrides=validate_runtime_overrides({k:v for k,v in configuration.items() if producer['config'].get(k)!=v})
    proof=json.loads(args.preflight.read_text())
    verify_preflight(proof,configuration=configuration,checkpoint_sha256=evidence['checkpoint_sha256'],
                     data_sha256=data_sha,implementation_sha256={p.name:digest(p) for p in (ROOT/'src/wp_vbd').glob('*.py')})
    if proof.get('script_sha256')!=digest(ROOT/'scripts/verify_elastic_rod_training_budget.py'):
        raise ValueError('Qualified preflight implementation changed')
    stat=args.data.stat()
    row={'dataset_id':'native_five_families_train_default','demo':'research.native_five_families',
         'loader':'elastic_rod','custom_vbd_contract':metadata['contract'],
         'path':relative(args.data),'sha256':data_sha,
         'qualification':relative(args.qualification),'qualification_sha256':digest(args.qualification),
         'qualified_train_starts':[0,1,600,1800],'validation_starts':[2800],
         'self_contact_motion_budget':.02,'self_contact_max_rows':metadata['self_contact_max_rows'],
         'self_contact_responsive_pairs_only':False,'teacher_substeps':3000,
         'admitted_file_stat':{'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}}
    rows=[row,*old_rows]
    if len({r['dataset_id'] for r in rows})!=len(rows):raise ValueError('Duplicate dataset')
    for old in old_rows:
        if digest(ROOT/old['path'])!=old['sha256']:raise ValueError('Inherited reference changed: '+old['dataset_id'])
        if old.get('validation_path') and digest(ROOT/old['validation_path'])!=old['validation_sha256']:
            raise ValueError('Inherited development reference changed: '+old['dataset_id'])
    baseline_sha=digest(ROOT/'checkpoints/shared_particle_hanging_v63/shared_strict.pt')
    if producer['development_comparison_policy']['old32_checkpoint_sha256']!=baseline_sha:
        raise ValueError('Inherited comparison checkpoint changed')
    manifest.update(status='NORMAL_COURSE_EXTENSION_PREPARED_NOT_TRAINED',scenes=rows,
        training_scope=validate_rows(rows,scope),current_course_official_demos=coverage['official_demos'],
        current_course_demo_count=len(coverage['official_demos']),current_course_dataset_count=len(rows),
        current_course_custom_pure_vbd_demos=['research.native_five_families'],
        current_course_custom_pure_vbd_demo_count=1,inherited_selected_official_demos=len(coverage['official_demos']),
        old32_demo_coverage_inherited=0,effective_model_config=configuration,model_configuration_overrides=overrides,
        development_baseline_policies=extend_policies(producer,rows,configuration),
        development_baseline_checkpoint_sha256=baseline_sha,
        selected_parent={'checkpoint':relative(trial),'checkpoint_sha256':digest(trial),
            'report':relative(parent/'training.json'),'report_sha256':digest(parent/'training.json'),
            'inference_checkpoint_sha256':evidence['checkpoint_sha256'],'selected_updates':limit},
        selected_export={'evidence':relative(args.selected_export),'evidence_sha256':digest(args.selected_export),
            'snapshot':relative(snapshot),'snapshot_sha256':digest(snapshot)})
    manifest['source_manifests_sha256'][relative(parent/'data_manifest.json')]=digest(parent/'data_manifest.json')
    manifest['new_dataset_preflight_sha256']={**manifest.get('new_dataset_preflight_sha256',{}),relative(args.preflight):digest(args.preflight)}
    manifest['resource_policy']={**manifest['resource_policy'],'host_self_trajectories':True,
        'host_encoder_checkpoint_inputs':True,'host_step_saved_tensors':True,
        'checkpoint_relation_ffn_rows':8192,'checkpoint_attention_blocks':True,'self_contact_broad_phase':'kdtree'}
    manifest['authored_reference_scope']={
        'source_contact_definition':metadata['source_self_contact_definition'],
        'source_force_equivalence_certified':False,'converged_teacher_certified':False,
        'policy':'Production-budget labels on an explicitly authored reciprocal topology. Parallel EE witnesses can differ from the native solver; source scope is not a quality certificate.'}
    manifest.pop('optimizer_learning_rate_migration',None)
    manifest['scope']='Retain all42 official pure-VBD demos and all66 parent datasets; add one authored five-family dataset to the same shared648 weights and selected Adam/RNG. Preparation is not training. All inherited caps, actual changed custom weights, native trajectory quality and realtime remain required.'
    args.report.parent.mkdir(parents=True,exist_ok=True)
    with args.report.open('x') as stream:json.dump(manifest,stream,indent=2,allow_nan=False);stream.write('\n')
    print(json.dumps({'status':manifest['status'],'official_demos':42,'custom_demos':1,
                      'datasets':len(rows),'new_optimizer_updates':0,'report':str(args.report)},indent=2))


if __name__=='__main__':main()
