"""Bound selected-parent and authored-JSON admission for material extensions."""
import hashlib
import json
from pathlib import Path

from .development_baseline import selected_parent_record
from .development_cases import verify_case_metrics
from .native_mixed_course import verify_preflight
from .training_scope import changed_weight_coverage,validate_rows,CUSTOM_PURE_VBD

PARAMETER_WINDOWS=(0,1,180,600,900,1800,2800)


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def selected_mixed_parent(producer,rows,scope):
    selected=selected_parent_record(producer)
    coverage=changed_weight_coverage(producer['optimization_samples'],scope,selected_step=producer['selected_step'])
    scheduled=validate_rows(rows,scope)
    if set(coverage['official_demos'])!=set(scope['target_demos']) or scheduled['scheduled_official_demos']!=coverage['official_demos']:
        raise ValueError('Selected actual parent must cover every official pure-VBD demo')
    required={r['dataset_id'] for r in rows if r.get('demo') in CUSTOM_PURE_VBD}
    if not required or not required<=set(coverage['custom_pure_vbd_datasets']):
        raise ValueError('Selected parent must actually change weights on every inherited authored dataset')
    verify_case_metrics(rows,selected['metrics'])
    if not all(o['within_cap'] for row in selected['metrics'] for o in row['objects']):
        raise ValueError('Every inherited development object must retain its cap')
    return selected,coverage


def qualified_candidate(row,root,expected_newton):
    """Bind the real full history and native factory without device imports."""
    import numpy as np
    root=Path(root);data=root/row['path'];qualification=root/row['qualification']
    if digest(data)!=row['sha256'] or digest(qualification)!=row['qualification_sha256']:
        raise ValueError('Exact authored dataset and qualification hashes required')
    report=json.loads(qualification.read_text())
    if report.get('status')!='NATIVE_MIXED_CAUSAL_HISTORY_AND_ALL_REFERENCE_PATHS_QUALIFIED' or report.get('output_sha256')!=row['sha256']:
        raise ValueError('Complete actual authored qualification required')
    if not all(report.get(k) is True for k in ('eligible_normal_course','eligible_normal_course_length','complete_reference_material_and_VF_EE_paths_verified')):
        raise ValueError('Complete compatible material and VF/EE paths required')
    if report.get('teacher_substeps')!=3000 or report.get('full_solver_or_shape_collision_calls')!={'construct':0,'step':0,'shape_collision':0}:
        raise ValueError('Five-second history qualification may not run a replacement solver')
    if report.get('script_sha256')!=digest(root/'scripts/prepare_elastic_rod_tape.py'):
        raise ValueError('Requalify with the current accepted-history verifier')
    with np.load(data,allow_pickle=False) as archive:
        meta=json.loads(str(archive['metadata']))
        if any(archive[k].shape[0]!=3001 for k in ('x','q','v','omega')):raise ValueError('All3000 accepted steps required')
        source_fields={k[8:] for k in archive.files if k.startswith('source__')}-{'joint_target_q_start'}
    if report.get('source_numeric_fields_verified')!=len(source_fields):raise ValueError('Complete native static-field reconstruction required')
    if meta.get('contract')!='elastic_rod_surface_joint_self_contact_v222' or meta.get('training_schema')!='accepted_elastic_rod_inputs_and_history_v226':
        raise ValueError('Explicit accepted elastic/rod state and history contract required')
    if meta.get('newton_sha')!=expected_newton or meta.get('official_demo')!=row['demo']:
        raise ValueError('Pinned native source and authored identity required')
    if not all(meta.get(k) is True for k in ('eligible_normal_course','eligible_normal_course_length','reference_format_qualified')):
        raise ValueError('Qualified normal data metadata required')
    if meta.get('teacher_substeps')!=3000 or meta.get('dt')!=1/600 or meta.get('self_contact_motion_budget')!=.02:
        raise ValueError('The complete measured timestep and motion envelope must be retained')
    definition=meta.get('asset_definition',{})
    if definition.get('schema')!='resolved_native_shared_asset_training_definition_v254' or definition.get('asset_schema')!='native_shared_asset_v1':
        raise ValueError('Embedded authored JSON definition required, not the old fixed fixture')
    factory=digest(root/'src/wp_vbd/native_assets.py')
    if definition.get('builder_sha256')!=factory or meta.get('factory_sha256')!=factory:
        raise ValueError('The actual authored builder changed')
    if definition.get('solver_configuration')!=meta.get('solver_configuration'):
        raise ValueError('Authored data solver policy changed')
    if meta['solver_configuration'].get('integrate_with_external_rigid_solver',False):
        raise ValueError('Only standalone VBD data are in scope')
    topology=meta.get('source_self_contact_definition',{})
    if topology!=report.get('source_self_contact_definition') or topology.get('compatible') is not True:
        raise ValueError('Qualified reciprocal source topology required')
    trace=Path(meta['source_trace_path'])/'manifest.json'
    if digest(trace)!=meta['source_trace_manifest_sha256']:raise ValueError('Actual native trace changed')
    recorded=json.loads(trace.read_text())
    if recorded.get('status')!='PHYSICAL_SUBSTEP_TRACE_READY' or recorded.get('failure') is not None or recorded.get('source_sha')!=expected_newton:
        raise ValueError('Complete pinned original trace required')
    if recorded.get('vbd_instances')!=1 or recorded.get('physical_vbd_calls')!=3000 or recorded.get('reference_step_calls')!=3000:
        raise ValueError('Exactly one native offline source and all physical calls required')
    if recorded.get('asset_definition')!=definition or recorded.get('collector_sha256')!=digest(root/'scripts/collect_elastic_rod_substeps.py'):
        raise ValueError('Trace must bind the exact embedded definition and collector')
    if recorded.get('configuration')!=meta['solver_configuration']:raise ValueError('Reference policy differs from qualified source')
    if recorded.get('native_solver_iterations')!=meta.get('teacher_iterations'):
        raise ValueError('Actual reference iteration budget changed')
    static=trace.parent/recorded['static']['path']
    if digest(static)!=recorded['static']['sha256']:raise ValueError('Original native static archive changed')
    if sum(chunk['steps'] for chunk in recorded['chunks'])!=3000:raise ValueError('Complete original trace chunks required')
    for chunk in recorded['chunks']:
        path=trace.parent/chunk['path']
        if path.stat().st_size!=chunk['bytes'] or digest(path)!=chunk['sha256']:
            raise ValueError('Original native trace chunk changed')
    for path,sha in definition['mesh_sha256'].items():
        if digest(path)!=sha:raise ValueError('Referenced native training mesh changed')
    soft=[o for o in definition['resolved_spec']['objects'] if o['family']=='softbody']
    parameters=row['physical_parameters']
    if len(soft)!=1 or soft[0]['material']['mu']!=parameters['mu_Pa'] or soft[0]['material']['lambda']!=parameters['lambda_Pa']:
        raise ValueError('Candidate material labels must match the actual authored softbody')
    return meta


def verify_parameter_preflight(proof,**bindings):
    return verify_preflight(proof,required_windows=PARAMETER_WINDOWS,episode_interactions=True,**bindings)
