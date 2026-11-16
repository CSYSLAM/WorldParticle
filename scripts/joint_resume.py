"""Validate an actual recorded joint-course prefix before optimizer scheduling."""
import copy
import hashlib
import json
from pathlib import Path
import random


def recorded_prefix(checkpoint,report,manifest,schedule,*,checkpoint_sha,report_sha,manifest_sha,audit_only=False):
    if schedule.get('schema')!='recorded_joint_optimizer_schedule_v1':raise ValueError('Declared joint optimizer schedule required')
    for key,value in [('source_checkpoint_sha256',checkpoint_sha),('source_report_sha256',report_sha),('data_manifest_sha256',manifest_sha)]:
        if schedule.get(key)!=value:raise ValueError('Optimizer schedule source changed: '+key)
    if checkpoint['data_manifest_sha256']!=manifest_sha or checkpoint['initial_checkpoint_sha256']!=manifest['initial_checkpoint_sha256']:
        raise ValueError('Continue the identical admitted course and initialization')
    if checkpoint['model_config']!=report['config'] or checkpoint['model_config']!=manifest['effective_model_config']:
        raise ValueError('Optimizer scheduling may not change the model/physics configuration')
    step=checkpoint['step']
    if schedule.get('source_step')!=step or not 0<step<=report['completed_updates']:raise ValueError('Actual recorded checkpoint prefix required')
    extension=schedule.get('completed_course_extension')
    recovery=schedule.get('producer_exit_recovery')
    allowed_status='TRAINING_COMPLETED' if extension else 'INTERRUPTED_FOR_OPTIMIZER_SCHEDULE'
    if recovery:
        proof_path=Path(recovery['process_exit_evidence'])
        if hashlib.sha256(proof_path.read_bytes()).hexdigest()!=recovery['process_exit_evidence_sha256']:
            raise ValueError('Recovery process evidence changed')
        proof=json.loads(proof_path.read_text())
        if proof['source_report_sha256']!=report_sha or proof['source_checkpoint_sha256']!=checkpoint_sha:
            raise ValueError('Recovery evidence must bind the actual producer and complete checkpoint')
        if proof['source_checkpoint_step']!=step or proof['producer_recorded_updates']!=report['completed_updates']:
            raise ValueError('Recovery evidence prefix changed')
        for binding in proof['producer_processes']:
            proc=Path('/proc')/str(binding['pid'])
            if (proc/'cmdline').exists():
                argv=(proc/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
                if argv==binding['argv'] and (proc/'cwd').resolve()==Path(binding['cwd']):
                    raise ValueError('Cannot recover while the original producer is live')
        if report['status'] not in ('RUNNING','FAILED_WITH_RETAINED_WEIGHTS'):
            raise ValueError('Recovery retains the actual interrupted producer status')
        if recovery['producer_recorded_updates']!=report['completed_updates'] or not proof['all_producer_identities_missing']:
            raise ValueError('An observed terminal producer is required')
    if not audit_only and not recovery and (report['status']!=allowed_status or step!=report['completed_updates']):
        raise ValueError('Stop and record the complete source checkpoint without losing recorded updates')
    if report['selected_step']!=0:raise ValueError('Already selected models need their existing export flow')
    origin=report.get('joint_course_origin_updates',report['resumed_valid_updates'])
    if origin!=manifest['selected_parent']['selected_updates'] or not origin<step:raise ValueError('Keep the original selected course origin')
    if extension:
        count=len(manifest['scenes']);source_limit=origin+manifest['joint_course_policy']['epochs']*count
        additional=extension.get('additional_courses')
        if ((not recovery and step!=source_limit) or extension.get('source_completed_updates')!=source_limit
            or type(additional) is not int or not 1<=additional<=24
            or extension.get('target_updates')!=source_limit+additional*count
            or not extension.get('user_requested_full_gpu') or not extension.get('reason')):
            raise ValueError('Bind a real completed course and an explicit bounded full-GPU extension')
        if recovery and extension!=report['optimizer_schedule'].get('completed_course_extension'):
            raise ValueError('Interrupted recovery must retain the original declared extension')
    samples=copy.deepcopy([row for row in report['optimization_samples'] if row['update']<=step])
    counts=copy.deepcopy(checkpoint['sampled_updates'])
    if set(counts)!={row['dataset_id'] for row in manifest['scenes']} or sum(counts.values())!=step:
        raise ValueError('Every original dataset and optimizer call must survive continuation')
    if [row['update'] for row in samples]!=list(range(1,step+1)) or any(sum(s['dataset']==key for s in samples)!=value for key,value in counts.items()):
        raise ValueError('Checkpoint counts disagree with actual recorded optimization samples')
    random.Random().setstate(checkpoint['rng_state'])
    monitors=copy.deepcopy([row for row in report['loss_monitor_history'] if row['after_update']<=step])
    if not monitors or monitors[0]['after_update']!=origin or any((row['after_update']-origin)%len(counts) for row in monitors):
        raise ValueError('Keep the real baseline and complete-course fixed validation history')
    lowest=copy.deepcopy(report.get('lowest_fixed_validation'))
    if lowest and lowest['update']>step:raise ValueError('A later candidate may not be silently inherited by an earlier prefix')
    return {'origin':origin,'step':step,'samples':samples,'counts':counts,'monitors':monitors,'lowest':lowest,
        'replay_samples':copy.deepcopy([row for row in report['optimization_samples'] if row['update']>step]) if recovery else [],
        'prior_schedules':copy.deepcopy(report.get('optimizer_schedule_history',
            [report['optimizer_schedule']] if report.get('optimizer_schedule') else [])),
        'validation_history':copy.deepcopy([row for row in report['validation_history'] if row['update']<=step]),
        'baseline_policies':copy.deepcopy(report['development_comparison_policy']['per_demo_physical_overrides']),
        'schedule':copy.deepcopy(schedule),'later_source_calls_not_inherited':report['completed_updates']-step,
        'audit_only':audit_only}


def verify_extension_arguments(schedule,args):
    extension=schedule.get('completed_course_extension')
    if not extension:
        if args.gradient_normalization!='clip':raise ValueError('Gradient normalization requires a declared new course')
        return
    for key,value in [('target_updates',args.updates),('gpu_memory_gib',args.gpu_memory_gib),
                      ('duty_cycle',args.duty_cycle),('gradient_normalization',args.gradient_normalization)]:
        if extension.get(key)!=value:raise ValueError('Declared completed-course extension differs: '+key)
