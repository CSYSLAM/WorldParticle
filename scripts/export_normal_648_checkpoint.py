"""Freeze a verified selected648-wide training prefix for native viewers."""
import argparse
from collections import Counter
import hashlib
import io
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',default='shared_normal_648_v84')
    parser.add_argument('--allow-running-selected',action='store_true',help='Freeze an already eligible selected prefix while later training continues')
    parser.add_argument('--allow-failed-selected',action='store_true',help='Export only an eligible retained prefix of a failed stopped producer; preserve its failure evidence')
    parser.add_argument('--allow-interrupted-selected',action='store_true',help='Export the retained eligible prefix of an explicitly recorded course replan')
    args=parser.parse_args()
    import torch
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.training_scope import load_scope,changed_weight_coverage,CUSTOM_PURE_VBD,validate_rows
    directory=ROOT/'reports'/args.stage;checkpoints=ROOT/'checkpoints'/args.stage
    report_path=directory/'training.json';report_raw=report_path.read_bytes();report=json.loads(report_raw)
    permitted=('TRAINING_COMPLETED',)+(('RUNNING',) if args.allow_running_selected else ())+ (('FAILED_WITH_RETAINED_WEIGHTS',) if args.allow_failed_selected else ())+(('INTERRUPTED_FOR_COURSE_REPLAN',) if args.allow_interrupted_selected else ())
    if report['status'] not in permitted or report['selected_step']<=0:
        raise ValueError('An eligible selected trained prefix is required; live export must be explicit')
    if report['status'] in ('FAILED_WITH_RETAINED_WEIGHTS','INTERRUPTED_FOR_COURSE_REPLAN'):
        from wp_vbd.development_baseline import selected_parent_record
        selected_parent_record(report)
    manifest_path=directory/'data_manifest.json';manifest=json.loads(manifest_path.read_text())
    source=checkpoints/'shared_candidate.pt';source_raw=source.read_bytes();source_digest=hashlib.sha256(source_raw).hexdigest()
    stored=torch.load(io.BytesIO(source_raw),weights_only=True,map_location='cpu');del source_raw
    stored.pop('optimizer',None)
    # Producer replaces candidates and reports atomically, in that order.
    # Bind one complete weight snapshot to an eligible report prefix.
    for attempt in range(8):
        if any(row['update']==stored['step'] and row['eligible'] for row in report['validation_history']):break
        time.sleep(.5);report_raw=report_path.read_bytes();report=json.loads(report_raw)
    assert report['status'] in permitted
    assert stored['step']<=report['selected_step'] and stored['data_manifest_sha256']==digest(manifest_path)
    if not args.allow_running_selected:assert stored['step']==report['selected_step']
    assert any(row['update']==stored['step'] and row['eligible'] for row in report['validation_history'])
    if any('development_windows' in row for row in manifest['scenes']):
        from wp_vbd.development_cases import verify_case_metrics
        selected=next(row for row in report['validation_history'] if row['update']==stored['step'] and row['eligible'])
        verify_case_metrics(manifest['scenes'],selected['metrics'])
    samples=[s for s in report['optimization_samples'] if s['update']<=stored['step']]
    assert len(samples)==stored['step']
    for label,count in stored['sampled_updates'].items():assert sum(s['dataset']==label for s in samples)==count
    active={}
    for s in samples:
        if s['gradient_norm_before_clip']>0 and s['actual_max_gradient_scalar_changed']:
            active[s['dataset']]=active.get(s['dataset'],0)+1
    scope=load_scope(ROOT);coverage=changed_weight_coverage(samples,scope)
    demos=coverage['official_demos'];official=set(scope['target_demos'])
    official_sampled=Counter(s['demo'] for s in samples if s.get('demo') in official)
    official_active=Counter(s['demo'] for s in samples if s.get('demo') in official and s['gradient_norm_before_clip']>0 and s['actual_max_gradient_scalar_changed'])
    assert sorted(official_active)==demos
    validate_rows(manifest['scenes'],scope)
    assert demos==manifest['current_course_official_demos']
    required_custom={r['dataset_id'] for r in manifest['scenes'] if r.get('demo') in CUSTOM_PURE_VBD}
    assert required_custom<=set(coverage['custom_pure_vbd_datasets'])
    config={'name':args.stage,'model':stored['model_config'],'loss_reference_dt':1/120,
            'newton_sha':manifest['newton_sha'],'report_dir':str(directory),'checkpoint_dir':str(checkpoints)}
    model=MixedSolver(config['model']);model.load_state_dict(stored['model'])
    assert model.protected_response.width==648
    assert sum(p.numel() for p in model.protected_response.parameters())==46882811
    live=report['status']=='RUNNING'
    suffix=f'_step_{stored["step"]:06d}' if live else ''
    target=checkpoints/(f'shared{suffix}.pt' if live else 'shared_strict.pt')
    if target.exists():raise FileExistsError('Keep the frozen inference artifact immutable')
    snapshot=directory/f'training_export{suffix}_snapshot.json'
    if snapshot.exists():
        if snapshot.read_bytes()!=report_raw:raise FileExistsError(snapshot)
    else:snapshot.write_bytes(report_raw)
    report_digest=hashlib.sha256(report_raw).hexdigest()
    temporary=target.with_suffix('.tmp')
    torch.save({'architecture':stored['architecture'],'config':config,'model':stored['model'],
                'step':stored['step'],'sampled_updates':stored['sampled_updates'],
                'active_weight_optimizer_samples':{**active,'capacity_probe/mixed08':manifest['initial_probe_updates']},
                'sampled_official_updates':dict(official_sampled),'active_weight_official_optimizer_samples':dict(official_active),
                'initial_checkpoint_sha256':stored['initial_checkpoint_sha256'],
                'source_training_report':str(snapshot.relative_to(ROOT)),
                'source_training_report_sha256':report_digest,'source_training_checkpoint_sha256':source_digest,
                'data_manifest_sha256':digest(manifest_path),'trained_official_demos':demos,
                'trained_custom_pure_vbd_demos':coverage['custom_pure_vbd_demos'],
                'trained_custom_pure_vbd_datasets':coverage['custom_pure_vbd_datasets']},temporary)
    temporary.replace(target)
    evidence={'status':'SELECTED_648_CHECKPOINT_EXPORTED_FOR_NATIVE_TASK_CHECKS','checkpoint':str(target.relative_to(ROOT)),
              'checkpoint_sha256':digest(target),'source_trial_checkpoint':str(source.relative_to(ROOT)),
              'source_trial_checkpoint_sha256':source_digest,'source_training_report':str(snapshot.relative_to(ROOT)),
              'source_training_report_sha256':report_digest,'producer_training_was_running':live,
              'source_producer_status':report['status'],'source_producer_failure':report.get('failure'),
              'selected_updates':stored['step'],'completed_updates':report['completed_updates'],
              'trained_official_demo_count':len(demos),'trained_official_demos':demos,
              'trained_custom_pure_vbd_demos':coverage['custom_pure_vbd_demos'],
              'trained_custom_pure_vbd_demo_count':len(coverage['custom_pure_vbd_demos']),
              'trained_custom_pure_vbd_datasets':coverage['custom_pure_vbd_datasets'],
              'selected_positive_changed_weight_samples':sum(active.values()),'initial_nonofficial_probe_updates':manifest['initial_probe_updates'],
              'old32_demo_coverage_inherited':0,'active_neural_parameters':46882811,
              'source_producer_commit':report['source_git_commit'],'optimizer_state_in_inference_artifact':False,
              'resumed_valid_prefix':report.get('resume_evidence'),
              'inference_bytes':target.stat().st_size,'full_vbd_prediction_calls':0,
              'scope':'Selected actual changed-weight training prefix of one648-wide shared model, exported in the existing native-viewer schema. Full native behavior, independent material/topology tests, all42 pure-VBD targets and realtime remain required; nine external-solver demos are excluded by the current user scope.'}
    (directory/f'inference_checkpoint{suffix}.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(evidence,indent=2))


if __name__=='__main__':main()
