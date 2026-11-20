"""Audit all standalone VBD trajectories and index their complete recorded timeline."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024**2),b''):value.update(block)
    return value.hexdigest()


def main():
    import numpy as np
    from wp_vbd.multirate_window import reference_grid,command_events
    from wp_vbd.full_demo_course import timeline_catalog,authored_frame_horizon
    from wp_vbd.newton_source import source_root,newton_revision
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--replacements',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hops',type=int,default=2)
    a=p.parse_args()
    scope=json.loads((ROOT/'configs/pure_vbd_demo_scope.json').read_text())
    source=json.loads(a.manifest.read_text())
    if source['newton_sha']!=scope['newton_sha']:raise ValueError('Pinned teacher revision changed')
    if newton_revision()!=scope['newton_sha']:raise ValueError('Select the pinned Newton checkout for authored horizon checks')
    candidates={demo:[] for demo in scope['target_demos']}
    for row in source['scenes']:
        if row.get('demo') in candidates:candidates[row['demo']].append(row)
    if a.replacements:
        for row in json.loads(a.replacements.read_text())['scenes']:
            if row['demo'] not in candidates:raise ValueError('External or unknown demo in replacement data')
            candidates[row['demo']].insert(0,row)
    admitted=[];audit=[]
    for demo,options in candidates.items():
        reasons=[];selected=None
        for row in options:
            path=Path(row['path']);path=path if path.is_absolute() else ROOT/path
            if not path.exists():reasons.append('Missing '+str(path));continue
            grid,metadata=reference_grid(path)
            if len(grid.segments)!=1 or metadata.get('data_layout')=='teacher_queried_windows_v1':
                reasons.append('Queried short windows are supplementary, not a continuous demo');continue
            if metadata.get('reference_final_check','').startswith('FAIL'):
                reasons.append('Source terminal check failed: '+metadata['reference_final_check'][:240]);continue
            seconds=float(grid.times[-1]-grid.times[0])
            category,module=demo.split('.')
            definition=source_root()/'newton/examples'/category/(module+'.py')
            if digest(definition)!=scope['source_evidence'][demo]['sha256']:
                raise ValueError('Authored demo definition changed: '+demo)
            authored_frames=authored_frame_horizon(definition)
            teacher_frames=metadata.get('teacher_frames')
            if not teacher_frames:
                reasons.append('Full recorded frame count missing');continue
            frame_dt=seconds/teacher_frames
            minimum=max(authored_frames*frame_dt,{'cloth.example_cloth_twist':10.0}.get(demo,0.))
            if seconds+1e-8<minimum:
                reasons.append(f'Only {seconds:g}s; authored motion requires at least {minimum:g}s');continue
            actual=digest(path)
            if actual!=row['sha256']:raise ValueError('Reference bytes changed: '+str(path))
            if metadata.get('source_sha',metadata.get('newton_sha',scope['newton_sha']))!=scope['newton_sha']:
                raise ValueError('Teacher revision mismatch: '+demo)
            events=command_events(path,len(grid.times),hard_only=True)
            # This authored demo uses centimetre-scale coordinates and an
            # initial native interval can move ~0.765 world units. Retain
            # its full normal-resolution trajectory and native h until the
            # large-step path is separately qualified. This exception is
            # explicit; it is not counted as a successful macro-step demo.
            native_only=demo=='cloth.example_cloth_rollers'
            catalog=timeline_catalog(grid,events=events,hops=a.hops,
                **({'target_dts':(float(metadata['dt']),)} if native_only else {}))
            if catalog['missing_phase_splits']:
                reasons.append('Incomplete timeline train/development coverage');continue
            selected={**row,'path':str(path.resolve()),'reference_dt':metadata['dt'],
                      'multirate':catalog,'full_demo_timeline':True,
                      'student_step_policy':'authored_native_h_pending_macro_qualification' if native_only else 'mixed_real_endpoint_durations'}
            # Old two-step qualification reports do not qualify an expanded
            # timeline. Its admission is tracked by this new byte-bound catalog
            # plus the runtime source contract and actual gradient preflight.
            for key in ('qualification','qualification_sha256','qualified_train_starts',
                        'validation_starts','development_windows','validation_path',
                        'validation_sha256','admitted_file_stat'):
                selected.pop(key,None)
            selected['timeline_provenance']={'teacher_states':len(grid.times),'physical_seconds':seconds,
                'teacher_frames':metadata.get('teacher_frames'),'teacher_iterations':metadata.get('teacher_iterations'),
                'source_terminal_check':metadata.get('reference_final_check','NOT_RECORDED'),
                'hard_events':len(events),'full_recorded_episode':True,
                'authored_motion_horizon_required_s':minimum,'authored_default_frames':authored_frames,
                'authored_source_sha256':digest(definition),
                'all_authored_tasks_completed_certified':False}
            break
        if selected:admitted.append(selected)
        audit.append({'demo':demo,'status':'INDEXED_PENDING_GRADIENT_PREFLIGHT' if selected else 'NEEDS_CONTINUOUS_REFERENCE',
                      'dataset':selected['dataset_id'] if selected else None,'attempt_notes':reasons,
                      'timeline':selected['timeline_provenance'] if selected else None})
    result={'schema':'full_pure_vbd_timeline_course_v1','newton_sha':scope['newton_sha'],
        'status':'INDEXED_PENDING_GRADIENT_PREFLIGHT' if len(admitted)==42 else 'INCOMPLETE_REFERENCES',
        'source_manifest':str(a.manifest.resolve()),'source_manifest_sha256':digest(a.manifest),
        'target_demo_count':42,'indexed_demo_count':len(admitted),'excluded_external_demo_count':9,
        'phase_bins':6,'horizon':a.hops,'scenes':admitted,'audit':audit,
        'scope':'Every recorded state is retained. Bounded autonomous windows rotate across all six timeline parts. '
                'Indexing alone is not optimization coverage or a guarantee of solver convergence/task completion.'}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':result['status'],'indexed':len(admitted),'pending':[r for r in audit if not r['dataset']]},indent=2))


if __name__=='__main__':main()
