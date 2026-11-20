"""Verify actual optimizer coverage against a byte-bound whole-demo course."""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']=''


def main():
    import torch
    from prepare_full_demo_course import digest
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1)
    training=json.loads(a.training.read_text());manifest=json.loads(a.manifest.read_text())
    checkpoint=torch.load(a.checkpoint,weights_only=True,map_location='cpu',mmap=True)
    bound=hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    if training['official_manifest_sha256']!=bound or checkpoint['training_summary']['official_manifest_sha256']!=bound:
        raise ValueError('Checkpoint, producer and course bytes must agree')
    if training['status'] not in ('COMPLETED','STOPPED_BY_USER_AT_PLATEAU') or training['model_config']!=checkpoint['model_config']:
        raise ValueError('A completed or explicitly user-stopped matching training producer required')
    step=checkpoint['step'];samples=[s for s in training['training_samples'] if s['update']<=step]
    if [s['update'] for s in samples]!=list(range(1,step+1)):raise ValueError('Incomplete optimizer prefix')
    rows={r['demo']:r for r in manifest['scenes']}
    scope=json.loads((ROOT/'configs/pure_vbd_demo_scope.json').read_text())
    if set(rows)!=set(scope['target_demos']):raise ValueError('Expected exactly the 42 standalone VBD demos')
    counts=Counter();effective=Counter();bins=defaultdict(set)
    augmentation=Counter()
    for sample in samples:
        for micro in sample['microbatches']:
            if micro['dataset'].startswith('aug_'):augmentation[micro['dataset']]+=1
            demo=micro.get('demo')
            if demo is None:continue
            if demo not in rows:raise ValueError('External or unregistered optimizer data')
            row=rows[demo];counts[demo]+=1;bins[demo].add(micro['phase_bin'])
            if micro['positive_measured_gradient'] and sample['actual_shared_weight_probe_changed']:effective[demo]+=1
            for block in row['multirate']['reserved_development_blocks']:
                if not (micro['teacher_end']<block['first'] or micro['start']>block['last']):
                    raise ValueError('Training crossed a reserved development state')
    summary=checkpoint['training_summary']
    if dict(counts)!=summary['official_demo_microbatches']:
        raise ValueError('Checkpoint sampling counters disagree with its recorded optimizer prefix')
    finite=all(bool(torch.isfinite(value).all()) for value in checkpoint['model'].values() if value.is_floating_point())
    if not finite:raise ValueError('Nonfinite checkpoint')
    for row in rows.values():
        if digest(row['path'])!=row['sha256']:raise ValueError('Teacher reference changed after training')
    per_demo=[{'demo':demo,'successful_microbatches':counts[demo],
               'positive_gradient_and_changed_weight_updates':effective[demo],
               'observed_phase_bins':sorted(bins[demo]),
               'recorded_seconds':rows[demo]['timeline_provenance']['physical_seconds']} for demo in sorted(rows)]
    complete=all(counts[d]>0 and effective[d]>0 and bins[d]==set(range(6)) for d in rows)
    result={'status':'PASS' if complete else 'INCOMPLETE_ACTUAL_COVERAGE','checkpoint_step':step,
        'checkpoint_sha256':digest(a.checkpoint),'finite_weights':finite,'official_manifest_sha256':bound,
        'training_status':training['status'],'producer_completed_updates':training['completed_updates'],
        'target_demos':42,'actual_positive_gradient_and_changed_weight_demos':sum(effective[d]>0 for d in rows),
        'all_six_phase_bins_per_demo':all(bins[d]==set(range(6)) for d in rows),
        'actual_official_microbatches':sum(counts.values()),'excluded_external_demos':9,'per_demo':per_demo,
        'selected_prefix_distinct_augmentation_trajectories':len(augmentation),
        'selected_prefix_augmentation_microbatches':sum(augmentation.values()),
        'native_dynamics_calls_during_learning':training['native_solver_calls_in_learning'],
        'full_task_and_realtime_quality_accepted':False,
        'scope':'Optimizer participation and temporal split evidence only. A mixed update changes shared weights jointly; '
                'this does not isolate each microbatch causal effect or certify full-scene inference quality.'}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='per_demo'},indent=2))


if __name__=='__main__':main()
