"""Select completed all-demo checkpoints using fixed development windows only."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']=''


def main():
    import torch
    from prepare_full_demo_course import digest
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    training=json.loads(a.training.read_text())
    if training['status']!='COMPLETED':raise ValueError('Finish all requested optimizer updates before selection')
    suites=('fixed_10_development','fixed_42_official_temporal_development')
    validation={(r['update'],r['suite']):r['windows'] for r in training['validation']}
    base={suite:{w['dataset']:w for w in validation[0,suite]} for suite in suites}
    expected=(10,42)
    candidates=[]
    for path in sorted(a.training.parent.glob('step_*.pt')):
        saved=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        step=saved['step'];summary=saved['training_summary'];reasons=[]
        coverage=summary['official_timeline_bins']
        if len(coverage)!=42 or any(set(bins)!=set(map(str,range(6))) for bins in coverage.values()):
            reasons.append('All six timeline parts of all42 demos have not yet participated')
        if len(summary['official_positive_gradient_microbatches'])!=42 or any(v<1 for v in summary['official_positive_gradient_microbatches'].values()):
            reasons.append('Missing measured positive gradients for an official demo')
        scores={}
        for suite,count in zip(suites,expected):
            windows=validation.get((step,suite),[])
            if len(windows)!=count or any('loss' not in w or not math.isfinite(w['loss']) for w in windows):
                reasons.append('Missing, rejected or nonfinite fixed development windows: '+suite);continue
            ratios=[]
            for window in windows:
                before=base[suite].get(window['dataset'])
                if before is None or 'loss' not in before or not math.isfinite(before['loss']):
                    reasons.append('Baseline development window not accepted: '+window['dataset']);continue
                # Equal-case relative scores avoid the large-coordinate
                # rollers loss eclipsing every robot/small-material example.
                if window['detail']!=before['detail']:
                    # Costs and load timing depend on solver execution; check
                    # only the identity of the actual teacher interval.
                    keys=('start','stride','strides','durations','dt','hops','teacher_end')
                    if any(window['detail'].get(k)!=before['detail'].get(k) for k in keys):
                        raise ValueError('Development comparison changed teacher interval')
                ratios.append(math.log((window['loss']+1e-3)/(before['loss']+1e-3)))
            if len(ratios)==count:scores[suite]=sum(ratios)/count
        candidates.append({'step':step,'checkpoint':str(path.resolve()),'eligible':not reasons,
            'reasons':reasons,'mean_log_relative_loss_by_suite':scores,
            'score':sum(scores.values())/len(suites) if len(scores)==len(suites) else None,
            'sha256':digest(path)})
    eligible=[row for row in candidates if row['eligible'] and row['score'] is not None]
    selected=min(eligible,key=lambda r:(r['score'],-r['step'])) if eligible else None
    result={'status':'DEVELOPMENT_CANDIDATE_SELECTED' if selected else 'NO_FULL_COVERAGE_DEVELOPMENT_ELIGIBLE_CANDIDATE',
        'selected':selected,'candidates':candidates,'completed_optimizer_updates':training['completed_updates'],
        'scope':'Selection on fixed independent augmentation and seen-asset temporal development windows, '
                'equal weighting by case and suite with 1e-3 loss floor. This is not a convergence or full-scene-quality certificate.'}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    if selected:
        target=a.output.parent/'selected.pt'
        if target.exists():raise FileExistsError('Preserve previous model selection')
        os.link(selected['checkpoint'],target)
    print(json.dumps(result,indent=2))
    if not selected:raise RuntimeError('No checkpoint qualifies for review; inspect each rejected development window')


if __name__=='__main__':main()
