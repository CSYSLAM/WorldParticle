"""Review comparable fixed-window validation losses, with an explicit rule."""
import argparse
import json
from pathlib import Path


def analyze(history):
    result={'status':'INSUFFICIENT_FIXED_VALIDATION_CHECKPOINTS','converged':False,'minimum_postbaseline_monitors':4,
        'rule':'At least four complete courses; last three balanced fixed-window means span <=2%, no dataset grows >5% across that tail, tail mean improves >=1% from baseline and stays within5% of the best observed score. An observed validation plateau is not a proof of physical correctness or a global optimum.'}
    if not history:return result
    keys=sorted(history[0]['dataset_means'])
    if any(sorted(row['dataset_means'])!=keys for row in history):raise ValueError('Fixed validation datasets changed')
    floors={k:max(float(history[0]['dataset_means'][k]),1e-8) for k in keys}
    scores=[sum(float(row['dataset_means'][k])/floors[k] for k in keys)/len(keys) for row in history]
    normalization=max(scores[0],1e-30);scores=[s/normalization for s in scores]
    result.update(balanced_relative_scores=scores,updates=[r['after_update'] for r in history])
    if len(history)>1:
        best=min(range(1,len(scores)),key=scores.__getitem__)
        result.update(best_monitor_update=history[best]['after_update'],best_relative_score=scores[best])
    if len(history)<5:return result
    tail=scores[-3:];spread=(max(tail)-min(tail))/max(abs(sum(tail)/3),1e-12)
    growth=max((float(history[-1]['dataset_means'][k])-float(history[-3]['dataset_means'][k]))/
        max(float(history[-3]['dataset_means'][k]),1e-8) for k in keys)
    mean_tail=sum(tail)/3;improvement=1-mean_tail
    rebound=(mean_tail-min(scores[1:]))/max(abs(min(scores[1:])),1e-8)
    plateau=spread<=.02 and growth<=.05 and rebound<=.05
    result.update(tail_relative_spread=spread,maximum_dataset_tail_growth=growth,relative_improvement=improvement,
        tail_mean_rebound_from_best=rebound,
        converged=bool(plateau and improvement>=.01),status='OBSERVED_FIXED_VALIDATION_PLATEAU' if plateau and improvement>=.01 else
        'PLATEAU_WITHOUT_VERIFIED_IMPROVEMENT' if plateau else 'FIXED_VALIDATION_LOSS_STILL_CHANGING')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    result=analyze(json.loads(args.training_report.read_text()).get('loss_monitor_history',[]))
    args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
