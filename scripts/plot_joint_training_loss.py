"""Plot comparable fixed-window validation and component losses."""
import argparse
import json
from pathlib import Path
import os
os.environ['CUDA_VISIBLE_DEVICES']=''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--training-report',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads(args.training_report.read_text());history=report.get('loss_monitor_history',[])
    if not history:return
    datasets=sorted({key for row in history for key in row['dataset_means']})
    labels={'native_five_families_train_default':'Loaded five-family mixed',
        'native_json_ground_stiff_train':'JSON mu=lambda=1000 Pa',
        'native_json_ground_soft_bulk_train':'JSON mu=100, lambda=1000 Pa'}
    fig,axes=plt.subplots(1,2,figsize=(12,4.4))
    for dataset in datasets:
        pairs=[(r['after_update'],r['dataset_means'][dataset]) for r in history if dataset in r['dataset_means']]
        x,y=np.array(pairs).T
        if not np.isfinite(y).all() or np.any(y<0):raise ValueError('Finite nonnegative measured losses required')
        axes[0].plot(x,y,'o-',label=labels.get(dataset,dataset));axes[1].plot(x,y/max(y[0],1e-8),'o-',label=labels.get(dataset,dataset))
    axes[0].set(title='Fixed development-window loss',ylabel='Two-hop scaled state + contact-repair loss')
    axes[0].set_yscale('symlog',linthresh=1e-8)
    axes[1].set(title='Same fixed windows, relative to initial loss',ylabel='Loss / initial loss');axes[1].axhline(1,color='gray',linestyle='--',linewidth=1)
    schedules=report.get('optimizer_schedule_history',[report['optimizer_schedule']] if report.get('optimizer_schedule') else [])
    for schedule in schedules:
        rate=schedule['migration']['learning_rate'];extension=schedule.get('completed_course_extension')
        label=f'lr -> {rate:g}' if not extension else f"{extension['gradient_normalization']} gradients; full GPU"
        for axis in axes:axis.axvline(schedule['source_step'],color='gray',linestyle=':',label=label)
    for axis in axes:axis.set_xlabel('Cumulative optimizer updates');axis.grid(alpha=.25);axis.legend(fontsize=8)
    fig.suptitle('Shared648 model; fixed authored + official-family windows; full development at course end',fontsize=11)
    fig.tight_layout();args.output.parent.mkdir(parents=True,exist_ok=True)
    temporary=args.output.with_suffix('.tmp');fig.savefig(temporary,format='png',dpi=160);temporary.replace(args.output);plt.close(fig)
    result={'status':'ACTUAL_FIXED_DEVELOPMENT_LOSS_CURVES_WRITTEN','completed_updates':report['completed_updates'],
        'monitor_count':len(history),'datasets':datasets,'fixed_monitor_history':history,
        'convergence':report.get('loss_convergence'),
        'scope':'Fixed authored + representative official-family windows. Raw losses from different training samples are not a comparable convergence curve. Full all42 development remains separate.'}
    target=args.output.with_suffix('.json');temporary=target.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.replace(target)


if __name__=='__main__':main()
