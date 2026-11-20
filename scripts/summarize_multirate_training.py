"""Compare fixed macro-step development windows before and after training."""
import argparse
from collections import defaultdict
import json
from pathlib import Path


def identity(row):
    return row['dataset'],row['start'],tuple(row['strides']),tuple(row['durations'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=json.loads(args.training.read_text())
    validations=report['validation']
    initial={identity(row):row for row in validations[0]['windows'] if row['learned']}
    physical={identity(row):row for row in validations[0]['windows'] if not row['learned']}
    final={identity(row):row for row in validations[-1]['windows'] if row['learned']}
    groups=defaultdict(list)
    for key,after in final.items():
        before=initial[key];baseline=physical.get(key,{})
        item={'start':key[1],'durations':list(key[3]),
              'initial_loss':before.get('loss'),'final_loss':after.get('loss'),
              'physical_loss':baseline.get('loss'),'rejected':after.get('rejected')}
        if 'loss' in before and 'loss' in after:
            item['relative_loss_change']=(after['loss']-before['loss'])/max(before['loss'],1e-12)
        for label,row in (('initial',before),('final',after),('physical',baseline)):
            hops=row.get('hops',[])
            duration=sum(hop['dt_s'] for hop in hops)
            metrics={}
            if duration:
                keys=set.intersection(*(set(hop['metrics']) for hop in hops))
                metrics={name:sum(hop['metrics'][name]*hop['dt_s'] for hop in hops)/duration for name in keys}
            item[label+'_metrics']=metrics
        groups[key[0]].append(item)
    datasets=[]
    for dataset,windows in groups.items():
        paired=[w for w in windows if 'relative_loss_change' in w]
        aggregate={}
        for label in ('initial','final','physical'):
            values=[w[label+'_loss'] for w in paired if w[label+'_loss'] is not None]
            aggregate[label+'_mean_loss']=sum(values)/len(values) if values else None
        change=(aggregate['final_mean_loss']/max(aggregate['initial_mean_loss'],1e-12)-1) if paired else None
        datasets.append({'dataset':dataset,**aggregate,'relative_mean_loss_change':change,
                         'paired_windows':len(paired),'rejected_final_windows':sum(w['rejected'] is not None for w in windows),
                         'windows':windows})
    result={'training_status':report['status'],'optimizer_updates':report['completed_updates'],
            'gpu':report['gpu'],'datasets':datasets,
            'scope':'Fixed development windows, paired by state start and physical duration. Training loss across unrelated windows is not a convergence comparison; full independent inference is assessed separately.'}
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    names=[row['dataset'].split('/')[0].replace('_train_default','').replace('_multirate_20261118','') for row in datasets]
    figure,axes=plt.subplots(1,2,figsize=(16,7),layout='constrained')
    for label,color,offset in (('initial','#748cab',-.25),('final','#207567',0),('physical','#c88635',.25)):
        axes[0].barh([i+offset for i in range(len(datasets))],
                     [row[label+'_mean_loss'] or 1e-12 for row in datasets],height=.24,label=label,color=color)
    axes[0].set(yticks=range(len(names)),yticklabels=names,xlabel='Mean development loss (log scale)',title='Same windows and physical spans')
    axes[0].set_xscale('log');axes[0].legend();axes[0].grid(axis='x',alpha=.2)
    for dataset in groups:
        steps=[];values=[]
        for checkpoint in validations:
            valid=[row['loss'] for row in checkpoint['windows'] if row['dataset']==dataset and row['learned'] and 'loss'in row]
            if valid:steps.append(checkpoint['after_update']);values.append(sum(valid)/len(valid))
        if values:
            axes[1].plot(steps,[v/max(values[0],1e-12) for v in values],marker='o',label=dataset.split('/')[0])
    axes[1].axhline(1,color='gray',linestyle='--',linewidth=1)
    axes[1].set(xlabel='Optimizer updates',ylabel='Development loss / initial loss',title='Fixed-window change')
    axes[1].legend(fontsize=7);axes[1].grid(alpha=.2)
    figure.savefig(args.output/'validation.png',dpi=160);plt.close(figure)
    print(json.dumps({k:v for k,v in result.items() if k!='datasets'},indent=2))
    for row in datasets:
        print(json.dumps({k:v for k,v in row.items() if k!='windows'}))


if __name__=='__main__':main()
