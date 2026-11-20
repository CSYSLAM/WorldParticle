"""Plot the pilot without mixing changing training windows with fixed validation."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training',type=Path,required=True)
    p.add_argument('--ablation',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();d=json.loads(a.training.read_text());ab=json.loads(a.ablation.read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'axes.titleweight':'bold'})
    official=[r for r in d['validation'] if r['suite']=='fixed_42_official_temporal_development']
    fig,axes=plt.subplots(1,4 if official else 3,figsize=(21 if official else 16,4.8),layout='constrained')
    x=np.array([r['update'] for r in d['training_samples']]);y=np.array([r['loss'] for r in d['training_samples']])
    median=np.array([np.median(y[max(0,i-19):i+1]) for i in range(len(y))])
    axes[0].plot(x,y,color='#9fb2c6',alpha=.6,lw=.8,label='Changing training windows')
    axes[0].plot(x,median,color='#137f86',lw=2.3,label='20-update median')
    axes[0].set(yscale='log',xlabel='Optimizer update',ylabel='Weighted rollout loss',title='Training / mixed scene families')
    axes[0].legend(fontsize=8)
    rows=[r for r in d['validation'] if r['suite']=='fixed_10_development']
    vx=[r['update'] for r in rows];vy=[np.mean([w['loss'] for w in r['windows'] if 'loss' in w]) for r in rows]
    axes[1].plot(vx,vy,'o-',color='#da8232',lw=2.3)
    for tx,ty in zip(vx,vy):axes[1].annotate(f'{ty:.4f}',(tx,ty),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
    improvement=100*(1-vy[-1]/vy[0]);axes[1].set(xlabel='Optimizer update',ylabel='Mean weighted loss',
        title=f'Fixed independent development / {improvement:.2f}% lower')
    axes[1].margins(y=.3)
    rounds=ab.get('physical_response_rounds_per_hop',int(d['model_config'].get('analytic_predictor_rounds',0))+int(d['model_config'].get('unroll',1)))
    selected=f'step {ab["checkpoint_step"]}' if 'checkpoint_step' in ab else 'corrector'
    values=list(ab['mean_loss'].values());labels=[f'Analytic only\n{rounds} physical rounds','Initial shared\ncorrector',f'Trained shared\n{selected}']
    bars=axes[-1].bar(labels,values,color=['#a7b4c2','#4b769a','#137f86'],width=.65)
    for bar,value in zip(bars,values):axes[-1].text(bar.get_x()+bar.get_width()/2,value+.015,f'{value:.4f}',ha='center')
    axes[-1].set(ylabel='Mean weighted loss',title='Paired held-out windows / CPU ablation',ylim=(0,max(values)*1.15))
    if official:
        ox=[r['update'] for r in official]
        base={w['dataset']:w for w in official[0]['windows']}
        oy=[np.exp(np.mean([np.log((w['loss']+1e-3)/(base[w['dataset']]['loss']+1e-3)) for w in r['windows']]))
            if all('loss' in w and 'loss' in base[w['dataset']] for w in r['windows']) else np.nan for r in official]
        axes[2].plot(ox,oy,'s-',color='#7655ab',lw=2.3)
        for row,tx,ty in zip(official,ox,oy):
            accepted=sum('loss' in w for w in row['windows'])
            label=f'{ty:.4f}' if np.isfinite(ty) else f'{accepted}/42 accepted'
            axes[2].annotate(label,(tx,ty if np.isfinite(ty) else 0),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
        axes[2].axhline(1.,ls='--',lw=1,color='#75808a')
        axes[2].set(xlabel='Optimizer update',ylabel='Equal-case relative loss (initial = 1)',title='Fixed 42 official temporal windows')
        axes[2].margins(y=.3)
    for ax in axes:ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.suptitle(f'Shared analytic predictor + learned corrector | {d["completed_updates"]} optimizer updates',fontsize=15,fontweight='bold')
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=180);fig.savefig(a.output.with_suffix('.svg'));plt.close(fig)


if __name__=='__main__':main()
