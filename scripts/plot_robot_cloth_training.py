"""Plot fixed development windows; random training phases are not trend lines."""
import argparse
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--training',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    report=json.loads(a.training.read_text());rows=report['validation']
    if not rows:raise ValueError('At least one fixed development evaluation required')
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    steps=[r['update'] for r in rows]
    for key,label,color in [('cloth_loss','Held-out cloth material','#1672b6'),
                            ('replay_loss','Existing object families','#d47819')]:
        y=[r[key] for r in rows]
        axes[0].plot(steps,y,'o-',label=label,color=color)
        axes[1].plot(steps,[100*(v/y[0]-1) for v in y],'o-',label=label,color=color)
    axes[0].set(xlabel='Completed optimizer updates',ylabel='Fixed-window objective',ylim=(0,None))
    axes[1].set(xlabel='Completed optimizer updates',ylabel='Change from frozen parent (%)')
    axes[1].axhline(0,color='.65',linewidth=.8)
    for ax in axes:ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.suptitle('Shared-backbone cloth fine-tuning | 3 substeps/frame, 1 NN/step')
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=160)


if __name__=='__main__':main()
