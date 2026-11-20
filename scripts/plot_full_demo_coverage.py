"""Plot actual selected-checkpoint coverage, separately from reference registration."""
import argparse
import json
from pathlib import Path


def main():
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    audit=json.loads(a.audit.read_text());rows=audit['per_demo']
    matrix=np.zeros((len(rows),6))
    for index,row in enumerate(rows):matrix[index,row['observed_phase_bins']]=1
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(14,13),width_ratios=[2.3,1,1],layout='constrained',sharey=True)
    axes[0].imshow(matrix,aspect='auto',cmap=matplotlib.colors.ListedColormap(['#e5e9ee','#198c87']),vmin=0,vmax=1)
    axes[0].set_yticks(range(len(rows)),[row['demo'] for row in rows],fontsize=7)
    axes[0].set_xticks(range(6),[f'{i+1}/6' for i in range(6)])
    axes[0].set(xlabel='Recorded motion timeline',title='Actual gradient participation by phase')
    for edge in np.arange(-.5,len(rows),1):axes[0].axhline(edge,color='white',lw=.6)
    for edge in np.arange(-.5,6,1):axes[0].axvline(edge,color='white',lw=.6)
    y=np.arange(len(rows));counts=[row['successful_microbatches'] for row in rows]
    axes[1].barh(y,counts,color='#3178a1',height=.72)
    axes[1].set(xlabel='Window microbatches',title='Within selected optimizer prefix')
    durations=[row['recorded_seconds'] for row in rows]
    axes[2].barh(y,durations,color='#ce8737',height=.72)
    axes[2].set(xlabel='Original physical seconds',title='Complete recorded reference')
    for axis in axes[1:]:axis.grid(axis='x',alpha=.15);axis.set_axisbelow(True)
    fig.suptitle(f'42 standalone VBD demos | selected update {audit["checkpoint_step"]}\n'
                 'Measured optimizer coverage; full-task quality and realtime speed remain separate',fontsize=13,fontweight='bold')
    a.output.parent.mkdir(parents=True,exist_ok=True);fig.savefig(a.output,dpi=170);fig.savefig(a.output.with_suffix('.svg'));plt.close(fig)


if __name__=='__main__':main()
