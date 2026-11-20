"""Compare independently recorded model rollouts to time-aligned teacher frames."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--recordings',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    with np.load(args.reference,allow_pickle=False) as archive:
        meta=json.loads(str(archive['metadata']))
        teacher={key:archive[key] for key in ('x','q','v','omega')}
    dt=meta['dt'];origin=meta['source_run']['initial_command_time']
    args.output.mkdir(parents=True,exist_ok=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure,axes=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
    result=[];teacher_plotted=False
    for directory in args.recordings:
        run=json.loads((directory/'run.json').read_text())
        if run['mode']!='shared_network_inference' or run['teacher_trajectory_reads'] or any(run['forbidden_solver_calls'].values()):
            raise ValueError('Independent neural/physical inference recordings required')
        if abs(run['initial_command_time']-origin)>1e-12:
            raise ValueError('Recording and teacher command time origins differ')
        with np.load(directory/'trajectory.npz',allow_pickle=False) as archive:
            times=archive['physical_time'];particle=archive['particle_position'];velocity=archive['particle_velocity']
        indices=np.rint(times/dt).astype(int)
        if indices[-1]>=len(teacher['x']) or np.abs(indices*dt-times).max()>1e-8:
            raise ValueError('Recorded macro endpoints must align with actual teacher states')
        n=run['particle_count'];target=teacher['x'][indices,:n]
        dv=velocity-teacher['v'][indices,:n]
        error=np.linalg.norm(particle-target,axis=-1)
        lift=particle.mean(1)[:,2]-particle[0].mean(0)[2]
        teacher_lift=target.mean(1)[:,2]-target[0].mean(0)[2]
        label=directory.name.replace('demo_','')
        axes[0].plot(times,lift,label=label)
        axes[1].plot(times,np.sqrt((error**2).mean(-1))*1000,label=label)
        if not teacher_plotted:
            axes[0].plot(times,teacher_lift,'k--',label='native VBD teacher',linewidth=2);teacher_plotted=True
        result.append({'recording':str(directory.resolve()),'status':run['status'],
            'step':run['checkpoint_step'],'device':run['device'],'learning_enabled':run.get('learning_enabled',True),
            'dt':run['physical_dt'],'physical_seconds':float(times[-1]),'physical_substeps':run['physical_substeps'],
            'network_calls':run['nn_backbone_calls'],'checkpoint_sha256':run['checkpoint_sha256'],
            'trajectory_sha256':digest(directory/'trajectory.npz'),
            'particle_position_rms_m':float(np.sqrt(((particle-target)**2).mean())),
            'particle_position_max_m':float(error.max()),'terminal_position_rms_m':float(np.sqrt(((particle[-1]-target[-1])**2).mean())),
            'particle_velocity_rms_m_s':float(np.sqrt((dv**2).mean())),
            'maximum_object_lift_m':float(lift.max()),'final_object_lift_m':float(lift[-1]),
            'teacher_maximum_object_lift_m':float(teacher_lift.max()),
            'minimum_final_tet_volume_ratio':run.get('final_tet_volume_ratio_min'),
            'tool_position_error_m':run.get('tool_position_error_m'),'wall_seconds':run['wall_seconds'],
            'full_vbd_calls_during_inference':0,'teacher_reads_during_inference':0})
    axes[0].set(xlabel='Physical time (s)',ylabel='Object center lift (m)',title='Independent rollout / same command program')
    axes[1].set(xlabel='Physical time (s)',ylabel='Particle position RMS (mm)',title='Teacher at the same physical endpoints')
    for axis in axes:axis.grid(alpha=.2);axis.legend(fontsize=8)
    figure.savefig(args.output/'demo_comparison.png',dpi=160);plt.close(figure)
    report={'reference':str(args.reference.resolve()),'reference_sha256':digest(args.reference),
            'recordings':result,'scope':'Teacher read only in this offline comparison. Inference starts from authored scene initialization and reads no teacher states. These are the recorded task/parameters, not held-out generalization.'}
    (args.output/'demo_comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
