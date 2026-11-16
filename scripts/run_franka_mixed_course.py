"""Monitor the qualified Franka replay pilot through GPU preflight and training.

The pilot has a finite declared budget and preserves all previous datasets.
It never promotes a research weight to a converged or visually accepted model.
"""
import argparse
import datetime
from importlib.util import find_spec
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def write(path, value):
    temporary=path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True)
    args=parser.parse_args()
    plan_path=args.plan.resolve();directory=plan_path.parent
    plan=json.loads(plan_path.read_text())
    if plan['status']!='PREPARED_NOT_LAUNCHED':raise ValueError('A fresh prepared pilot is required')
    command=list(plan['argv']);command[0]=sys.executable
    if command[3]!='scripts/train_normal_shared_648.py':raise ValueError('Unexpected training entry')
    manifest_path=ROOT/command[command.index('--manifest')+1]
    if manifest_path.resolve().parent!=directory:raise ValueError('Plan must bind its own course directory')
    manifest=json.loads(manifest_path.read_text())
    if not manifest.get('research_initialization') or not manifest.get('research_sampling'):
        raise ValueError('Explicit fresh research initialization and replay schedule required')
    missing=[name for name in ('git','collada') if find_spec(name) is None]
    if missing:
        raise RuntimeError('Missing asset readers '+str(missing)+
            '; install requirements-vbd-assets.txt in this Python environment before running the full course')
    if any((directory/name).exists() for name in ('pipeline_journal.json','pretraining_development.json','training.json')):
        raise FileExistsError('Keep earlier pilot evidence immutable')
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda('cuda:0',ROOT)
    state={'status':'STARTING','driver_pid':os.getpid(),'new_optimizer_updates':0,
           'official_demos_scheduled':plan['official_demo_count'],'unique_datasets':plan['total_datasets']}
    child=None
    env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','MKL_NUM_THREADS':'1',
         'CUBLAS_WORKSPACE_CONFIG':':4096:8','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
    def persist(status=None):
        if status:state['status']=status
        state['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        write(directory/'pipeline_journal.json',state)
        plan.update(status=state['status'],new_optimizer_updates=state['new_optimizer_updates'])
        write(plan_path,plan)
    def observe_training():
        progress=directory/'pretraining_development_progress.json'
        if progress.exists():
            reviewed=json.loads(progress.read_text())
            state.update(preflight_completed_windows=reviewed['completed_windows'],
                         preflight_expected_windows=reviewed['expected_windows'],
                         preflight_initial_quality_failures=sum(not obj['within_cap']
                             for row in reviewed['metrics'] for obj in row['objects']))
        path=directory/'training.json'
        if not path.exists():return
        report=json.loads(path.read_text())
        state.update(new_optimizer_updates=report['completed_updates'],
            trainer_status=report['status'],fixed_loss_monitors=len(report.get('loss_monitor_history',[])),
            loss_convergence=report.get('loss_convergence'),
            last_optimization_sample=report.get('optimization_samples',[])[-1:] or None,
            actual_sampled_datasets=sum(v>0 for v in report.get('sampled_updates',{}).values()))
    def run_phase(label, argv, log):
        nonlocal child
        with (directory/log).open('xb') as stream:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
            state.update(child_pid=child.pid,child_argv=argv)
            persist(label)
            while child.poll() is None:
                observe_training();persist();time.sleep(10)
            observe_training();state['child_returncode']=child.returncode;persist()
            if child.returncode:raise RuntimeError(f'{label} exited with {child.returncode}; inspect {log}')
    try:
        run_phase('GPU_PREFLIGHT_NO_OPTIMIZATION',command+['--development-only'],'preflight.log')
        proof=json.loads((directory/'pretraining_development.json').read_text())
        if (proof['status']!='RESEARCH_INITIAL_DEVELOPMENT_READY' or proof['new_optimizer_updates']!=0 or
                not proof['all_predictions_accepted']):
            raise ValueError('Complete mixed-course preflight did not pass')
        state['preflight_development_windows']=len(proof['metrics'])
        state['initial_all_quality_caps_met']=proof['all_mobile_objects_within_existing_caps']
        run_phase('TRAINING_SHARED_MODEL_WITH_FIXED_WINDOW_MONITORS',command,'training.log')
        report=json.loads((directory/'training.json').read_text())
        if report['status']!='TRAINING_COMPLETED':raise ValueError('Pilot did not complete normally')
        state.update(selected_step=report['selected_step'],
            checkpoint_directory=str(ROOT/'checkpoints'/directory.name))
        persist('PILOT_COMPLETED_NATIVE_ROLLOUT_REVIEW_PENDING')
    except BaseException as error:
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=30)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        observe_training();state['failure']=repr(error)
        persist('PILOT_STOPPED_RETAIN_ACTUAL_EVIDENCE')
        raise
    finally:
        ownership.close()


if __name__=='__main__':main()
