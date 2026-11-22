"""Interleave teacher and learned material modules at the SAME held state.

The driver runs analytic material response throughout. Shadow readout calls
do not advance state, add backbone calls, or constitute full-task FPS tests.
"""
import argparse,json,runpy,statistics,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--repeats',type=int,default=7)
    a=p.parse_args()
    if a.repeats<3:p.error('At least three interleaved samples required')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    import torch
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.analytic_material_response import correct
    class Probe:
        done=False
        def observer(self,model,scene,state,metric,coarse,source,update,w,baseline,dt):
            if self.done or not model.config.get('capture_cuda_graphs'):return None
            def sample(*unused):
                self.done=True;head=model.shared_material_head
                if head is None:raise ValueError('Trained shared material checkpoint required')
                counter=getattr(scene,'_analytic_material_last_solves',0)
                methods={'analytic':lambda:correct(scene,state,metric,coarse,source,update,
                    maximum_columns=model.config.get('analytic_material_maximum_columns',96),
                    maximum_nodes=model.config.get('analytic_material_maximum_nodes',8192),
                    blocked=model.config.get('analytic_material_block_solve',True)),
                    'learned':lambda:head.correct(scene,state,metric,coarse,source,update,w,baseline,model.protected_response.last_latent,dt)}
                rows=[];values={}
                try:
                    for fn in methods.values():fn();fn()
                    for repeat in range(a.repeats):
                        for name in (('analytic','learned') if repeat%2==0 else ('learned','analytic')):
                            torch.cuda.synchronize();start=time.perf_counter();value,valid=methods[name]();torch.cuda.synchronize()
                            elapsed=time.perf_counter()-start
                            if not bool(valid.all()):raise ValueError('Invalid module result')
                            rows.append(dict(repeat=repeat,method=name,seconds=elapsed));values[name]=value.detach()
                    target=values['analytic']-update;error=values['learned']-values['analytic']
                    mass=scene.mass[None,:,None].double()*error.new_tensor([1,1,1,*([float(scene.length)**2]*3)])
                    relative=float((mass*error.square()).sum()/(mass*target.square()).sum().clamp_min(1e-30))
                    median={name:statistics.median(r['seconds'] for r in rows if r['method']==name) for name in methods}
                    result=dict(status='COMPLETE_MODULE_DIAGNOSTIC',samples=rows,median_seconds=median,
                        latency_reduction_percent=100*(1-median['learned']/median['analytic']),
                        relative_squared_correction_error=relative,
                        maximum_update_difference_m_or_rad=float(error.abs().max()),
                        same_state_latent_weights_physical_metric=True,additional_backbone_calls=0,
                        scope='One held Franka state; warmed interleaved module latency only, not full-frame or full-task speedup. Driver timing includes diagnostic probes and is not an inference FPS result.')
                    (a.output/'module_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
                    print(json.dumps(result),flush=True)
                finally:scene._analytic_material_last_solves=counter
            return sample
    probe=Probe();original=MixedSolver.__init__
    def initialize(self,*args,**kwargs):
        original(self,*args,**kwargs);self.config['shared_material_use_learned']=False
        self.material_correction_recorder=probe
    MixedSolver.__init__=initialize;argv=sys.argv
    try:
        sys.argv=['scripts/benchmark_robot_step.py','--case','franka','--checkpoint',str(a.checkpoint.resolve()),
            '--recording',str(a.recording.resolve()),'--output',str(a.output/'instrumented_driver'),
            '--frame-loop','--repeats','1','--variants','reference','fused_graph']
        runpy.run_path(str(ROOT/'scripts/benchmark_robot_step.py'),run_name='__main__')
    finally:MixedSolver.__init__=original;sys.argv=argv
    if not probe.done:raise RuntimeError('Material module diagnostic was never executed')


if __name__=='__main__':main()
