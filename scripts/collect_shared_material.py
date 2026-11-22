"""Capture offline final-stage material labels over a complete robot demo."""
import argparse,json,os,runpy,sys,hashlib
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scene',choices=('franka','cloth','w1'),required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=2)
    p.add_argument('--frames',type=int)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    recipe=json.loads((ROOT/'configs/robot_validation_cloth_ft.json').read_text());row=recipe['scenes'][a.scene]
    os.environ.setdefault('WP_VBD_NEWTON_ROOT',str(ROOT/recipe['newton_root']))
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.material_correction_data import MaterialCorrectionRecorder
    from wp_vbd.inference_policy import resolve_policy
    recorder=MaterialCorrectionRecorder(a.output,scene=a.scene,stride=a.stride)
    initialize=MixedSolver.__init__
    def instrument(self,*args,**kwargs):
        initialize(self,*args,**kwargs)
        if self.shared_material_head is not None:raise ValueError('Capture starts from unchanged analytic material baseline')
        self.material_correction_recorder=recorder
    MixedSolver.__init__=instrument
    command=[row['runner'],'--checkpoint',str(ROOT/recipe['checkpoint']),'--output',str(a.output/'rollout'),
        '--device',a.device,*row['arguments'],'--execution-policy',resolve_policy('auto',scene=a.scene,device=a.device)]
    if a.frames is not None:command[command.index('--frames')+1]=str(a.frames)
    original_argv=sys.argv;complete=False
    try:
        sys.argv=command;runpy.run_path(str(ROOT/row['runner']),run_name='__main__');complete=True
    finally:
        sys.argv=original_argv;MixedSolver.__init__=initialize
        report=recorder.finish(complete=complete)
        report['checkpoint_sha256']=hashlib.sha256((ROOT/recipe['checkpoint']).read_bytes()).hexdigest()
        (a.output/'capture.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report),flush=True)


if __name__=='__main__':main()
