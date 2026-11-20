"""Run current-weight robot demos and collect their videos in one directory."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
CASES=('franka','w1','cloth','piper','w1_dual','franka_rigid_cloth')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    p.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    p.add_argument('--device',choices=('cpu','cuda:0'),default='cuda:0')
    p.add_argument('--window',action='store_true')
    p.add_argument('--recipe',type=Path,default=ROOT/'configs/robot_validation_cloth_ft.json')
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();recipe=json.loads(a.recipe.read_text())
    output=(a.output or ROOT/'outputs'/('robot_suite_'+datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    commands=[]
    for i,name in enumerate(a.cases,1):
        row=recipe['scenes'][name];recording=output/'recordings'/name;video=output/'videos'/f'{i:02d}_{name}.mp4'
        infer=[sys.executable,'-u',str(ROOT/'scripts/run_robot_validation.py'),'--scene',name,
               '--recipe',str(a.recipe.resolve()),'--device',a.device,'--output',str(recording)]
        if a.window:infer.append('--window')
        render=[sys.executable,'-u',str(ROOT/row['renderer']),'--recording',str(recording),'--output',str(video)]
        commands.append(dict(scene=name,inference_command=infer,render_command=render,
                             recording=str(recording),video=str(video)))
    if a.dry_run:
        print(json.dumps({'output':str(output),'cases':commands},indent=2));return
    output.mkdir(parents=True,exist_ok=False)
    for name in ('recordings','videos','previews','logs'):(output/name).mkdir()
    report={'status':'RUNNING','recipe':str(a.recipe.resolve()),'checkpoint_sha256':recipe['checkpoint_sha256'],'cases':commands}
    def write():
        tmp=output/'manifest.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(output/'manifest.json')
    write();print('Output:',output,flush=True)
    env=dict(os.environ,WP_VBD_NEWTON_ROOT=str(ROOT/recipe['newton_root']))
    for row in commands:
        row['status']='RUNNING';write()
        with (output/'logs'/(row['scene']+'.log')).open('w') as log:
            code=subprocess.call(row['inference_command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        result=Path(row['recording'])/'run.json'
        run=json.loads(result.read_text()) if result.exists() else {}
        if (code or run.get('status')!='COMPLETE' or run.get('checkpoint_sha256')!=recipe['checkpoint_sha256']):
            row.update(status='FAILED_INFERENCE',exit_code=code);write();continue
        row.update(status='RENDERING',neural_calls=run.get('nn_backbone_calls'),
                   physical_substeps=run.get('physical_substeps'),nominal_substeps=run.get('substeps',3))
        write()
        with (output/'logs'/(row['scene']+'_render.log')).open('w') as log:
            code=subprocess.call(row['render_command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if code==0:
            for suffix in ('.json','.poster.png','.final.png','.contact_sheet.png'):
                artifact=Path(row['video']).with_suffix(suffix)
                if artifact.exists():artifact.replace(output/'previews'/artifact.name)
        row['status']='RECORDED_REQUIRES_REVIEW' if code==0 else 'FAILED_RENDER';write()
    report['status']='RECORDED_REQUIRES_REVIEW' if all(r['status']=='RECORDED_REQUIRES_REVIEW' for r in commands) else 'FAILED'
    write()
    if report['status']=='FAILED':raise SystemExit(1)


if __name__=='__main__':main()
