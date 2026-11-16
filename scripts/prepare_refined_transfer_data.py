"""Collect immutable full-duration rod episodes with explicit refinement pairs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/refined_transfer_data_v14.json')
    parser.add_argument('--episodes',nargs='+')
    parser.add_argument('--generate',action='store_true')
    args=parser.parse_args();cfg=json.loads(args.config.read_text())
    source=subprocess.check_output(['git','-C','/home/oem/code/engine/newton','rev-parse','HEAD'],text=True).strip()
    assert source==cfg['newton_sha'],'Pinned reference changed'
    reports=ROOT/cfg['report_dir'];reports.mkdir(parents=True,exist_ok=True)
    (reports/'.gitignore').write_text('*.log\n*.npz\n')
    environment={**os.environ,'CUDA_VISIBLE_DEVICES':'','OPENBLAS_NUM_THREADS':'1'}
    for episode in cfg['episodes']:
        if args.episodes and episode['name'] not in args.episodes:continue
        pair=[]
        for budget in cfg['budgets']:
            path=ROOT/episode.get('existing',{}).get(str(budget),str(Path(cfg['data_dir'])/(episode['name']+f'_cpu{budget}.npz')))
            if not path.exists() and args.generate:
                command=[sys.executable,str(ROOT/'scripts/generate_rod_history.py'),'--demo',cfg['demo'],'--frames',str(cfg['frames']),'--factor',str(episode['factor']),'--device','cpu','--iterations',str(budget),'--output',str(path),'--report',str(reports/(episode['name']+f'_cpu{budget}_final.json'))]
                with (reports/(episode['name']+f'_cpu{budget}.log')).open('w') as log:
                    subprocess.run(command,cwd=ROOT,env=environment,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not path.exists():raise FileNotFoundError(path)
            with np.load(path,allow_pickle=False) as data:
                metadata=json.loads(str(data['metadata']))
                assert metadata['official_demo']==cfg['demo'] and metadata['teacher_frames']==cfg['frames']
                assert metadata['parameter_factor']==episode['factor'] and metadata['teacher_iterations']==budget
                assert metadata['teacher_device']=='cpu'
                assert metadata['reference_final_check']=='PASS',metadata['reference_final_check']
                assert all(np.isfinite(data[key]).all() for key in ('x','q','v','omega','history__dual','input_force','input_torque'))
            pair.append({'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'iterations':budget,'terminal_check':'PASS'})
        audit=reports/(episode['name']+'_refinement.json')
        subprocess.run([sys.executable,str(ROOT/'scripts/audit_rod_reference_refinement.py'),'--lower',str(ROOT/pair[0]['path']),'--higher',str(ROOT/pair[1]['path']),'--report',str(audit)],cwd=ROOT,env=environment,check=True,stdout=subprocess.DEVNULL)
        result=json.loads(audit.read_text())
        assert result['refinement_limits']=={'position_over_L':cfg['position_refinement_limit_over_L'],'orientation_rad':cfg['orientation_refinement_limit_rad']}
        record={'status':'REFINEMENT_CHECKED','demo':cfg['demo'],'episode':episode['name'],'split':episode['split'],'parameter_factor':episode['factor'],'sources':pair,'refinement_report':str(audit.relative_to(ROOT)),'within_refinement_limits':result['within_refinement_limits'],'training_admission_granted':False,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest(),'newton_sha':source}
        (reports/(episode['name']+'_episode.json')).write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps(record),flush=True)


if __name__=='__main__':main()
