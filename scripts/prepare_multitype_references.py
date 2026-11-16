"""Serial complete material references; measure refinement before admission."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/shared_multitype_reference_v20.json')
    parser.add_argument('--families', nargs='+')
    parser.add_argument('--episodes', nargs='+')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd='/home/oem/code/engine/newton',text=True).strip() == cfg['newton_sha']
    reports = ROOT/cfg['report_dir']; reports.mkdir(parents=True, exist_ok=True)
    (reports/'.gitignore').write_text('*.log\n*.npz\n')
    records = []
    for family in cfg['families']:
        if args.families and family['name'] not in args.families:continue
        for episode in cfg['episodes']:
            if args.episodes and episode['name'] not in args.episodes:continue
            name = family['name']+'_'+episode['name']; sources = []
            for budget in family['budgets']:
                path = ROOT/cfg['data_dir']/(name+f'_{budget}.npz')
                report = reports/(name+f'_{budget}.json')
                if not path.exists():
                    command=[sys.executable,str(ROOT/'scripts/generate_official.py'),'--demo',family['demo'],'--factor',str(episode['factor']),'--frames',str(family['frames']),'--iterations',str(budget),'--output',str(path),'--report',str(report)]
                    if family['name'] in ('softbody','cloth'):command.append('--capture-substeps')
                    print('record',name,budget,flush=True)
                    with (reports/(name+f'_{budget}.log')).open('w') as log:
                        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                evidence=report
                if not evidence.exists() and cfg.get('reuse_report_dir'):
                    evidence=ROOT/cfg['reuse_report_dir']/(name+f'_{budget}.json')
                result=json.loads(evidence.read_text()); assert result['data_sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
                values=load(path); metadata=json.loads(str(values['metadata']))
                assert metadata['official_demo']==family['demo'] and metadata['parameter_factor']==episode['factor']
                assert metadata['teacher_frames']==family['frames'] and metadata['teacher_iterations']==budget
                sources.append({'path':str(path.relative_to(ROOT)),'sha256':result['data_sha256'],'iterations':budget,'terminal_check':result['terminal_check']})
            low,high=[load(ROOT/source['path']) for source in sources]
            dynamic={'metadata','x','q','v','omega'}
            same={key:bool(np.array_equal(low[key],high[key])) for key in low if key not in dynamic}
            assert all(same.values()),same
            mobile=high['movable']; length=float(high['reference_length'])
            position=np.linalg.norm(low['x'][:,mobile].astype(np.float64)-high['x'][:,mobile],axis=-1)
            a,b=[values['q'][:,mobile].astype(np.float64) for values in (low,high)]
            cosine=np.abs((a*b).sum(-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)))
            rotation=2*np.arccos(cosine.clip(0,1))
            finite=all(np.isfinite(high[key]).all() for key in ('x','q','v','omega'))
            minimum=None
            if len(high['tets']):
                points=high['x'][:,high['tets']]
                minimum=float(np.linalg.det((points[:,:,1:]-points[:,:,:1]).swapaxes(-1,-2)@high['tet_Dm']).min())
            within=bool(position.max()/length<cfg['position_refinement_limit_over_L'] and rotation.max()<cfg['orientation_refinement_limit_rad'])
            record={'name':name,'demo':family['demo'],'split':episode['split'],'factor':episode['factor'],'sources':sources,'same_static_and_current_inputs':all(same.values()),'maximum_position_difference_over_L':float(position.max()/length),'maximum_orientation_difference_rad':float(rotation.max()),'finite':finite,'minimum_detF':minimum,'within_refinement_limits':within,'original_terminal_checks_passed':all(source['terminal_check']=='PASS' for source in sources),'training_admission_granted':False,'config_sha256':hashlib.sha256(args.config.read_bytes()).hexdigest()}
            records.append(record);(reports/(name+'_refinement.json')).write_text(json.dumps(record,indent=2)+'\n')
            print(json.dumps(record),flush=True)
    print('Reference preparation finished; admission is a separate decision.',flush=True)


if __name__=='__main__':main()
