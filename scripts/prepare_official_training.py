"""Reproducible official-demo parameter splits and shared replay manifest."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
NEWTON=Path('/home/oem/code/engine/newton')
DEMONSTRATIONS={
    'softbody':('softbody.example_softbody_hanging',120),
    'cloth':('vbd.example_cloth_stiff_material_hanging',120),
    'rigid':('basic.example_basic_dzhanibekov',480),
}
VARIANTS=[('train',1.),('train_extra',1.25),('validation',.8),('test',1.1)]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--generate',action='store_true');a=p.parse_args()
    import numpy as np
    directory=ROOT/'.local/official_v5/default';directory.mkdir(parents=True,exist_ok=True)
    report_dir=ROOT/'reports/official_demos/v5';report_dir.mkdir(parents=True,exist_ok=True)
    rows=[]
    for name,(identifier,frames) in DEMONSTRATIONS.items():
        for variant,factor in VARIANTS:
            if name=='softbody' and variant=='validation':factor=1.15
            path=directory/f'{name}_{variant}.npz'
            if path.exists():
                with np.load(path) as existing:old_factor=json.loads(str(existing['metadata']))['parameter_factor']
                if old_factor!=factor:path.rename(path.with_name(path.stem+f'_failed_factor{old_factor:g}.npz'))
            if a.generate and not path.exists():
                command=['uv','run','--no-sync','--offline','--project',str(NEWTON),'python',str(ROOT/'scripts/generate_official.py'),'--demo',identifier,'--frames',str(frames),'--factor',str(factor),'--output',str(path)]
                with (report_dir/f'{name}_{variant}_teacher.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            if not path.exists():raise FileNotFoundError(path)
            with np.load(path) as data:
                metadata=json.loads(str(data['metadata']));x=data['x'];minimum_det=None
                if len(data['tets']):
                    pts=x[:,data['tets']];deformation=(pts[:,:,1:]-pts[:,:,:1]).swapaxes(-1,-2)@data['tet_Dm'];minimum_det=float(np.linalg.det(deformation).min())
                clearance=float((x[:,data['site_owner'],2]-data['plane_height']).min()) if len(data['site_owner']) else None
                finite=all(np.isfinite(data[k]).all() for k in ('x','q','v','omega'))
                geometry=(minimum_det is None or minimum_det>=.01) and (clearance is None or clearance>=0)
            row={'index':len(rows),'dataset_id':name+'_'+variant,'demo':identifier,'path':str(path.relative_to(ROOT)),'split':'train' if variant.startswith('train') else variant,'parameter_factor':factor,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'dt':metadata['dt'],'teacher_iterations':metadata['teacher_iterations'],'substeps':metadata['teacher_substeps'],'minimum_detF':minimum_det,'tet_orientation_floor':.01,'minimum_physical_plane_clearance':clearance,'status':'PASS' if finite and geometry else 'FAIL','teacher_target':'Original authored iteration budget, not a convergence claim'}
            rows.append(row);print(row['dataset_id'],row['status'],flush=True)
    replay=json.loads((ROOT/'reports/coupling_v3/conditioned/data_manifest.json').read_text())
    # Preserve all five families, both attachment layouts and material replay.
    for row in replay['scenes']:rows.append({**row,'dataset_id':'replay_'+row.get('dataset_id',str(row['index'])),'index':len(rows)})
    report={'status':'PASS' if all(r['status']=='PASS' for r in rows) else 'FAIL','teacher_target':'Official authored budgets, older finite-penalty replay keeps its independently checked reference contract','scenes':rows,'newton_sha':'25298aea1691e060c86f95e8d2b773c898c006d3','strict_reference_convergence':'Not established for official stiff cloth; 64/128 diagnostic retained outside training manifest.'}
    (report_dir/'data_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    verification=json.loads((ROOT/'reports/official_demos/material_verification.json').read_text())
    old=json.loads((ROOT/'reports/coupling_v3/conditioned/verification.json').read_text())
    (report_dir/'verification.json').write_text(json.dumps({'status':'PASS' if verification['status']==old['status']=='PASS' else 'FAIL','material_checks':verification,'previous_mixed_checks':old,'scope':'Operator verification. This is not all-demo acceptance or strict official reference convergence.'},indent=2)+'\n')
    if report['status']!='PASS':raise RuntimeError('Nonfinite or geometrically invalid teacher data retained for diagnosis')

if __name__=='__main__':main()
