"""Add original moving-boundary demos without reading dynamic teacher inputs."""
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]

def main():
    import numpy as np
    directory=ROOT/'.local/official_v5/controlled';directory.mkdir(exist_ok=True)
    report_dir=ROOT/'reports/official_demos/controlled';report_dir.mkdir(exist_ok=True)
    manifest=json.loads((ROOT/'reports/official_demos/v5/data_manifest.json').read_text())
    for variant,factor in [('train',1.),('train_extra',1.25),('validation',.8),('test',1.1)]:
        path=directory/f'stretch_{variant}.npz'
        if not path.exists():
            command=['uv','run','--no-sync','--offline','--project','/home/oem/code/engine/newton','python',str(ROOT/'scripts/generate_official.py'),'--demo','vbd.example_cloth_stiff_material_stretch','--frames','360','--factor',str(factor),'--output',str(path)]
            with (report_dir/f'{variant}_teacher.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        with np.load(path) as data:
            metadata=json.loads(str(data['metadata']));ids=data['kinematic_ids'];x=data['x'];inputs=data['input_x']
            assert not data['movable'][ids].any()
            assert len(inputs)==len(x)-1 and inputs.shape[1]==len(ids)
            error=float(np.max(np.abs(x[1:,ids]-inputs)))
            assert error==0,error
            mobile=data['movable'];finite=all(np.isfinite(data[k]).all() for k in ('x','q','v','omega','input_x','input_q'))
            # Input positions contain ONLY prescribed boundary entities.
            assert inputs.shape[1]<x.shape[1]
            row={'index':len(manifest['scenes']),'dataset_id':'stretch_'+variant,'demo':metadata['official_demo'],'path':str(path.relative_to(ROOT)),'split':'train' if variant.startswith('train') else variant,'parameter_factor':factor,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'status':'PASS' if finite and error==0 else 'FAIL','dt':metadata['dt'],'substeps':metadata['teacher_substeps'],'fixed_target_error':error,'kinematic_entity_count':len(ids),'dynamic_teacher_inputs':False}
        manifest['scenes'].append(row);print(row,flush=True)
    manifest['status']='PASS' if all(r['status']=='PASS' for r in manifest['scenes']) else 'FAIL'
    manifest['external_input_contract']='Original authored prescribed boundary x/q only; no dynamic future teacher state enters inference.'
    (report_dir/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verification=json.loads((ROOT/'reports/official_demos/v5/verification.json').read_text())
    verification['prescribed_boundary_recording']='PASS; every teacher output fixed position exactly equals its current explicit command'
    (report_dir/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    config=json.loads((ROOT/'configs/shared_official_v5.json').read_text())
    config.update(name='shared_official_controlled_v6',report_dir='reports/official_demos/controlled',checkpoint_dir='checkpoints/shared_official_controlled_v6',initialize_from=str(ROOT/'checkpoints/mixed_conditioned_v4/shared.pt'),initialize_reset_coarse_head=True)
    config['model'].update(guard_predictor_merit=True,collective_retraction=True,coarse_gain_mode='centered',coarse_composition='balanced',exact_inertia=True)
    config['training'].update(steps=480,horizon=2,validation_steps=240,validation_interval=80,max_seconds=3600,fp64_gradient_norm=True,learning_rate=.0002)
    config['scope']['prescribed_boundaries']='Original step controls; explicit fixed-entity-only input arrays for training, original controller for native inference'
    (ROOT/'configs/shared_official_controlled_v6.json').write_text(json.dumps(config,indent=2)+'\n')

if __name__=='__main__':main()
