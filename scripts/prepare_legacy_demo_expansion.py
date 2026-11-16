"""Reuse complete authored-budget material/rod histories for broader training."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
import numpy as np


def main():
    wanted={'vbd.example_cloth_stiff_material_stretch','vbd.example_cable_bend_stiffness','vbd.example_cable_dahl_hysteresis','vbd.example_cable_michell_threshold','vbd.example_cable_torsion_material_mapping'}
    old=json.loads((ROOT/'reports/rod_history_v9/data_manifest.json').read_text());rows=[]
    for row in old['scenes']:
        if row.get('demo') not in wanted or row['split']=='test':continue
        # These authored low-budget trajectories have known refinement errors.
        # Reuse the existing high-budget full references below instead.
        if row['demo'] in {'vbd.example_cable_dahl_hysteresis','vbd.example_cable_michell_threshold'}:continue
        path=ROOT/row['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']))
            assert metadata['official_demo']==row['demo']
            for key in ('x','q','v','omega'):
                assert np.isfinite(archive[key]).all()
            if metadata.get('history_contract'):
                required=['dual','kappa','increment','stress','previous_x','previous_q','reset']
                assert all('history__'+key in archive.files for key in required)
                assert all(np.isfinite(archive['history__'+key]).all() for key in required)
            if metadata.get('contacts') not in ('none','analytic_plane'):raise ValueError('Additional physical contacts are not part of this replay adapter')
        row={**row,'teacher_target':'Original finite-budget dynamics with explicit controls/history; comparable behavior, no converged-trajectory or exact-pose claim.'}
        rows.append(row)
    for demo,source,budget in [
        ('vbd.example_cable_dahl_hysteresis','.local/rod_chain_v12/dahl_cpu1024_full.npz',1024),
        ('vbd.example_cable_michell_threshold','.local/refined_michell_v18/original_audit_cpu256.npz',256),
    ]:
        path=ROOT/source
        with np.load(path,allow_pickle=False) as archive:
            metadata=json.loads(str(archive['metadata']));assert metadata['official_demo']==demo and metadata['teacher_iterations']==budget
            assert metadata['history_contract']=='rod_compliant_alm_v1'
            for key in ('x','q','v','omega','history__dual','history__kappa','history__increment','history__stress','history__previous_x','history__previous_q','history__reset'):
                assert np.isfinite(archive[key]).all()
            size=len(archive['x'])-1
        rows.append({'dataset_id':demo+'/refined_temporal','demo':demo,'path':source,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'split':'train_temporal_prefix_and_validation_suffix','training_index_range':[0,int(.6*size)],'validation_starts':np.linspace(int(.65*size),size-3,6,dtype=int).tolist(),'teacher_iterations':budget,'teacher_target':'Existing high-budget reference, temporal development split. No held-out material or full refinement certificate claimed.'})
    assert {r['demo'] for r in rows}==wanted and len(rows)==11
    result={'status':'PASS','newton_sha':'25298aea1691e060c86f95e8d2b773c898c006d3','scenes':rows,'new_official_candidates':sorted(wanted),'training_updates_granted':0,'scope':'Previously converted material/rod data eligibility. Test episodes remain excluded. Actual shared optimizer gradients, selected weights and native behavior must be verified independently.'}
    directory=ROOT/'reports/legacy_demo_expansion_v25';directory.mkdir(parents=True,exist_ok=True);(directory/'.gitignore').write_text('*.log\n*.npz\n');(directory/'data_manifest.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':result['status'],'episodes':len(rows),'new_demos':len(wanted)},indent=2))


if __name__=='__main__':main()
