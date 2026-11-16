"""Admit refined transfer episodes to a bounded calibration with legacy replay."""
import copy
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]


def read(path):return json.loads((ROOT/path).read_text())


def main():
    report_dir=ROOT/'reports/shared_refined_v16'
    if (report_dir/'training.json').exists():raise FileExistsError('Keep completed or failed optimizer evidence immutable')
    required={
        'curved_geometry':'reports/rod_flexure_v15/mode_verification.json',
        'fp64_virtual_work':'reports/rod_precision_v14/geometry_verification.json',
        'six_scene_runtime':'reports/rod_flexure_v15/runtime_final.json',
        'legacy_replay':'reports/rod_flexure_v15/legacy_replay_verification.json',
    }
    for path in required.values():assert read(path)['status']=='PASS',path
    for name in ('native_transfer_probe','native_mapping_probe'):
        result=read('reports/rod_flexure_v15/'+name+'.json')
        assert result['test_final']=='PASS' and result['failure'] is None,name
    rows=[];audits=[]
    # These replay rows retain their original material splits and fingerprints.
    for row in read('reports/rod_cycle_v10/data_manifest.json')['scenes']:
        if row['dataset_id'].startswith(('replay_coupled/','replay_legacy_plane/')):rows.append(copy.deepcopy(row))
    for name in ('train','train_extra','validation','test'):
        episode=read('reports/refined_transfer_v14/'+name+'_episode.json')
        audit=read(episode['refinement_report'])
        assert audit['within_refinement_limits'] and episode['within_refinement_limits']
        source=episode['sources'][-1]
        assert source['iterations']==256 and source['terminal_check']=='PASS'
        path=ROOT/source['path'];assert hashlib.sha256(path.read_bytes()).hexdigest()==source['sha256']
        rows.append({'index':len(rows),'dataset_id':episode['demo']+'/refined_'+name,'demo':episode['demo'],'path':source['path'],'split':episode['split'],'sha256':source['sha256'],'parameter_factor':episode['parameter_factor'],'substeps':3900,'dt':1/600,'status':'PASS','history_contract':'rod_compliant_alm_v1','teacher_refinement_report':episode['refinement_report']})
        audits.append(episode['refinement_report'])
    for index,row in enumerate(rows):row['index']=index
    report_dir.mkdir(parents=True,exist_ok=True);(report_dir/'.gitignore').write_text('*.log\n*.npz\n')
    manifest={'status':'PASS','newton_sha':read('configs/refined_transfer_data_v14.json')['newton_sha'],'scenes':rows,
        'teacher_admission_scope':'Preliminary fixed-h trajectory training for one already admitted demo: finite complete source episodes, original task assertions, identical-input iteration refinement and independent local operator checks. Not full P1 physical/time-step certification.',
        'refinement_reports':audits,'test_isolation':'Complete parameter episodes excluded from optimizer and model selection; only manifest/file fingerprints are inspected.',
        'excluded_new_labels':['All default-budget rod references, including known underconverged Dahl and bend-stiffness episodes'],
        'inherited_actual_training_demo_count':10,'all_51_demo_acceptance':False}
    (report_dir/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    verification={'status':'PASS','checks':required,'native_prototype_tasks':['native_transfer_probe','native_mapping_probe'],'scope':'Admission to this bounded calibration only. Curve FP32 source-force arithmetic discrepancy and Dahl independent-reference discrepancy remain recorded; no global physics or all-demo certification.'}
    (report_dir/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    config=read('configs/shared_flexure_probe.json')
    config.update(name='shared_refined_v16',report_dir='reports/shared_refined_v16',checkpoint_dir='checkpoints/shared_refined_v16',initialize_from=str(ROOT/'checkpoints/shared_cycle_v10/shared_strict.pt'))
    config['training'].update(steps=64,learning_rate=1e-5,validation_interval=32,validation_steps=240,validation_steps_by_demo={'vbd.example_cable_twist_transfer':600},validation_cyclic_shape=False,official_sampling_weight=1,ensure_initial_coverage=True,trainable_prefixes=['response.entity_encoders.3.'],batch_seed=20261103)
    config['scope'].update(operator_probe='Calibrate already trained shared weights with refined complete transfer episodes and previous coupled/plane replay. No new demos claimed.',validation_scope='12 single-step samples span each whole validation episode; autonomous transfer model selection covers first 1s. Full original terminal tasks remain independent.',training_curriculum='Only existing rod input encoder (5760 parameters); all other shared tensors frozen. Step-zero selection is retained if no update improves validation; do not force a release.')
    (ROOT/'configs/shared_refined_v16.json').write_text(json.dumps(config,indent=2)+'\n')
    print(json.dumps({'status':'READY_FOR_BOUNDED_CALIBRATION','counts':{split:sum(row['split']==split for row in rows) for split in ('train','validation','test')},'optimizer_steps':64,'trained_demo_count_increase':0},indent=2))


if __name__=='__main__':main()
