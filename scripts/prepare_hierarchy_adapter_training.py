"""Archive a rejected full-response run before the declared adapter curriculum."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REPORTS=ROOT/'reports/shared_hierarchy_v20'
WEIGHTS=ROOT/'checkpoints/shared_hierarchy_v20'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    report=json.loads((REPORTS/'training.json').read_text())
    assert report['status']=='COMPLETED' and report['completed_updates']==128
    assert report['selected_checkpoint_step']==0,'A trained checkpoint is already eligible; do not launch the conditional follow-up'
    assert all(not row['validation']['selection']['eligible'] for row in report['history'] if row['step'])
    cfg=json.loads((ROOT/'configs/shared_hierarchy_adapters_v20.json').read_text())
    assert cfg['model']==report['config']['model']
    assert cfg['training']['trainable_prefixes']==['protected_response.hierarchy.','protected_response.transport.']
    assert cfg['training']['learning_rate']==1e-5
    for key,value in report['config']['training'].items():
        if key not in ('trainable_prefixes','learning_rate'):assert cfg['training'][key]==value
    assert cfg['initialize_from']==report['config']['initialize_from']
    assert sha(Path(cfg['initialize_from']))==report['initialization_checkpoint_sha256']
    archive=REPORTS/'attempts/full_response_128';weight_archive=WEIGHTS/'attempts/full_response_128'
    if archive.exists() or weight_archive.exists():raise FileExistsError('Preserve the archived experiment')
    archive.mkdir();weight_archive.mkdir()
    files={}
    for name in ('initial.pt','shared.pt','last.pt'):
        path=WEIGHTS/name;files[name]={'sha256':sha(path),'archive':str((weight_archive/name).relative_to(ROOT))}
        path.rename(weight_archive/name)
    for name in ('training.json','training.log'):(REPORTS/name).rename(archive/name)
    change={'status':'ADAPTER_CURRICULUM_DECLARED_AFTER_VALIDATION_REJECTION','previous_run':str((archive/'training.json').relative_to(ROOT)),'previous_report_sha256':sha(archive/'training.json'),'previous_completed_updates':128,'previous_selected_update':0,'previous_checkpoints':files,'new_stage_budget':128,'new_trainable_prefixes':cfg['training']['trainable_prefixes'],'new_learning_rate':cfg['training']['learning_rate'],'common_data_manifest_sha256':sha(REPORTS/'data_manifest.json'),'initial_checkpoint_sha256':sha(Path(cfg['initialize_from'])),'source_configuration':'configs/shared_hierarchy_adapters_v20.json','validation_caps_and_splits_unchanged':True,'fresh_holdout_observed':False,'scope':'Development curriculum change, not an independent architecture comparison. All 128 rejected updates remain part of the reported development cost.'}
    (REPORTS/'curriculum_revision.json').write_text(json.dumps(change,indent=2)+'\n')
    print(json.dumps(change,indent=2))


if __name__=='__main__':main()
