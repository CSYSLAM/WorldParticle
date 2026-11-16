"""Prepare the first ordinary-demo course for the new 648-wide model."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report',type=Path,default=ROOT/'reports/shared_normal_648_v81/data_manifest.json')
    args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    prior=ROOT/'reports/shared_particle_hanging_v63/data_manifest.json'
    legacy=ROOT/'reports/shared_legacy_suite_v25/data_manifest.json'
    original=json.loads(prior.read_text())['current_optimization_scenes']
    old=json.loads(legacy.read_text())['scenes']
    old_demos=set(json.loads((ROOT/'reports/shared_particle_hanging_v63/checkpoint.json').read_text())['trained_official_demos'])
    legacy_demos=old_demos-{r['demo'] for r in original}
    # Explicit first-stage scheduling under the user's shared-GPU budget.
    # These assets remain required in the all51 inventory and later courses.
    deferred={'urdf_train_default','soft_rigid_train_default','soft_rigid_mix_train_default','cloth_hanging_self_train_default'}
    rows=[]
    for r in original:
        if r['dataset_id'] in deferred:continue
        label=r['dataset_id']
        loader=('self' if label.startswith('cloth_') else 'rod' if label.startswith('cable_')
                else 'surface' if label.startswith('gripper_')
                else 'joint' if label in ('basic_joints_train_default','mimic_default.npz_train_default','newton_cradle_train_default','conveyor_forces_train_default','conveyor_train_default')
                else 'rigid')
        rows.append({**r,'loader':loader})
    for r in old:
        if r.get('demo') not in legacy_demos or r['split'] not in ('train','train_temporal_prefix_and_validation_suffix'):continue
        validation=next((v for v in old if v.get('demo')==r['demo'] and v['split']=='validation'),None)
        rows.append({**r,'loader':'legacy',
                     **({'validation_path':validation['path'],'validation_sha256':validation['sha256']} if validation else {})})
    for r in old:
        if r.get('demo') is None and r['split']=='train':rows.append({**r,'loader':'legacy'})
    demos=sorted({r['demo'] for r in rows if r.get('demo')})
    assert len(legacy_demos)==10 and len(demos)==26,(legacy_demos,demos)
    for r in rows:
        path=ROOT/r['path'];assert digest(path)==r['sha256'],path
        stat=path.stat();r['admitted_file_stat']={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns}
        if r.get('validation_path'):assert digest(ROOT/r['validation_path'])==r['validation_sha256']
    initial=ROOT/'.local/shared_capacity_v78/candidate648_probe.pt'
    report={'status':'NORMAL_COURSE_DATA_PREPARED_NOT_TRAINED','initial_checkpoint':str(initial.relative_to(ROOT)),
            'initial_checkpoint_sha256':digest(initial),'initial_probe_updates':3,'inherited_selected_official_demos':0,
            'newton_sha':'25298aea1691e060c86f95e8d2b773c898c006d3','all_official_targets':51,
            'current_course_official_demos':demos,'current_course_demo_count':len(demos),'scenes':rows,
            'current_course_dataset_count':len(rows),'deferred_previous_dataset_ids':sorted(deferred),
            'source_manifests_sha256':{str(p.relative_to(ROOT)):digest(p) for p in (prior,legacy)},
            'resource_policy':{'gpu_process_budget_gib':12,'torch_allocator_budget_gib':10,'batch_size':1,'horizon':2,'maximum_duty_cycle':.25},
            'scope':'User-requested ordinary-demo first stage with one648-wide shared backbone. Old32-wide coverage is not inherited. All51 remain targets; dense self contact, large coupling and external exchange are scheduled later, not removed. Existing separate qualified validation episodes or temporal suffixes are development only.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='scenes'},indent=2))


if __name__=='__main__':main()
