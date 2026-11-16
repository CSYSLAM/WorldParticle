"""Prepare a frozen course only after its owned complete CUDA probe exits."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
        if fields[0]=='Z':return None
        return {'start_ticks':int(fields[19]),'argv':[s.decode() for s in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if s]}
    except FileNotFoundError:return None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,required=True);args=parser.parse_args()
    plan=json.loads(args.plan.read_text());journal=Path(plan['journal']);worker=Path(plan['worker'])
    if journal.exists():raise FileExistsError('Preserve previous preparation evidence')
    state={'status':'WAITING_FOR_BOUND_PREFLIGHT_PROCESS','plan_sha256':digest(args.plan),
           'worker_commit':plan['worker_commit'],'new_optimizer_updates':0,'new_training_admissions':0}
    def persist(status):
        state['status']=status;state['observed_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
        temporary=journal.with_suffix('.tmp');temporary.write_text(json.dumps(state,indent=2)+'\n');temporary.replace(journal)
    def check_source():
        assert subprocess.check_output(['git','-C',str(worker),'rev-parse','HEAD'],text=True).strip()==plan['worker_commit']
        assert plan['module_sha256']=={name:digest(worker/'src/wp_vbd'/name) for name in plan['module_sha256']}
        assert digest(worker/'scripts/prepare_normal_648_extension.py')==plan['builder_sha256']
    check_source();persist(state['status'])
    try:
        while True:
            live=identity(plan['preflight_profile_pid'])
            if live is None or live['start_ticks']!=plan['preflight_profile_identity']['start_ticks']:break
            assert live==plan['preflight_profile_identity'],'Bound GPU profile command changed'
            time.sleep(5)
        check_source()
        profile=json.loads(Path(plan['profile_report']).read_text());assert profile['child_exit_code']==0
        assert profile['child_argv']==plan['preflight_argv']
        proof_path=Path(plan['preflight_report']);proof=json.loads(proof_path.read_text())
        assert proof['status']=='PASS' and sorted(r['index'] for r in proof['cases'])==[0,4,700,1400,3000]
        assert proof['new_optimizer_updates']==proof['new_training_admissions']==proof['full_vbd_prediction_calls']==0
        assert proof['implementation_sha256']==plan['module_sha256']
        destination=worker/plan['worker_proof_path'];destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert digest(destination)==digest(proof_path)
        else:destination.write_bytes(proof_path.read_bytes())
        expected=Path(plan['root_manifest']);built=worker/plan['worker_manifest']
        assert not expected.exists() and not built.exists()
        state['preflight_report_sha256']=digest(proof_path);persist('QUALIFIED_FIVE_PHASE_PROBE_PREPARING_COURSE')
        task_env=os.environ.copy();task_env['CUDA_VISIBLE_DEVICES']=''
        with Path(plan['builder_log']).open('xb') as log:
            subprocess.run(plan['builder_argv'],cwd=worker,env=task_env,stdout=log,stderr=subprocess.STDOUT,check=True)
        manifest=json.loads(built.read_text());assert manifest['status']=='NORMAL_COURSE_EXTENSION_PREPARED_NOT_TRAINED'
        assert manifest['current_course_demo_count']==41 and manifest['current_course_dataset_count']==65
        parent=json.loads((worker/'reports/shared_normal_648_v161/data_manifest.json').read_text())
        by_id={r['dataset_id']:r for r in manifest['scenes']}
        assert all(by_id[r['dataset_id']]==r for r in parent['scenes']) and len(parent['scenes'])==64
        expected.parent.mkdir(parents=True,exist_ok=True);expected.write_bytes(built.read_bytes())
        state.update(manifest_sha256=digest(expected),prepared_demos=41,prepared_datasets=65,
                     unchanged_parent_datasets=64,scope='Preparation only after real five-phase CUDA success; all-course development and changed-weight optimizer/selection still required. Active trained coverage stays40/42.')
        persist('FORTY_ONE_DEMO_COURSE_PREPARED_NOT_TRAINED')
    except Exception as error:
        state['failure']=repr(error);persist('PREFLIGHT_OR_PREPARATION_FAILED_NO_TRAINING');raise


if __name__=='__main__':main()
