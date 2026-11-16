"""Bound combined authored-data training without repeated full-suite prechecks."""
import hashlib
import json
from pathlib import Path

from .native_mixed_course import verify_preflight,REQUIRED_WINDOWS,MIXED_POLICY
from .native_parameter_course import PARAMETER_WINDOWS
from .training_scope import load_scope,validate_rows


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def verify_joint_training_policy(manifest,root,configuration,resume,updates,validation_interval,monitor_interval):
    root=Path(root);policy=manifest.get('joint_course_policy',{})
    if policy.get('kind')!='joint_authored_course_deferred_full_development_v282' or not policy.get('user_requested_combined_training'):
        raise ValueError('Explicit user-directed combined-course policy required')
    if not policy.get('full_development_at_end') or policy.get('pretraining_full_suite_skipped') is not True:
        raise ValueError('Declare deferred full development without fabricating a pretraining PASS')
    rows=manifest['scenes'];scope=load_scope(root);scheduled=validate_rows(rows,scope)
    if scheduled['scheduled_official_demo_count']!=42:raise ValueError('Retain every pure-VBD official demo')
    epochs=policy['epochs'];count=len(rows)
    if type(epochs) is not int or epochs<1 or monitor_interval!=count or validation_interval!=epochs*count:
        raise ValueError('One monitor per complete course and full development at the declared end required')
    if not resume or updates!=resume['valid_updates_recovered']+epochs*count:
        raise ValueError('Exact selected-parent optimizer prefix and update budget required')
    inherited=json.loads((root/policy['inherited_manifest']).read_text())
    if digest(root/policy['inherited_manifest'])!=policy['inherited_manifest_sha256']:
        raise ValueError('Inherited manifest changed')
    by_id={r['dataset_id']:r for r in rows}
    if len(by_id)!=count or not all(by_id.get(r['dataset_id'])==r for r in inherited['scenes']):
        raise ValueError('Every inherited dataset/window must remain unchanged')
    additions=set(by_id)-{r['dataset_id'] for r in inherited['scenes']}
    if additions!=set(policy['cuda_admissions']) or len(additions)!=3:
        raise ValueError('Exactly the loaded mixed asset and both qualified material variants required')
    implementation={p.name:digest(p) for p in (root/'src/wp_vbd').glob('*.py')}
    for dataset,binding in policy['cuda_admissions'].items():
        path=root/binding['path'];row=by_id[dataset]
        if digest(path)!=binding['sha256'] or digest(root/row['path'])!=row['sha256']:
            raise ValueError('Actual new CUDA admission/data changed: '+dataset)
        proof=json.loads(path.read_text())
        verify_preflight(proof,configuration=configuration,checkpoint_sha256=policy['parent_inference_sha256'],
            data_sha256=row['sha256'],implementation_sha256=implementation,
            required_windows=REQUIRED_WINDOWS if dataset=='native_five_families_train_default' else PARAMETER_WINDOWS,
            episode_interactions=dataset!='native_five_families_train_default')
    if not all(configuration.get(key)==value for key,value in MIXED_POLICY.items()):
        raise ValueError('Use the measured composed-material backward/geometry policy')
    return True
