"""Keep each development comparator's physical policy across course extensions."""
from copy import deepcopy
from .development_cases import case_key,validation_rows

PHYSICAL_KEYS=frozenset(('self_retain_accepted_primal_iterate','self_geometric_step_cap','contact_metric_ray_step','elastic_prestress_metric','contact_activation_ray_step','contact_motion_ray_step','contact_structural_support','rod_caller_pose_history','elastic_rod_self_contact','self_contact_refresh_geometry','elastic_rod_refresh_self_geometry','learned_contact_repair'))
RUNTIME_MIGRATION_KEYS=PHYSICAL_KEYS|{'compact_frozen_contact_metric','contact_row_checkpointing',
                                    'checkpoint_relation_ffn_rows','checkpoint_attention_blocks',
                                    'elastic_rod_differentiated_residual_backward'}


def validate_runtime_overrides(overrides):
    """Admit only explicit verified switches or a positive neural row bound."""
    if not isinstance(overrides,dict):raise ValueError('Runtime overrides must be a dictionary')
    for key,value in overrides.items():
        if key not in RUNTIME_MIGRATION_KEYS:raise ValueError('Unsupported runtime override: '+key)
        if key=='checkpoint_relation_ffn_rows':
            if type(value) is not int or value<=0:raise ValueError('Positive integer relation FFN checkpoint bound required')
        elif value is not True:raise ValueError('Runtime switch migrations must explicitly enable a verified policy: '+key)
    return deepcopy(overrides)


def validate_policy(policy):
    if not isinstance(policy,dict) or any(key not in PHYSICAL_KEYS or type(value) is not bool for key,value in policy.items()):
        raise ValueError('Only explicit boolean physical comparison settings are supported')
    return deepcopy(policy)


def verify_pretraining_origin(manifest, proof, resume_evidence):
    """Bind a runtime preflight to its recorded resume or fresh research weights.

    Research initialization carries no selected producer status or optimizer
    history. Its exact initializer and provenance must both match the preflight.
    Existing recorded continuation checks keep their original meaning.
    """
    if resume_evidence:
        if proof.get('parent_checkpoint_sha256') != resume_evidence['checkpoint_sha256']:
            raise ValueError('Pretraining proof belongs to a different resumed checkpoint')
        return 'recorded_checkpoint_continuation'
    origin = manifest.get('research_initialization')
    if not isinstance(origin, dict):
        raise ValueError('A recorded continuation or explicit research initialization is required')
    if (origin.get('weights_copied_exactly') is not True or
            origin.get('optimizer') != 'fresh_AdamW' or
            origin.get('new_optimizer_updates') != 0 or
            manifest.get('initial_probe_updates') != 0 or
            manifest.get('inherited_selected_official_demos') != 0):
        raise ValueError('Research initialization must start with fresh optimizer history and no selected coverage')
    if (proof.get('initial_checkpoint_sha256') != manifest['initial_checkpoint_sha256'] or
            proof.get('research_initialization') != origin or
            proof.get('parent_checkpoint_sha256') is not None):
        raise ValueError('Pretraining proof does not bind this exact research initializer')
    return 'fresh_research_initialization'


def recorded_policies(metrics,metadata):
    policies={}
    for row in metrics:
        demo=case_key(row)
        if demo in policies:raise ValueError('Duplicate development comparator: '+demo)
        if 'baseline_physical_overrides' in row:
            policy=row['baseline_physical_overrides']
        elif demo in metadata.get('per_demo_physical_overrides',{}):
            policy=metadata['per_demo_physical_overrides'][demo]
        elif row.get('comparison_baseline','old32_original_physical_policy')=='old32_original_physical_policy':
            policy={}
        elif row['comparison_baseline']=='old32_matched_physical_policy':
            if not metadata.get('new_demo_baseline_physical_overrides'):
                raise ValueError('Matched comparator is missing its recorded physical settings')
            policy=metadata['new_demo_baseline_physical_overrides']
        else:raise ValueError('Unknown comparison baseline: '+str(row.get('comparison_baseline')))
        policies[demo]=validate_policy(policy)
    return policies


def selected_parent_record(producer):
    status=producer.get('status');step=producer.get('selected_step',0)
    if status not in ('TRAINING_COMPLETED','FAILED_WITH_RETAINED_WEIGHTS','INTERRUPTED_FOR_COURSE_REPLAN') or type(step) is not int or step<=0:
        raise ValueError('A stopped producer with an eligible selected prefix is required')
    if status=='INTERRUPTED_FOR_COURSE_REPLAN' and not producer.get('interruption_reason'):
        raise ValueError('An intentionally interrupted producer must record its actual replan reason')
    if status=='FAILED_WITH_RETAINED_WEIGHTS' and (not producer.get('failure') or type(producer.get('completed_updates')) is not int):
        raise ValueError('A failed producer must retain its actual failure and completed call count')
    if step>producer.get('completed_updates',step):raise ValueError('Selected prefix exceeds completed optimizer calls')
    selected=[r for r in producer['validation_history'] if r['update']==producer['selected_step'] and r['eligible']]
    if len(selected)!=1:raise ValueError('One eligible selected development record required')
    return selected[0]


def extend_policies(producer,scene_rows,model_config):
    selected=selected_parent_record(producer)
    policies=recorded_policies(selected['metrics'],producer.get('development_comparison_policy',{}))
    demos={case_key(row) for row in validation_rows(scene_rows)}
    if not set(policies)<=demos:raise ValueError('An inherited development comparator would be dropped')
    new_policy=validate_policy({key:True for key in PHYSICAL_KEYS if model_config.get(key,False)})
    for demo in demos-set(policies):policies[demo]=deepcopy(new_policy)
    return policies
