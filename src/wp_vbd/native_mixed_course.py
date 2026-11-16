"""Admission checks for the first authored mixed course after all pure demos."""
import math

from .development_baseline import selected_parent_record
from .training_scope import changed_weight_coverage


REQUIRED_WINDOWS=(0,1,600,1800,2800)
MIXED_POLICY={'elastic_rod_self_contact':True,'elastic_rod_refresh_self_geometry':True,
              'elastic_rod_differentiated_residual_backward':True}
INTERACTION_FIELDS=('active_soft_cloth_self_rows','active_dynamic_particle_body_rows',
                    'rod_rows_with_nonzero_wrench','finite_joint_rows_with_nonzero_wrench')


def verify_interaction_exposure(cases,*,episode=False):
    if episode:
        for field in INTERACTION_FIELDS:
            if not any(hop.get(field,0)>0 for row in cases for hop in row.get('actual_evaluated_rows',[])):
                raise ValueError('Authored episode lacks actual evaluated interaction: '+field)
    else:
        for row in cases:
            if row['index'] in (0,1):
                for hop in row['actual_evaluated_rows']:
                    if not all(hop.get(k,0)>0 for k in INTERACTION_FIELDS):
                        raise ValueError('Actual loaded mixed interactions required in the initial windows')
    return True


def selected_official_parent(producer,scope):
    selected=selected_parent_record(producer)
    coverage=changed_weight_coverage(producer['optimization_samples'],scope,selected_step=producer['selected_step'])
    if set(coverage['official_demos'])!=set(scope['target_demos']):
        raise ValueError('The initial authored mixed extension requires a selected parent covering every pure VBD demo')
    if coverage['custom_pure_vbd_datasets']:
        raise ValueError('This builder admits the first mixed asset; later authored assets need distinct development coverage')
    if not all(o['within_cap'] for row in selected['metrics'] for o in row['objects']):
        raise ValueError('Selected parent must preserve every existing development cap')
    return selected,coverage


def verify_preflight(proof,*,configuration,checkpoint_sha256,data_sha256,implementation_sha256,
                     required_windows=REQUIRED_WINDOWS,episode_interactions=False):
    if proof.get('status')!='PASS' or proof.get('device')!='cuda:0':
        raise ValueError('Actual successful CUDA mixed training preflight required; CPU is not admission')
    if proof.get('checkpoint_sha256')!=checkpoint_sha256 or proof.get('data_sha256')!=data_sha256:
        raise ValueError('Mixed preflight must bind this selected parent and exact reference data')
    if proof.get('effective_model_config')!=configuration or proof.get('implementation_sha256')!=implementation_sha256:
        raise ValueError('Mixed preflight must bind the exact model policy and current implementation')
    if not proof.get('adam_moments_resident_during_forward_backward') or not proof.get('all_model_weights_and_adam_unchanged'):
        raise ValueError('Resident unchanged full Adam state required')
    if proof.get('effective_trainable_parameters')!=46882811:
        raise ValueError('The full shared648 network must participate')
    if proof.get('new_optimizer_updates')!=0 or proof.get('new_training_admissions')!=0 or proof.get('full_vbd_prediction_calls')!={'construct':0,'step':0}:
        raise ValueError('Preflight must perform no optimizer or VBD prediction calls')
    if not all(proof.get(k) for k in ('host_reference_trajectories','host_encoder_checkpoint_inputs','host_step_saved_tensors')):
        raise ValueError('Qualified host trajectory and recomputation policy required')
    if proof.get('reference_state_device')!='cpu' or proof.get('duty_cycle')!=.25:
        raise ValueError('Host reference and25 percent compute-window duty required')
    if proof.get('self_contact_broad_phase')!='kdtree' or proof.get('self_contact_motion_budget')!=.02:
        raise ValueError('The exact complete query policy and2cm envelope must be measured')
    gpu=proof.get('gpu_resource_budget',{})
    if not gpu.get('enabled') or any(gpu.get(k)!=v for k,v in
        (('process_budget_gib',12),('torch_allocator_cap_gib',10),('native_reserve_gib',2))):
        raise ValueError('Actual12GiB process/10GiB Torch/native-reserve proof required')
    if not 0<gpu.get('observed_process_max_mib',0)<=12*1024 or gpu.get('process_observations',0)<=0:
        raise ValueError('Observed successful CUDA process memory required')
    cases=proof.get('cases',[])
    if sorted(row.get('index',-1) for row in cases)!=list(required_windows):
        raise ValueError('Every required actual mixed window must pass')
    for row in cases:
        norm=row.get('gradient_norm',0)
        if row.get('parameter_elements_with_grad')!=46882811 or not math.isfinite(norm) or norm<=0:
            raise ValueError('Every mixed window needs complete nonzero shared gradients')
        if not all(math.isfinite(v) and v>0 for v in row.get('backbone_gradient_norms',{}).values()) or set(row.get('backbone_gradient_norms',{}))!={'hierarchy.encoder','hierarchy.decoder','hierarchy.material'}:
            raise ValueError('Encoder, decoder and material gradients required')
        if row.get('own_geometry_queries')!=2 or row.get('backward_geometry_queries')!=0 or not row.get('accepted_history_unchanged_by_backward'):
            raise ValueError('Two causal queries and unchanged backward history required')
        if row.get('optimizer_step_executed') is not False or not 0<row.get('observed_process_gpu_mib',0)<=12*1024:
            raise ValueError('Measured window budget and zero optimizer calls required')
        if len(row.get('actual_evaluated_rows',[]))!=2:
            raise ValueError('Both actual physical hops must be recorded')
    verify_interaction_exposure(cases,episode=episode_interactions)
    return True
