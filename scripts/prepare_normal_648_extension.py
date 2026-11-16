"""Extend the shared648 course with qualified complete assets and replay."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
    return h.hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent-stage',default='shared_normal_648_v84')
    parser.add_argument('--report',type=Path,default=ROOT/'reports/shared_normal_648_v88/data_manifest.json')
    parser.add_argument('--course',choices=('mixed','large_normal','coupled_normal','gift_normal','dat_normal','dat_twist_normal','poker_normal','matched_rod_normal','rod_joint_normal','rod_contact_normal','rollers_normal','rj45_caller_normal'),default='mixed')
    parser.add_argument('--selected-export',type=Path,help='Explicit selected-prefix export from a stopped parent course')
    parser.add_argument('--preflight-directory',type=Path,help='Read new-asset runtime proofs from this directory instead of the parent stage')
    parser.add_argument('--contact-metric-ray-step',action='store_true',help='Explicitly migrate the contact proposal policy; requires matching effective-runtime proofs and pretraining development checks')
    parser.add_argument('--elastic-prestress-metric',action='store_true',help='Explicitly migrate the elastic metric; requires matching effective-runtime proofs and unchanged inherited development caps')
    parser.add_argument('--compact-frozen-contact-metric',action='store_true')
    parser.add_argument('--checkpoint-contact-rows',action='store_true')
    parser.add_argument('--contact-activation-ray-step',action='store_true')
    parser.add_argument('--contact-motion-ray-step',action='store_true')
    parser.add_argument('--contact-structural-support',action='store_true')
    parser.add_argument('--caller-pose-history',action='store_true')
    parser.add_argument('--checkpoint-relation-ffn-rows',type=int,default=0)
    parser.add_argument('--checkpoint-attention-blocks',action='store_true')
    parser.add_argument('--host-step-saved-tensors',action='store_true')
    parser.add_argument('--extension-learning-rate',type=float,help='Declare a uniform rate change while retaining selected Adam moments and RNG')
    parser.add_argument('--learning-rate-reason',help='Required explanation for an explicit continuation rate change')
    args=parser.parse_args()
    rod_courses=('matched_rod_normal','rod_joint_normal','rod_contact_normal')
    if args.course=='rod_contact_normal' and not (args.checkpoint_contact_rows and args.host_step_saved_tensors):
        parser.error('The complete contact-rod course requires explicit row recomputation and host step storage')
    if args.checkpoint_relation_ffn_rows<0:parser.error('Nonnegative relation FFN checkpoint bound required')
    if args.course=='rollers_normal' and not (args.contact_motion_ray_step and args.checkpoint_contact_rows and args.host_step_saved_tensors and args.checkpoint_relation_ffn_rows>0 and args.checkpoint_attention_blocks):
        parser.error('Complete rollers require the explicit remaining-motion policy and qualified full-backbone memory settings')
    if args.course=='rj45_caller_normal' and not (args.caller_pose_history and args.contact_structural_support and args.contact_activation_ray_step and args.contact_motion_ray_step and args.checkpoint_contact_rows and args.host_step_saved_tensors):
        parser.error('Caller mesh course requires explicit qualified pose/contact policies and host/checkpoint storage')
    if (args.extension_learning_rate is None)!=(args.learning_rate_reason is None):parser.error('A learning-rate change and its reason must be supplied together')
    if args.report.exists():raise FileExistsError(args.report)
    parent=ROOT/'reports'/args.parent_stage
    export_path=args.selected_export if args.selected_export else parent/'inference_checkpoint.json'
    producer=json.loads((parent/'training.json').read_text())
    assert producer['status'] in ('TRAINING_COMPLETED','FAILED_WITH_RETAINED_WEIGHTS','STOPPED_FOR_GPU_INFERENCE_BENCHMARK'),'A stopped terminal parent course is required before preparing a continuation'
    evidence=json.loads(export_path.read_text())
    from wp_vbd.development_baseline import extend_policies,selected_parent_record
    parent_selected=selected_parent_record(producer)
    expected_config=copy.deepcopy(producer['config'])
    if args.contact_metric_ray_step:expected_config['contact_metric_ray_step']=True
    if args.elastic_prestress_metric:expected_config['elastic_prestress_metric']=True
    if args.compact_frozen_contact_metric:expected_config['compact_frozen_contact_metric']=True
    if args.checkpoint_contact_rows:expected_config['contact_row_checkpointing']=True
    if args.contact_activation_ray_step:expected_config['contact_activation_ray_step']=True
    if args.contact_motion_ray_step:expected_config['contact_motion_ray_step']=True
    if args.contact_structural_support:expected_config['contact_structural_support']=True
    if args.caller_pose_history:expected_config['rod_caller_pose_history']=True
    if args.checkpoint_relation_ffn_rows:expected_config['checkpoint_relation_ffn_rows']=args.checkpoint_relation_ffn_rows
    if args.checkpoint_attention_blocks:expected_config['checkpoint_attention_blocks']=True
    runtime_overrides={k:v for k,v in expected_config.items() if v!=producer['config'].get(k)}
    proof_directory=ROOT/args.preflight_directory if args.preflight_directory else parent
    assert producer['selected_step']==evidence['selected_updates']<=producer['completed_updates']
    # An export made while later updates ran pins its immutable report
    # snapshot. Bind that snapshot to the stopped producer's same selected
    # prefix rather than comparing its hash to the later terminal report.
    export_report_path=ROOT/evidence['source_training_report'] if evidence.get('source_training_report') else parent/'training.json'
    assert digest(export_report_path)==evidence['source_training_report_sha256']
    export_report=json.loads(export_report_path.read_text())
    assert export_report['source_git_commit']==producer['source_git_commit'] and export_report['config']==producer['config']
    selected_updates=evidence['selected_updates']
    assert [r for r in export_report['optimization_samples'] if r['update']<=selected_updates]==[r for r in producer['optimization_samples'] if r['update']<=selected_updates]
    assert [v for v in export_report['validation_history'] if v['update']==selected_updates and v['eligible']]==[parent_selected]
    trial=ROOT/evidence['source_trial_checkpoint']
    assert digest(trial)==evidence['source_trial_checkpoint_sha256']
    manifest=copy.deepcopy(json.loads((parent/'data_manifest.json').read_text()))
    old_rows=manifest['scenes']
    full=json.loads((ROOT/'reports/shared_particle_hanging_v63/data_manifest.json').read_text())['current_optimization_scenes']
    additional=[];proofs={}
    courses={
        'mixed':{'soft_rigid_train_default':('mixed','soft_rigid_full_backbone_cuda.json'),
                 'soft_rigid_mix_train_default':('mixed','soft_rigid_mix_full_backbone_cuda.json')},
        'large_normal':{'urdf_train_default':('joint','urdf_full_backbone_cuda.json'),
                        'cloth_hanging_self_train_default':('self','hanging_full_backbone_cuda.json')},
        'coupled_normal':{'rigid_soft_train_default':('mixed','rigid_soft_full_backbone_cuda.json'),
                          'soft_cloth_train_default':('self','soft_cloth_full_backbone_cuda.json')},
        'gift_normal':{'softbody_gift_train_default':('self','gift_full_backbone_cuda.json')},
        'dat_normal':{'dat_rigid_soft_train_default':('self','dat_full_backbone_cuda.json')},
        'dat_twist_normal':{'dat_rigid_soft_train_default':('self','dat_full_backbone_cuda.json'),
                            'cloth_twist_self_train_default':('self','twist_full_backbone_cuda.json')},
        'poker_normal':{'poker_cards_prescribed_train_default':('self','full_backbone_cuda.json')},
        'matched_rod_normal':{'cable_plectoneme_train_default':('rod','plectoneme_full_backbone_cuda.json'),
                              'cable_twist_buckling_train_default':('rod','buckling_full_backbone_cuda.json')},
        'rod_joint_normal':{'cable_cross_slide_table_train_default':('rod','cross_slide_table_full_backbone_cuda.json')},
        'rollers_normal':{'cloth_rollers_train_default':('self','full_backbone_five_phases_cuda.json')},
        'rj45_caller_normal':{'contacts_rj45_plug_train_default':('rod','full_backbone_cuda.json')}}
    courses['rod_contact_normal']={**courses['matched_rod_normal'],**courses['rod_joint_normal'],
                                  'cable_pile_train_default':('rod','pile_full_backbone_cuda.json')}
    pairs=courses[args.course]
    complete_courses=('coupled_normal','gift_normal','dat_normal','dat_twist_normal','poker_normal','rollers_normal','rj45_caller_normal',*rod_courses)
    if args.course in complete_courses:
        specifications_by_course={
            'coupled_normal':(('rigid_soft_train_default','rigid_soft_v72/data_qualification.json','MIXED_TEACHER_DATA_ADMITTED',
                         [800,900],[1100]),
                        ('soft_cloth_train_default','soft_cloth_v66/data_qualification.json','SELF_CONTACT_TEACHER_DATA_QUALIFIED',
                         [400,800],[1600])),
            'gift_normal':(('softbody_gift_train_default','softbody_gift_v93/data_qualification.json','SELF_CONTACT_TEACHER_DATA_QUALIFIED',
                         [1500,1800,2100],[3000]),),
            'dat_normal':(('dat_rigid_soft_train_default','rigid_soft_self_v94/data_qualification.json','SELF_CONTACT_TEACHER_DATA_QUALIFIED',
                         [100,300,600],[1000]),),
            'poker_normal':(('poker_cards_prescribed_train_default','poker_cards_v123/data_qualification.json','SELF_CONTACT_TEACHER_DATA_QUALIFIED',
                             [743,900,1800],[5000]),),
            'rollers_normal':(('cloth_rollers_train_default','cloth_rollers_v114/data_qualification.json','SELF_CONTACT_TEACHER_DATA_QUALIFIED',
                               [0,4,700,1400],[3000]),),
            'rj45_caller_normal':(('contacts_rj45_plug_train_default','rj45_caller_v216/data_qualification.json','RJ45_AUTHORED_CAUSAL_TEACHER_QUALIFIED_FOR_MESH_PREFLIGHT',
                                   [0,5,240,480],[660]),),
            'matched_rod_normal':(('cable_plectoneme_train_default','rod_matched_v134/plectoneme_data_qualification.json','ROD_CONTACT_TEACHER_DATA_QUALIFIED',
                                   [4300,4797],[5800]),
                                  ('cable_twist_buckling_train_default','rod_matched_v134/buckling_data_qualification.json','ROD_CONTACT_TEACHER_DATA_QUALIFIED',
                                   [1000,2200],[3500])),
            'rod_joint_normal':(('cable_cross_slide_table_train_default','rod_joint_v144/cross_slide_table_data_qualification.json','ROD_CONTACT_TEACHER_DATA_QUALIFIED',
                                [1200,4800],[8500]),)}
        specifications_by_course['dat_twist_normal']=specifications_by_course['dat_normal']+(
            ('cloth_twist_self_train_default','cloth_self_contact_v64/twist_data_qualification.json','SELF_CONTACT_TEACHER_DATA_QUALIFIED',
             [2000,2627],[3273]),)
        specifications_by_course['rod_contact_normal']=(specifications_by_course['matched_rod_normal']+
            specifications_by_course['rod_joint_normal']+(
                ('cable_pile_train_default','pile_v146/data_qualification.json','ROD_CONTACT_TEACHER_DATA_QUALIFIED',
                 [300,900],[2400]),))
        specifications=specifications_by_course[args.course]
        for label,stage,status,starts,validation in specifications:
            qualification_path=ROOT/'reports'/stage
            qualification=json.loads(qualification_path.read_text())
            final_key=('source_original_final_check' if args.course=='rj45_caller_normal' else
                       'original_source_final_check' if args.course in rod_courses else 'source_final_check')
            assert qualification['status']==status and qualification[final_key]=='PASS'
            path=Path(qualification['path']);path=path if path.is_absolute() else ROOT/path
            assert digest(path)==qualification['sha256']
            full.append({'dataset_id':label,'demo':qualification['demo'],'path':str(path.relative_to(ROOT)),
                         'sha256':qualification['sha256'],'teacher_iterations':qualification.get('teacher_iterations',10),
                         'split':'train_temporal_windows_and_separate_development_windows',
                         'qualified_train_starts':starts,'validation_starts':validation,
                         'data_qualification':str(qualification_path.relative_to(ROOT)),
                         'data_qualification_sha256':digest(qualification_path)})
            if args.course in ('dat_normal','dat_twist_normal','poker_normal'):full[-1]['self_contact_max_rows']=1000000
            if args.course=='rollers_normal':
                assert qualification['particle_count']==5824 and qualification['body_count']==0
                full[-1].update(self_contact_max_rows=3000000,self_contact_responsive_pairs_only=True,
                                self_contact_motion_budget=.8)
            if args.course in rod_courses:
                import numpy as np
                with np.load(path,allow_pickle=False) as archive:metadata=json.loads(str(archive['metadata']))
                assert metadata['matched_contact_training_schema']==qualification['matched_contact_training_schema']=='accepted_matcher_cache_v133'
                assert metadata['reference_final_check']=='PASS' and metadata['cross_step_contact_history']
                full[-1]['teacher_iterations']=metadata['teacher_iterations']
                if label=='cable_cross_slide_table_train_default':
                    assert metadata['finite_rod_training_schema']==qualification['finite_rod_training_schema']=='accepted_joint_controls_winding_v144'
                    assert qualification['finite_joint_count']>0
            if args.course=='rj45_caller_normal':
                import numpy as np
                from wp_vbd.rigid_caller_program import SCHEMA
                with np.load(path,allow_pickle=False) as archive:metadata=json.loads(str(archive['metadata']))
                assert metadata['rigid_caller_training_schema']==SCHEMA and metadata['reference_final_check']=='PASS'
                assert metadata['rigid_mesh_reference_schema']=='authored_sdf_mesh_identity_and_reference_mass_v216'
                assert not metadata['cross_step_contact_history'] and len(metadata['mesh_geometry_identity'])==3
                assert qualification['teacher_substeps']==metadata['teacher_substeps']==720 and metadata['teacher_frames']==120
                assert qualification['body_count']==37 and qualification['rod_count']==34 and qualification['finite_joint_count']==2
                full[-1]['teacher_iterations']=metadata['teacher_iterations']
            manifest['source_manifests_sha256'][str(qualification_path.relative_to(ROOT))]=digest(qualification_path)
    for label,(loader,filename) in pairs.items():
        row=copy.deepcopy(next(r for r in full if r['dataset_id']==label))
        if args.course=='dat_twist_normal':
            proof_path=ROOT/({'dat_rigid_soft_train_default':'reports/shared_normal_648_v112/dat_full_backbone_cuda.json',
                              'cloth_twist_self_train_default':'reports/cloth_twist_v117/full_backbone_cuda.json'}[label])
        else:proof_path=proof_directory/filename
        proof=json.loads(proof_path.read_text())
        assert proof['status'] in ('PASS','GENERAL_JOINT_FULL_BACKBONE_WINDOWS_DIAGNOSED') and proof['device']=='cuda:0'
        assert proof['checkpoint_sha256']==evidence['checkpoint_sha256']
        if runtime_overrides or args.course in ('gift_normal','dat_normal','dat_twist_normal','poker_normal','rollers_normal','rj45_caller_normal',*rod_courses):
            assert proof['effective_model_config']==expected_config,'Preflight must use the exact effective runtime configuration'
            assert proof['model_configuration_overrides']==runtime_overrides
        if bool(proof.get('host_step_saved_tensors',False))!=args.host_step_saved_tensors:
            raise ValueError('Preflight host step storage and the declared resource policy differ')
        assert proof['data_sha256']==row['sha256']==digest(ROOT/row['path'])
        if loader=='mixed':
            assert all(r['gradient_scope']=='full_shared_backbone' and max(r['active_dynamic_particle_body_rows'])>0
                       and all(r['module_gradient_norms'][name]>0 for name in ('protected_response.hierarchy.encoder','protected_response.hierarchy.decoder','protected_response.hierarchy.material')) for r in proof['results'])
        elif loader=='joint':
            assert proof['effective_trainable_parameters']==46882811
            assert all(r['two_steps_accepted'] and r['full_backbone_gradient_norm_sum']>0 for r in proof['cases'])
        elif loader=='rod':
            assert proof['effective_trainable_parameters']==46882811 and proof['full_shared_backbone_gradients_enabled']
            assert all(r['two_hop_accepted'] and r['shared_head_gradient_norm']>0
                       and all(value>0 for value in r['backbone_gradient_norms'].values()) for r in proof['results'])
            assert proof['parameter_versions_unchanged']
            if args.course=='rj45_caller_normal':
                assert not proof['cross_step_contact_history'] and proof['rigid_caller_training_schema']==SCHEMA
                assert proof['implementation_sha256']=={p.name:digest(p) for p in (ROOT/'src/wp_vbd').glob('*.py')}
                assert proof['new_training_admissions']==0
            else:assert proof['cross_step_contact_history']
        else:
            assert proof['effective_trainable_parameters']==46882811 and proof['full_shared_backbone_gradients_enabled']
            assert all(r['two_hop_accepted'] and all(v>0 for v in r['backbone_gradient_norms'].values()) for r in proof['cases'])
        assert proof['new_optimizer_updates']==0 and proof['full_vbd_prediction_calls']==0
        assert proof['gpu_resource_budget']['observed_process_max_mib']<=12*1024
        if args.course in complete_courses:
            cases=proof['results'] if loader in ('mixed','rod') else proof['cases']
            assert sorted(r['index'] for r in cases)==sorted(row['qualified_train_starts']+row['validation_starts'])
            assert proof['effective_trainable_parameters']==46882811 and proof['adam_moments_resident_during_forward_backward']
            if loader=='rod':
                assert proof['host_rod_trajectories'] and proof['reference_state_device']=='cpu'
                assert proof['host_encoder_checkpoint_inputs'] and proof['encoder_recomputation_enabled']
                assert all(len(r['actual_evaluated_rows'])==2 and all(step['rod_rows_with_nonzero_wrench']>0 for step in r['actual_evaluated_rows']) for r in cases)
                if label in ('cable_plectoneme_train_default','cable_cross_slide_table_train_default','cable_pile_train_default'):
                    assert all(any(step['active_rigid_contact_rows']>0 for step in r['actual_evaluated_rows']) for r in cases)
                    assert all(max(r['matched_rigid_contact_rows'])>0 for r in cases)
                if label=='cable_cross_slide_table_train_default':
                    assert all(all(step['finite_joint_rows_with_nonzero_wrench']>0 for step in r['actual_evaluated_rows']) for r in cases)
                if args.course=='rj45_caller_normal':
                    assert all(r['backward_geometry_queries']==0 and r['accepted_history_versions_unchanged_by_backward']
                        and not r['optimizer_step_executed'] and r['nonfinite_gradient_entries']==0 for r in cases)
                    # Authored headless RJ45 can have zero finite-joint or mesh
                    # load; do not invent load coverage from candidate counts.
                    assert all(all('finite_joint_rows_with_nonzero_wrench' in step for step in r['actual_evaluated_rows']) for r in cases)
                assert proof['duty_cycle']==.25
                gpu=proof['gpu_resource_budget']
                assert gpu['enabled'] and gpu['process_budget_gib']==12 and gpu['torch_allocator_cap_gib']==10 and gpu['native_reserve_gib']==2
            elif loader=='mixed':assert proof['host_mixed_trajectories']
            else:
                assert proof['broad_phase']=='kdtree'
                assert all(all(r['self_relation_embedding_gradient_norms'][str(k)]>0 for k in (9,10)) for r in cases)
                if label=='dat_rigid_soft_train_default':
                    assert all(r['active_dynamic_particle_body_rows']>0 and all(r['active_self_rows_by_kind'][str(k)]>0 for k in (9,10)) for r in cases)
                elif label=='cloth_twist_self_train_default':
                    assert all(all(r['active_self_rows_by_kind'][str(k)]>0 for k in (9,10)) for r in cases)
                elif label=='poker_cards_prescribed_train_default':
                    assert all(all(r['active_self_rows_by_kind'][str(k)]>0 for k in (9,10)) for r in cases)
                    assert any(r['active_prescribed_particle_body_rows_by_body'].get('1',0)>0 for r in cases),'The moving sphere must carry real particle contact load'
                    assert proof.get('host_encoder_checkpoint_inputs') and proof['encoder_recomputation_enabled']
                elif label=='cloth_rollers_train_default':
                    assert all(all(r['active_self_rows_by_kind'][str(k)]>0 for k in (9,10)) for r in cases)
                    assert proof['host_encoder_checkpoint_inputs'] and proof['encoder_recomputation_enabled']
                    assert proof['host_self_trajectories'] and proof['reference_state_device']=='cpu'
                    assert proof['responsive_self_pairs_only'] and proof['self_contact_pair_scope']=='at_least_one_responsive_primitive_vertex'
                    # Legacy report key carries the source coordinate unit;
                    # this authored scene uses cm (0.8 source units = 0.008m).
                    assert proof['self_contact_motion_budget_m']==.8 and proof['self_candidate_capacity']==3000000
                    assert proof['checkpoint_relation_ffn_rows']==args.checkpoint_relation_ffn_rows and proof['checkpoint_attention_blocks']
                    assert proof['host_allocator_trimming_enabled'] and proof['reference_storage']['constant_time_storage_enabled']
                    assert proof['reference_storage']['unique_tensor_storage_bytes']<proof['reference_storage']['logical_bytes']
                    assert proof['gpu_compute_window_duty_cycle']==.25
                    gpu=proof['gpu_resource_budget']
                    assert gpu['enabled'] and gpu['process_budget_gib']==12 and gpu['torch_allocator_cap_gib']==10 and gpu['native_reserve_gib']==2
                    assert all(r['all_requested_steps_accepted'] and r['prediction_horizon']==2 and r['backward_geometry_queries']==0 and r['accepted_history_unchanged_by_backward'] for r in cases)
                    assert all(len(r['contact_motion_ray_scales'])==2 and all(len(hop)==5 and all(all(cap['valid']) for cap in hop) for hop in r['contact_motion_ray_scales']) for r in cases)
                else:assert all(r['active_soft_cloth_self_rows']>0 for r in cases)
                if args.course in ('gift_normal','dat_normal','dat_twist_normal','poker_normal'):
                    assert proof['host_self_trajectories'] and proof['reference_state_device']=='cpu'
                    if expected_config.get('contact_metric_ray_step'):
                        assert all(len(r['contact_metric_ray_scales'])==2 and all(len(step)==4 for step in r['contact_metric_ray_scales']) for r in cases)
        stat=(ROOT/row['path']).stat()
        row.update(loader=loader,admitted_file_stat={'bytes':stat.st_size,'mtime_ns':stat.st_mtime_ns})
        additional.append(row);proofs[str(proof_path.relative_to(ROOT))]=digest(proof_path)
    rows=additional+old_rows
    from wp_vbd.training_scope import load_scope,validate_rows
    training_scope=validate_rows(rows,load_scope(ROOT))
    assert len({r['dataset_id'] for r in rows})==len(rows)==len(old_rows)+len(pairs)
    for row in rows:
        assert digest(ROOT/row['path'])==row['sha256']
        if row.get('validation_path'):assert digest(ROOT/row['validation_path'])==row['validation_sha256']
    demos=sorted({r['demo'] for r in rows if r.get('demo')})
    if args.course=='rj45_caller_normal':
        assert set(demos)==set(load_scope(ROOT)['target_demos']),'The final caller extension must retain all42 pure-VBD demos'
    assert len(demos)==evidence['trained_official_demo_count']+len(pairs) and set(evidence['trained_official_demos'])<set(demos)
    baseline_sha=digest(ROOT/'checkpoints/shared_particle_hanging_v63/shared_strict.pt')
    if producer.get('development_comparison_policy',{}).get('old32_checkpoint_sha256'):
        assert producer['development_comparison_policy']['old32_checkpoint_sha256']==baseline_sha
    baseline_policies=extend_policies(producer,rows,expected_config)
    manifest.update(status='NORMAL_COURSE_EXTENSION_PREPARED_NOT_TRAINED',
                    training_scope=training_scope,
                    scenes=rows,current_course_official_demos=demos,current_course_demo_count=len(demos),
                    current_course_dataset_count=len(rows),inherited_selected_official_demos=evidence['trained_official_demo_count'],
                    old32_demo_coverage_inherited=0,
                    effective_model_config=expected_config,model_configuration_overrides=runtime_overrides,
                    development_baseline_policies=baseline_policies,development_baseline_checkpoint_sha256=baseline_sha,
                    deferred_previous_dataset_ids=sorted(set(manifest['deferred_previous_dataset_ids'])-set(pairs)),
                    selected_parent={'checkpoint':str(trial.relative_to(ROOT)),'checkpoint_sha256':digest(trial),
                                     'report':str((parent/'training.json').relative_to(ROOT)),'report_sha256':digest(parent/'training.json'),
                                     'inference_checkpoint_sha256':evidence['checkpoint_sha256'],'selected_updates':evidence['selected_updates']},
                    new_dataset_preflight_sha256=proofs,
                    selected_export={'evidence':str(export_path.relative_to(ROOT)) if export_path.is_absolute() else str(export_path),
                                     'evidence_sha256':digest(export_path),
                                     'snapshot':str(export_report_path.relative_to(ROOT)),
                                     'snapshot_sha256':digest(export_report_path)},
                    scope=f'Continue the same selected648-wide model and Adam state with {len(pairs)} qualified {args.course} assets plus unchanged parent replay. Only the selected parent prefix is inherited; later unselected optimizer calls remain separate. No new coverage until changed-weight samples and expanded development selection pass. Current pure-VBD target42 and full native quality/realtime remain required.')
    manifest['source_manifests_sha256'][str((parent/'data_manifest.json').relative_to(ROOT))]=digest(parent/'data_manifest.json')
    if args.extension_learning_rate is not None:
        from wp_vbd.training_schedule import learning_rate_migration
        manifest['optimizer_learning_rate_migration']=learning_rate_migration(
            producer['optimizer']['learning_rate'],args.extension_learning_rate,args.learning_rate_reason)
    else:manifest.pop('optimizer_learning_rate_migration',None)
    if args.course=='poker_normal':
        manifest['resource_policy']={**manifest.get('resource_policy',{}),'host_encoder_checkpoint_inputs':True}
    if args.course in (*rod_courses,'rj45_caller_normal'):
        manifest['resource_policy']={**manifest.get('resource_policy',{}),'host_encoder_checkpoint_inputs':True,'host_rod_trajectories':True}
    if args.course=='rollers_normal':
        manifest['resource_policy']={**manifest.get('resource_policy',{}),'host_encoder_checkpoint_inputs':True,
            'host_self_trajectories':True,'compact_host_reference_constants':True,'trim_host_allocator':True}
    if args.checkpoint_relation_ffn_rows:
        manifest['resource_policy']={**manifest.get('resource_policy',{}),'checkpoint_relation_ffn_rows':args.checkpoint_relation_ffn_rows}
    if args.checkpoint_attention_blocks:
        manifest['resource_policy']={**manifest.get('resource_policy',{}),'checkpoint_attention_blocks':True}
    if args.host_step_saved_tensors:
        manifest['resource_policy']={**manifest.get('resource_policy',{}),'host_step_saved_tensors':True}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k not in ('scenes','current_course_official_demos')},indent=2))


if __name__=='__main__':main()
