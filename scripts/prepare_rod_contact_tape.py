"""Qualify complete fresh-contact rod references with causal new-model states."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), '/home/oem/code/engine/newton']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tape', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--expected-frames', type=int, default=360)
    parser.add_argument('--fresh-rod-inputs',action='store_true')
    parser.add_argument('--dahl-history',action='store_true')
    parser.add_argument('--matched-contact-history',action='store_true')
    parser.add_argument('--finite-joints',action='store_true',help='Explicit composed rod/finite-joint controls and accepted winding schema')
    parser.add_argument('--bounded-analytic-contact-buffers',action='store_true',help='Declare complete current-query AABB allocation and sparse cold accepted history')
    parser.add_argument('--bundle-release-events',action='store_true'); args = parser.parse_args()
    if args.bundle_release_events and not (args.fresh_rod_inputs and args.dahl_history):
        parser.error('Bundle release requires explicit fresh inputs and Dahl memory')
    if args.finite_joints and not args.fresh_rod_inputs:parser.error('Composed finite joints require the explicit full accepted pose input contract')
    if args.bounded_analytic_contact_buffers and not args.matched_contact_history:parser.error('Bounded analytic allocation requires complete authored contact history')
    if args.expected_frames <= 0: parser.error('--expected-frames must be positive')
    if args.output.exists() or args.report.exists(): raise FileExistsError('Preserve data qualifications')
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.reference_tape import read_tape
    from wp_vbd.rod_contact_scene import export_model
    from wp_vbd.rod_contact_data import SOURCE_FIELDS,FINITE_SOURCE_FIELDS
    from wp_vbd.rod_contact_system import strain
    from wp_vbd.joint_history import rod_strain_increment
    from wp_vbd.data import State
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = str(ROOT/'.local/p0/warp-cache'); wp.init(); wp.set_device('cpu')
    manifest_path = args.tape/'manifest.json'; manifest = json.loads(manifest_path.read_text())
    assert manifest['status'] == 'PHYSICAL_SUBSTEP_TRACE_READY' and manifest['vbd_instances'] == 1
    assert manifest['source_sha'] == '25298aea1691e060c86f95e8d2b773c898c006d3'
    if args.fresh_rod_inputs and not args.matched_contact_history:
        assert manifest['collision_refresh_policy']=='every_physical_substep'
        assert manifest['collision_policy_override']['physical_material_asset_control_preserved']
    example, solvers, _ = make_example(manifest['id'], reference=False, preserve_defaults=True)
    configuration = solvers[0][1]; assert configuration == manifest['solvers'][0]['configuration']
    model = example.model; n = model.body_count; dt = example.sim_dt
    contact_policy=None
    if args.matched_contact_history:
        from wp_vbd.rigid_contact_memory import authored_policy
        assert manifest.get('authored_final_check')=='PASS' and configuration['rigid_contact_history']
        assert manifest['collision_refresh_policy']=='authored' and not manifest.get('collision_policy_override')
        contact_policy=authored_policy(example)
    if args.finite_joints:
        from wp_vbd.rod_joint_scene import export_model
    export_options={'contact_policy':contact_policy}
    if args.bounded_analytic_contact_buffers:export_options['bounded_analytic_buffers']=True
    scene = export_model(model, configuration, manifest['id'], newton, dt, 'cpu',**export_options)
    fixed = np.flatnonzero(~scene.movable[:-1].numpy())
    mobile = np.flatnonzero(scene.movable[:-1].numpy())
    fixed_pose = model.body_q.numpy()[fixed].copy()
    if args.fresh_rod_inputs: assert len(fixed)>0
    if args.dahl_history:
        assert bool((scene.rod.rod_dahl_eps>0).any()), 'Dahl qualification needs active material memory'
    else:assert not bool(scene.rod.rod_dahl_eps.any()), 'Nonzero Dahl requires explicit physical-memory import'
    static_path = args.tape/manifest['static']['path']
    assert hashlib.sha256(static_path.read_bytes()).hexdigest() == manifest['static']['sha256']
    with np.load(static_path, allow_pickle=False) as archive:
        source = {key.removeprefix('solver0__model__'):archive[key].copy() for key in archive.files if key.startswith('solver0__model__')}
        if args.bundle_release_events:
            assert manifest['id']=='cable.example_cable_bundle_hysteresis' and manifest.get('controller_clock_contract')
            command_time=float(archive['controller_initial__sim_time_array'][0]);assert command_time==0
    if args.finite_joints:
        assert model.joint_target_q_start is model.joint_q_start
        source['joint_target_q_start']=source['joint_q_start'].copy()
        for name in FINITE_SOURCE_FIELDS:
            assert np.array_equal(getattr(model,name).numpy(),source[name]),name
        spec=scene.joint_spec;valid=spec.dof_index.numpy()>=0
        used=np.zeros(model.joint_dof_count,bool);used[spec.dof_index.numpy()[valid]]=True
        rod_q_start=model.joint_target_q_start.numpy()[scene.rod.source_joint_ids.numpy()]
        rod_qd_start=model.joint_qd_start.numpy()[scene.rod.source_joint_ids.numpy()]
        rod_target_indices=rod_q_start[:,None]+np.arange(7)
        rod_velocity_indices=rod_qd_start[:,None]+np.arange(6)
        joint_inputs=[]
    if args.bundle_release_events:
        from wp_vbd.rod_prescribed_events import BundleReleaseEvents
        removal=BundleReleaseEvents(example,scene,wp)
        assert np.array_equal(fixed,removal.ids.numpy())
        expected_removal=removal.expected.numpy();removal_masks=[];command_times=[]
    ids = scene.rod.source_joint_ids.numpy()
    if args.matched_contact_history:
        from wp_vbd.rigid_contact_memory import initial_history
        from wp_vbd.rod_reference_memory import accepted_cache
        storage_capacity=max(1,max(int(fields['matching_after__pipeline0___prev_count'][0]) for fields,_ in read_tape(args.tape)))
        initial=scene.initial_state(model,wp)
        contact_history=[{name:value[0].numpy() for name,value in initial_history(scene,initial,capacity=storage_capacity).items()}]
        previous_numeric=None;previous_generation=None
        # The raw trace predates serialization of this public set wrapper.
        # Bind it to the pinned authored constructor and identical source args;
        # do not claim byte equality to a field absent from the original tape.
        source['shape_collision_filter_pairs']=model.shape_collision_filter_pairs.pairs_array().copy()
    def convert(pose, velocity):
        q = pose[:, 3:].astype(np.float64); q /= np.linalg.norm(q, axis=-1, keepdims=True)
        com = scene.body_com[:-1].numpy()
        x = pose[:, :3]+com+2*np.cross(q[:, :3], np.cross(q[:, :3], com)+q[:, 3:]*com)
        return {'x':np.concatenate((x,np.zeros((1,3)))), 'q':np.concatenate((q,np.asarray([[0.,0.,0.,1.]]))),
                'v':np.concatenate((velocity[:, :3].astype(np.float64),np.zeros((1,3)))),
                'omega':np.concatenate((velocity[:, 3:].astype(np.float64),np.zeros((1,3))))}
    def dual(data, prefix): return np.concatenate((data[prefix+'__joint_lambda_lin'][ids],data[prefix+'__joint_lambda_ang'][ids]),-1).astype(np.float64)
    states = []; duals = []; forces = []; torques = []; previous = None; calls = 0; candidates = []
    pin_x=[];pin_q=[]
    for data, timing in read_tape(args.tape):
        assert timing['sequence'] == calls and timing['solver_slot'] == 0 and timing['dt'] == dt
        assert bool(data['aux_before___update_rigid_history']), 'Retained collision history is a separate contract'
        if args.matched_contact_history:
            names=('_prev_contact_lambda','_prev_contact_normal','_prev_contact_frame')
            if previous_numeric is not None:
                assert all(np.array_equal(previous_numeric[name],data['aux_before__'+name]) for name in names)
            else:assert not data['aux_before___prev_contact_lambda'].any(),'Initial source numeric history must be cold'
            generation=int(data['contacts_before__contact_generation'][0])
            if previous_generation is not None:assert generation==previous_generation+1,'One fresh authored query per physical call required'
            previous_generation=generation
            contact_history.append(accepted_cache(scene,initial,data,storage_capacity))
            previous_numeric={name:data['aux_after__'+name] for name in names}
        incoming = (data['input__body_q'],data['input__body_qd']); output = (data['output__body_q'],data['output__body_qd'])
        assert all(np.isfinite(value).all() for value in (*incoming,*output))
        if args.fresh_rod_inputs:
            assert np.array_equal(output[0][fixed],incoming[0][fixed]), 'Source changed the prescribed pose during dynamics'
            command=convert(*incoming);pin_x.append(command['x'][fixed]);pin_q.append(command['q'][fixed])
            if args.bundle_release_events:
                clock=float(data['controller_before__sim_time_array'][0])
                assert clock>command_time and np.isclose(clock-command_time,dt,rtol=2e-4,atol=1e-7)
                prior=fixed_pose if previous is None else previous[0][fixed]
                mask=(incoming[0][fixed]!=prior).any(-1) & (incoming[0][fixed]==expected_removal).all(-1)
                assert not mask.any() or command_time>=example.obstacle_release_time
                if command_time>=example.obstacle_release_time:assert np.array_equal(incoming[0][fixed],expected_removal)
                removal_masks.append(mask);command_times.append(command_time);command_time=clock
        else:
            for pose, velocity in (incoming, output):
                assert np.array_equal(pose[fixed], fixed_pose), 'Moving prescribed bodies need their explicit causal input preparation'
                assert not velocity[fixed].any(), 'Stationary prescribed bodies must have zero velocity'
        if previous is None:
            pose=incoming[0].copy()
            if args.fresh_rod_inputs:pose[fixed]=fixed_pose
            states.append(convert(pose,incoming[1]));duals.append(dual(data,'aux_before'))
        else:
            if args.fresh_rod_inputs:
                assert np.array_equal(previous[0][mobile],incoming[0][mobile])
                assert np.array_equal(previous[1],incoming[1]), 'This pose-driven contract does not admit velocity edits'
            else:assert all(np.array_equal(a,b) for a,b in zip(previous,incoming))
            assert np.array_equal(duals[-1],dual(data,'aux_before'))
        if args.finite_joints:
            assert not data['control__joint_f'][~used].any(),'Unrepresented rod/FREE generalized effort'
            for name,index in [('joint_target_q',rod_target_indices),('joint_target_qd',rod_velocity_indices)]:
                assert np.array_equal(data['control__'+name][index],source[name][index]),'Changed rod rest/control input'
            commands=[]
            for name,index in [('joint_target_q',spec.target_index.numpy()),('joint_target_qd',spec.dof_index.numpy()),('joint_f',spec.dof_index.numpy())]:
                value=np.zeros(index.shape,np.float64);value[valid]=data['control__'+name][index[valid]];commands.append(value)
            joint_inputs.append(commands)
        else:
            assert not data['control__joint_f'].any()
            for name in ('joint_target_q','joint_target_qd'): assert np.array_equal(data['control__'+name],source[name])
        for name,value in data.items():
            if not name.startswith('model_input__'):continue
            key=name.removeprefix('model_input__')
            if key in source: assert np.array_equal(value,source[key]), 'Unrepresented mutable input: '+key
        wrench=data['input__body_f'].astype(np.float64)
        forces.append(np.concatenate((wrench[:,:3],np.zeros((1,3)))))
        torques.append(np.concatenate((wrench[:,3:],np.zeros((1,3)))))
        candidates.append(int(data['contacts_before__rigid_contact_count'][0]))
        states.append(convert(*output));duals.append(dual(data,'aux_after'));previous=output;calls+=1
    assert calls == manifest['physical_vbd_calls'] == args.expected_frames*example.sim_substeps
    if args.finite_joints:
        # Accumulated controller metrics are not solver state and cannot be
        # recreated by assigning only the final pose. Require the original
        # full-run per-frame and final checks captured by the collector.
        assert manifest['authored_final_check']=='PASS'
        assert manifest['authored_post_step_checks']=={'requested':True,'calls':args.expected_frames}
    else:
        example.state_0.body_q.assign(previous[0]);example.state_0.body_qd.assign(previous[1]);example.sim_time=calls*dt;example.test_final()
    values={key:np.stack([state[key] for state in states]) for key in states[0]}
    if args.fresh_rod_inputs:
        from wp_vbd import math3d as so3
        values['v'][1:,fixed]=(values['x'][1:,fixed]-values['x'][:-1,fixed])/dt
        values['omega'][1:,fixed]=so3.log(so3.multiply(torch.as_tensor(values['q'][1:,fixed]),
                                                     so3.conjugate(torch.as_tensor(values['q'][:-1,fixed])))).numpy()/dt
        if args.bundle_release_events:
            mask=np.asarray(removal_masks);assert np.array_equal(mask.sum(0),np.ones(len(fixed),dtype=int))
            for step,body in zip(*np.where(mask)):
                values['v'][step+1,fixed[body]]=0.;values['omega'][step+1,fixed[body]]=0.
    with torch.no_grad():
        c=strain(scene,State(*(torch.as_tensor(values[key]) for key in ('x','q','v','omega'))))[...,3:]
        increment=torch.cat((torch.zeros_like(c[:1]),rod_strain_increment(c[1:],c[:-1])),0)
    # No-Dahl causal memory follows THIS model's commit semantics. Numeric
    # duals are imported from the source's already accepted past, never from
    # the future target. Do not reuse VBD's private previous-pose buffer.
    kappa=c.numpy().copy();kappa[0]=0.
    stress=np.zeros_like(kappa)
    if args.dahl_history:
        from wp_vbd.joint_history import DahlMemory,begin_dahl,commit_dahl
        memory=DahlMemory(torch.zeros_like(c[0]),torch.zeros_like(c[0]),torch.zeros_like(c[0]))
        enabled=torch.ones(len(ids),dtype=torch.bool)
        for step in range(calls):
            frozen,_,_=begin_dahl(memory,c[step],scene.rod.joint_stiffness[:,3:],scene.rod.rod_dahl_eps,
                                 scene.rod.rod_dahl_tau,enabled,torch.full_like(enabled,step==0))
            memory=commit_dahl(frozen,c[step+1],scene.rod.joint_stiffness[:,3:],scene.rod.rod_dahl_eps,scene.rod.rod_dahl_tau,enabled)
            kappa[step+1]=memory.kappa.numpy();increment[step+1]=memory.increment;stress[step+1]=memory.stress.numpy()
        assert np.isfinite(stress).all() and np.abs(stress).max()>0
    values.update(history__rod_dual=np.asarray(duals),history__rod_kappa=kappa,
                  history__rod_increment=increment.numpy(),history__rod_stress=stress,
                  history__rod_previous_x=values['x'].copy(),history__rod_previous_q=values['q'].copy(),
                  history__rod_initialized=np.arange(calls+1)>0,input_force=np.asarray(forces),input_torque=np.asarray(torques),
                  source_rigid_contact_candidate_count=np.asarray(candidates,np.int64))
    if args.matched_contact_history:
        assert len(contact_history)==calls+1
        values.update({'history__'+name:np.stack([row[name] for row in contact_history]) for name in contact_history[0]})
    if args.finite_joints:
        previous_angle=spec.coordinate_offset[None,:,3].clone();angles=[]
        with torch.no_grad():
            for index in range(calls+1):
                accepted=State(*(torch.as_tensor(values[key][index:index+1]) for key in ('x','q','v','omega')))
                angle=scene.joint_coordinates(accepted,previous_angle,jacobian=False).axis[...,3]
                previous_angle=torch.where(spec.angular_mode==1,angle,0.)
                angles.append(previous_angle[0].numpy().copy())
        values['history__joint_angle_prev']=np.asarray(angles)
        assert np.max(np.abs(np.diff(values['history__joint_angle_prev'],axis=0)))<np.pi-1e-6
        for index,name in enumerate(('input_joint_target','input_joint_target_rate','input_joint_effort')):
            values[name]=np.asarray([command[index] for command in joint_inputs])
    if args.fresh_rod_inputs:
        values.update(input_pin_x=np.asarray(pin_x),input_pin_q=np.asarray(pin_q))
        from wp_vbd.rod_contact_system import path_feasible
        accepted=State(*(torch.as_tensor(values[key][:-1]) for key in ('x','q','v','omega')))
        commanded_x=accepted.x.clone();commanded_q=accepted.q.clone()
        commanded_x[:,fixed]=torch.as_tensor(values['input_pin_x']);commanded_q[:,fixed]=torch.as_tensor(values['input_pin_q'])
        commanded=State(commanded_x,commanded_q,accepted.v,accepted.omega)
        angle=so3.log(so3.multiply(commanded.q,so3.conjugate(accepted.q))).norm(dim=-1)
        travel=(commanded.x-accepted.x).norm(dim=-1)+scene.body_radius*angle
        if args.bundle_release_events:
            # This declared non-material boundary is removed, not swept at 6000 m/s.
            travel[:,fixed]=torch.where(torch.as_tensor(mask),0.,travel[:,fixed])
            values.update(input_geometry_removal_mask=mask,input_controller_command_time=np.asarray(command_times))
        assert bool((travel[:,fixed]<=.05*scene.length).all())
        assert bool(path_feasible(scene,accepted,commanded).all()), 'Prescribed rod path leaves the admitted material domain'
    metadata={**scene.metadata,'teacher_frames':args.expected_frames,'teacher_substeps':calls,'teacher_iterations':configuration['iterations'],
              'reference_final_check':'PASS','newton_sha':manifest['source_sha'],'zero_Dahl_unprescribed_training_schema':not len(fixed) and not args.dahl_history,
              'zero_Dahl_stationary_training_schema':not args.fresh_rod_inputs and not args.dahl_history,'stationary_body_indices':[] if args.fresh_rod_inputs else fixed.tolist(),
              'zero_Dahl_prescribed_fresh_training_schema':args.fresh_rod_inputs and not args.dahl_history,
              'prescribed_rod_training_schema':args.fresh_rod_inputs,
              'dahl_prescribed_fresh_training_schema':args.fresh_rod_inputs and args.dahl_history,
              'prescribed_geometry_removal_training_schema':args.bundle_release_events,
              'prescribed_body_indices':fixed.tolist() if args.fresh_rod_inputs else [],
              'teacher_collision_policy_override':manifest.get('collision_policy_override'),
              'source_trace_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'history_provenance':'COM previous pose is last accepted native output; zero-Dahl kappa/increment derived causally from accepted current/past poses. Numeric duals are the source accepted past auxiliary state, not future labels.',
              'teacher_target':'Source material, asset, control and iteration budget; explicit collision-cadence override is recorded separately. No converged-teacher or private-pose-history equivalence claim.'}
    if args.fresh_rod_inputs:
        metadata['history_provenance']+=' Fixed pose commands are recorded inputs before each source solve; history retains the prior accepted poses including the initial pose before the first command. Prescribed velocities are reconstructed causally from accepted pose differences.'
    if args.matched_contact_history:
        metadata.update(matched_contact_training_schema='accepted_matcher_cache_v133',contact_history_storage_capacity=storage_capacity,
                        authored_contact_cadence_verified='one_fresh_query_per_physical_call',
                        contact_filter_provenance='Pinned original constructor and source args; the raw tape did not serialize the public filter-set wrapper.')
        metadata['history_provenance']+=' Contact cache and multiplier at input t are from accepted source call t-1. Query geometry cache is post-collision/pre-dynamics; only accepted physical calls commit it. All accepted rows are stored, unused capacity suffix removed; no matching labels from the future target.'
    if args.finite_joints:
        metadata.update(finite_rod_training_schema='accepted_joint_controls_winding_v144',
            source_target_q_start_alias_verified=True,
            source_accumulated_controller_checks=manifest['authored_post_step_checks'],
            source_original_final_check_provenance='Pinned full original collector ran all per-frame checks and final check; no claim to reconstitute accumulated controller metrics from a final pose.')
        metadata['history_provenance']+=' Finite controls are captured before each source call. Single-axis winding is reconstructed sequentially from accepted current/past COM poses in this model coordinate convention; no future target used as input history.'
    if args.dahl_history:
        metadata['history_provenance']=metadata['history_provenance'].replace('zero-Dahl kappa/increment derived causally from accepted current/past poses','Dahl kappa/increment/stress replayed with this model\'s begin/commit laws from accepted current/past poses, once per physical step')
    if args.bundle_release_events:
        metadata['geometry_removal_events']=[{'input_index':int(i),'bodies':fixed[mask[i]].tolist(),'command_time':command_times[i]} for i in np.flatnonzero(mask.any(-1))]
        metadata['history_provenance']+=' Explicit release events rebaseline only non-rod prescribed obstacle poses and zero their velocity; rod physical memory remains unchanged. No arbitrary teleport admission.'
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**{key:value.numpy() for key,value in scene.values.items()},
                        **{'rod__'+key:value.numpy() for key,value in vars(scene.rod).items()},
                        **({'joint__'+key:value.numpy() for key,value in vars(scene.joint_spec).items()} if args.finite_joints else {}),
                        **{'source__'+key:source[key] for key in (*SOURCE_FIELDS,*(FINITE_SOURCE_FIELDS if args.finite_joints else ())) if key in source},
                        **values,metadata=np.asarray(json.dumps(metadata)))
    report={'status':'ROD_CONTACT_TEACHER_DATA_QUALIFIED','demo':manifest['id'],'path':str(args.output),
            'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'teacher_frames':args.expected_frames,'teacher_substeps':calls,
            'body_count':n,'rod_count':len(ids),'original_source_final_check':'PASS',
            'native_state_and_past_dual_continuity_exact':not args.fresh_rod_inputs,'chunk_hashes_verified':True,
            'dynamic_state_and_past_dual_continuity_exact':True,'fixed_pose_commands_explicit':args.fresh_rod_inputs,
            'stationary_body_indices':metadata['stationary_body_indices'],'stationary_prescribed_pose_and_zero_velocity_exact':not args.fresh_rod_inputs,
            'prescribed_body_indices':metadata['prescribed_body_indices'],'teacher_collision_policy_override':metadata['teacher_collision_policy_override'],
            'source_contact_candidates_not_loads':sum(candidates),'causal_history_provenance':metadata['history_provenance'],
            'Dahl_history_active':args.dahl_history,'maximum_causal_Dahl_stress':float(np.abs(stress).max()),
            'geometry_removal_events':metadata.get('geometry_removal_events',[]),
            'matched_contact_training_schema':metadata.get('matched_contact_training_schema'),
            'contact_history_storage_capacity':metadata.get('contact_history_storage_capacity'),
            'finite_rod_training_schema':metadata.get('finite_rod_training_schema'),
            'finite_joint_count':len(scene.joint_spec.owners) if args.finite_joints else 0,
            'new_optimizer_updates':0,'new_training_admissions':0,'full_vbd_prediction_calls':0,
            'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'scope':'Complete authored reference qualification with explicit causal memory and boundary inputs. Any collision-cadence or geometry-removal policy is recorded. Actual queried losses, selected optimization and native model tasks are independent gates.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__ == '__main__':main()
