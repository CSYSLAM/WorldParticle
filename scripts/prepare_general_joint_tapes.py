"""Convert complete authored joint tapes into explicit COM/history/control inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.reference_tape import read_tape
    from wp_vbd.general_joint_scene import export_model
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--demo',choices=('basic.example_basic_joints','contacts.example_newton_cradle','basic.example_basic_conveyor_forces','basic.example_basic_urdf'),default='basic.example_basic_joints');args=parser.parse_args()
    identifier=args.demo;tape=ROOT/'.local/official_substeps_v1'/identifier
    cradle=identifier=='contacts.example_newton_cradle';conveyor=identifier=='basic.example_basic_conveyor_forces';urdf=identifier=='basic.example_basic_urdf'
    if urdf:tape=ROOT/'.local/packed_world_v32/reference'/identifier
    out=ROOT/('.local/packed_world_v32/urdf_authored.npz' if urdf else '.local/contact_observables_v29/conveyor_forces_authored.npz' if conveyor else '.local/general_joint_v28/newton_cradle_authored.npz' if cradle else '.local/general_joint_v26/basic_joints_authored.npz')
    reports=ROOT/('reports/packed_world_v32' if urdf else 'reports/contact_observables_v29' if conveyor else 'reports/joint_contact_v28' if cradle else 'reports/general_joint_rows_v24');reports.mkdir(parents=True,exist_ok=True);report=reports/'tape_admission.json'
    if out.exists() or report.exists():raise FileExistsError('Keep joint tape admission immutable')
    manifest_path=tape/'manifest.json';manifest=json.loads(manifest_path.read_text());assert manifest['status']=='PHYSICAL_SUBSTEP_TRACE_READY' and manifest['vbd_instances']==1 and manifest['source_sha']=='25298aea1691e060c86f95e8d2b773c898c006d3'
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu');torch.set_num_threads(1)
    example,solvers,_=make_example(identifier,reference=False,preserve_defaults=True);config=solvers[0][1];assert config==manifest['solvers'][0]['configuration']
    source_static=tape/manifest['static']['path'];assert hashlib.sha256(source_static.read_bytes()).hexdigest()==manifest['static']['sha256']
    checked=[];source_fields={};derived_rest_roundoff={}
    with np.load(source_static,allow_pickle=False) as original:
        fields=('body_mass','body_inv_mass','body_inertia','body_inv_inertia','body_com','body_q','shape_type','shape_transform','shape_scale','shape_margin','shape_collision_radius','shape_material_ke','shape_material_kd','shape_material_mu','joint_type','joint_parent','joint_child','joint_X_p','joint_X_c','joint_axis','joint_dof_dim','joint_q_start','joint_qd_start','joint_target_q_start','joint_q','joint_enabled','joint_target_ke','joint_target_kd','joint_limit_lower','joint_limit_upper','joint_limit_ke','joint_limit_kd')
        if urdf:fields+=('body_world','shape_world','gravity','body_flags','shape_body','shape_flags','joint_armature','joint_friction')
        for name in fields:
            current=getattr(example.model,name).numpy()
            if name=='joint_target_q_start':
                # This older tape schema omitted the derived target offset.
                # These admitted scalar/FREE assets have matching q/target
                # coordinate layouts; verify against the captured q offsets.
                recorded=original['solver0__model__joint_q_start']
                assert np.array_equal(current,recorded)
                source_fields['source__'+name]=recorded.copy()
                continue
            recorded=original['solver0__model__'+name];source_fields['source__'+name]=recorded.copy()
            if name=='body_q':
                position=float(np.abs(current[:,:3]-recorded[:,:3]).max());rotation=float(np.abs(current[:,3:]-recorded[:,3:]).max())
                assert max(position,rotation)<=1e-6,'FK differences exceed float32 rest-pose admission bound'
                derived_rest_roundoff={'maximum_position_m':position,'maximum_quaternion_component':rotation}
                example.model.body_q.assign(recorded)
            else:
                assert np.array_equal(current,recorded),name
                checked.append(name)
    scene=export_model(example.model,config,identifier,newton,example.sim_dt,'cpu',native_contact_export=conveyor,packed_worlds=urdf);spec=scene.joint_spec
    body_com=scene.values['body_com'].numpy();single=spec.angular_mode.numpy()==1;dof=spec.dof_index.numpy();safe=np.maximum(dof,0);valid=dof>=0;target=spec.target_index.numpy()
    def controls(value,index):
        result=np.zeros(index.shape,np.float64);mask=index>=0;result[mask]=value[index[mask]];return result
    def history(values,prefix):
        result=np.zeros(len(single),np.float64);result[single]=values[prefix+'__joint_angle_prev'][safe[single,3]];return result
    def state(pose,velocity):
        q=pose[:,3:].astype(np.float64)
        if urdf:q=q/np.linalg.norm(q,axis=-1,keepdims=True)
        rotated=body_com+2*np.cross(q[:,:3],np.cross(q[:,:3],body_com)+q[:,3:]*body_com)
        return {'x':pose[:,:3].astype(np.float64)+rotated,'q':q,'v':velocity[:,:3].astype(np.float64),'omega':velocity[:,3:].astype(np.float64)}
    trajectory=[];histories=[];force=[];targets=[];rates=[];efforts=[];previous=None;calls=0
    if urdf:
        jt=example.model.joint_type.numpy();starts=example.model.joint_qd_start.numpy()
        free=np.concatenate([np.arange(starts[j],starts[j+1]) for j in np.where(jt==int(newton.JointType.FREE))[0]])
    for index,(values,timing) in enumerate(read_tape(tape)):
        assert timing['sequence']==index and timing['solver_slot']==0 and timing['dt']==example.sim_dt
        assert set(timing['observables_requested'])==({'contact_f'} if conveyor else set())
        pose=values['input__body_q'];velocity=values['input__body_qd']
        if previous is None:trajectory.append(state(pose,velocity));histories.append(history(values,'aux_before'))
        else:assert np.array_equal(previous[0],pose) and np.array_equal(previous[1],velocity),'Out-of-solver body control needs an explicit kinematic channel'
        for key,value in values.items():
            if key.startswith('model_input__'):
                current=getattr(example.model,key.removeprefix('model_input__'),None)
                if hasattr(current,'numpy'):assert np.array_equal(value,current.numpy()),'Mutable model property needs a time-dependent training channel: '+key
        previous=(values['output__body_q'],values['output__body_qd']);trajectory.append(state(*previous));histories.append(history(values,'aux_after'))
        inputs=(values['input__body_f'],controls(values['control__joint_target_q'],target),controls(values['control__joint_target_qd'],dof),controls(values['control__joint_f'],dof))
        for destination,value in zip((force,targets,rates,efforts),inputs):
            if urdf and destination:assert np.array_equal(destination[0],value),'A claimed constant input changed'
            else:destination.append(value.copy())
        if urdf:assert not np.any(values['control__joint_f'][free]),'FREE joint efforts need their own explicit wrench channel'
        calls+=1
        if conveyor and (index+1)%example.sim_substeps==0:
            example.state_0.body_q.assign(previous[0]);example.state_0.body_qd.assign(previous[1]);example.test_post_step()
    assert calls==manifest['physical_vbd_calls']
    example.state_0.body_q.assign(previous[0]);example.state_0.body_qd.assign(previous[1]);example.sim_time=calls*example.sim_dt;example.test_final()
    arrays={key:np.stack([s[key] for s in trajectory]) for key in trajectory[0]}
    arrays.update(history__joint_angle_prev=np.asarray(histories),force=np.asarray(force),input_joint_target=np.asarray(targets),input_joint_target_rate=np.asarray(rates),input_joint_effort=np.asarray(efforts))
    assert all(np.isfinite(v).all() for v in arrays.values())
    metadata={**scene.metadata,'teacher_iterations':config['iterations'],'teacher_frames':manifest['frames_completed'],'teacher_substeps':calls,'native_preserve_defaults':True,'reference_final_check':'PASS','newton_sha':manifest['source_sha'],'source_trace_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'teacher_target':'Original finite-budget source with explicit angle history and controls. Physical joint law is the objective finite constitutive fixed point; no converged-trajectory equivalence claim.'}
    if urdf:metadata['constant_time_inputs']=['force','input_joint_target','input_joint_target_rate','input_joint_effort']
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,**{key:value.numpy() for key,value in scene.values.items()},**{'joint__'+key:value.numpy() for key,value in vars(spec).items() if torch.is_tensor(value)},**source_fields,**arrays,metadata=np.asarray(json.dumps(metadata)))
    result={'status':'JOINT_TEACHER_DATA_ADMITTED','demo':identifier,'path':str(out.relative_to(ROOT)),'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'teacher_iterations':config['iterations'],'teacher_substeps':calls,'original_final_check':'PASS','static_fields_bitexact':checked,'derived_target_q_start':'Matches captured joint_q_start exactly for this basic-joints asset, which has no FREE/ROD coordinate-size distinction. Older tape schema omitted joint_target_q_start.','derived_fk_rest_roundoff':derived_rest_roundoff,'recorded_physical_rest_preserved_exactly':True,'chunk_hashes_checked':True,'body_state_continuity_bitexact':True,'winding_history_and_controls_explicit':True,'full_vbd_constructor_calls':0,'full_vbd_step_calls':0,'new_shared_optimizer_updates':0,'scope':'Data eligibility only; actual optimizer gradients and selected shared weights remain separate.'}
    if conveyor:
        result['task_post_step_replayed_from_recorded_native_outputs']=True
        result['derived_target_q_start']='Matches captured q_start exactly, including all FREE seven-coordinate target blocks; current target array size also matches the source static model. No claim for general ROD targets.'
        result['controller_force_inputs_explicit']=True
        result['teacher_transport']={'minimum_maximum_travel_m':float(example.max_travel.min()),'mean_final_speed_m_per_s':float(np.linalg.norm(example.state_0.body_qd.numpy()[example.tracked_bodies,:3],axis=1).mean())}
    if urdf:
        result['derived_target_q_start']='Matches captured q_start exactly, including all FREE target blocks; source and native target array layouts match.'
        result.update(world_count=example.world_count,body_count=example.model.body_count,constant_inputs_verified_across_all_steps=metadata['constant_time_inputs'],source_free_joint_efforts_zero=True,source_armature_policy='Pinned VBD explicitly does not support joint armature; shared model also does not introduce that ignored source property into dynamics.')
    report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
