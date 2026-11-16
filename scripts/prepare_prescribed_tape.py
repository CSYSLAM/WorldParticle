"""Admit the authored kinematic conveyor with complete boundary-pose history."""
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']
import numpy as np
import torch
import warp as wp
import newton
import newton.viewer
from wp_vbd.official import make_example
from wp_vbd.reference_tape import read_tape
from wp_vbd.general_joint_scene import export_model
from wp_vbd.asset_roundoff import restore_free_quaternions
from wp_vbd import math3d as so3


def main():
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu');torch.set_num_threads(1)
    identifier='basic.example_basic_conveyor';tape=ROOT/'.local/official_substeps_v1'/identifier
    directory=ROOT/'reports/prescribed_motion_v30';directory.mkdir(parents=True,exist_ok=True);target=ROOT/'.local/prescribed_motion_v30/conveyor_authored.npz';report_path=directory/'tape_admission.json'
    if target.exists() or report_path.exists():raise FileExistsError('Keep prescribed tape admission immutable')
    manifest_path=tape/'manifest.json';manifest=json.loads(manifest_path.read_text());assert manifest['status']=='PHYSICAL_SUBSTEP_TRACE_READY' and manifest['vbd_instances']==1 and manifest['source_sha']=='25298aea1691e060c86f95e8d2b773c898c006d3'
    example,solvers,arguments=make_example(identifier,reference=False,preserve_defaults=True,host_scalar_compat=True);configuration=solvers[0][1];assert configuration==manifest['solvers'][0]['configuration']
    def forbidden(*unused,**kwargs):raise AssertionError('Full VBD forbidden in data conversion')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden
    source_static=tape/manifest['static']['path'];assert hashlib.sha256(source_static.read_bytes()).hexdigest()==manifest['static']['sha256']
    source_fields={};exact=[];roundoff={}
    with np.load(source_static,allow_pickle=False) as source:
        for name in ('body_mass','body_inv_mass','body_inertia','body_inv_inertia','body_com','body_flags','body_q','shape_type','shape_body','shape_transform','shape_scale','shape_margin','shape_collision_radius','shape_flags','shape_material_ke','shape_material_kd','shape_material_mu','joint_type','joint_parent','joint_child','joint_X_p','joint_X_c','joint_axis','joint_dof_dim','joint_q_start','joint_qd_start','joint_q','joint_enabled','joint_target_ke','joint_target_kd','joint_limit_lower','joint_limit_upper','joint_limit_ke','joint_limit_kd'):
            value=source['solver0__model__'+name];current=getattr(example.model,name).numpy();source_fields['source__'+name]=value.copy()
            if name=='body_q':
                maximum=float(np.abs(current-value).max());assert maximum<=1e-6;roundoff[name]=maximum;example.model.body_q.assign(value)
            elif name=='joint_q':roundoff[name]=restore_free_quaternions(example.model,value,newton)
            else:assert np.array_equal(current,value),name;exact.append(name)
        starts=example.model.joint_target_q_start.numpy();assert np.array_equal(starts,source['solver0__model__joint_q_start'])
        assert example.model.joint_target_q.shape==source['solver0__model__joint_target_q'].shape
        source_fields['source__joint_target_q_start']=starts.copy()
    scene=export_model(example.model,configuration,identifier,newton,example.sim_dt,'cpu',prescribed_motion=True);spec=scene.joint_spec
    kin=scene.prescribed_ids.numpy();dynamic=np.where(scene.movable[:scene.world].numpy())[0];single=spec.angular_mode.numpy()==1;qd=spec.dof_index.numpy();qi=spec.target_index.numpy()
    def coordinates(pose,velocity):
        p=torch.as_tensor(pose).double();q=so3.normalize(p[:,3:]);v=velocity.astype(np.float64)
        return {'x':(p[:,:3]+so3.rotate(q,scene.body_com[:-1])).numpy(),'q':q.numpy(),'v':v[:,:3],'omega':v[:,3:]}
    def winding(values,prefix):
        result=np.zeros(len(spec.owners),np.float64);result[single]=values[prefix+'__joint_angle_prev'][qd[single,3]];return result
    def controls(value,indices):
        result=np.zeros(indices.shape,np.float64);valid=indices>=0;result[valid]=value[indices[valid]];return result
    trajectory=[];angle=[];old_x=[];old_q=[];initialized=[];force=[];targets=[];rates=[];efforts=[];inputs={k:[] for k in ('x','q','v','omega')};previous=None
    resets=0
    for index,(values,timing) in enumerate(read_tape(tape)):
        assert timing['sequence']==index and timing['solver_slot']==0 and timing['dt']==example.sim_dt and not timing['observables_requested']
        native_pose=values['input__body_q'];native_velocity=values['input__body_qd'];current=coordinates(native_pose,native_velocity)
        if previous is None:
            trajectory.append(current);angle.append(winding(values,'aux_before'))
            mask=values['aux_before___rigid_pose_rebaseline_mask'];assert bool(mask[0]);resets+=1
            old_x.append(current['x'][kin]);old_q.append(current['q'][kin]);initialized.append(False)
        else:
            assert np.array_equal(previous[0][dynamic],native_pose[dynamic]) and np.array_equal(previous[1][dynamic],native_velocity[dynamic]),'Only prescribed bodies may be changed outside the physics step'
            assert np.array_equal(values['aux_before__body_q_prev'][kin],previous[0][kin]),'Committed source prescribed pose history changed unexpectedly'
        for key,value in current.items():inputs[key].append(value[kin])
        for key,value in values.items():
            if key.startswith('model_input__'):
                native=getattr(example.model,key.removeprefix('model_input__'),None)
                if hasattr(native,'numpy'):assert np.array_equal(native.numpy(),value),'Mutable asset input requires a separate channel: '+key
        previous=(values['output__body_q'],values['output__body_qd']);output=coordinates(*previous)
        assert np.array_equal(previous[0][kin],native_pose[kin]),'Prescribed pose was changed by the source dynamics solver'
        trajectory.append(output);angle.append(winding(values,'aux_after'));old_x.append(output['x'][kin]);old_q.append(output['q'][kin]);initialized.append(True)
        assert np.array_equal(values['aux_after__body_q_prev'][kin],previous[0][kin])
        force.append(values['input__body_f']);targets.append(controls(values['control__joint_target_q'],qi));rates.append(controls(values['control__joint_target_qd'],qd));efforts.append(controls(values['control__joint_f'],qd))
    assert len(force)==manifest['physical_vbd_calls']
    example.state_0.body_q.assign(previous[0]);example.state_0.body_qd.assign(previous[1]);example.sim_time=len(force)*example.sim_dt;example.test_final()
    arrays={key:np.stack([s[key] for s in trajectory]) for key in trajectory[0]}
    arrays.update(history__joint_angle_prev=np.asarray(angle),history__prescribed_x_prev=np.asarray(old_x),history__prescribed_q_prev=np.asarray(old_q),history__prescribed_initialized=np.asarray(initialized,dtype=bool),force=np.asarray(force),input_joint_target=np.asarray(targets),input_joint_target_rate=np.asarray(rates),input_joint_effort=np.asarray(efforts),**{'input_prescribed_'+key:np.asarray(value) for key,value in inputs.items()})
    assert all(np.isfinite(value).all() for value in arrays.values())
    metadata={**scene.metadata,'teacher_iterations':configuration['iterations'],'teacher_frames':manifest['frames_completed'],'teacher_substeps':len(force),'native_preserve_defaults':True,'native_host_scalar_compat':True,'reference_final_check':'PASS','newton_sha':manifest['source_sha'],'source_trace_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),'teacher_target':'Authored finite-budget reference with explicit imposed COM poses/rates and committed pose history; only dynamic DOFs are prediction targets.'}
    target.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(target,**{k:v.numpy() for k,v in scene.values.items()},**{'joint__'+k:v.numpy() for k,v in vars(spec).items() if torch.is_tensor(v)},**source_fields,**arrays,metadata=json.dumps(metadata))
    result={'status':'PRESCRIBED_TEACHER_DATA_ADMITTED','demo':identifier,'path':str(target.relative_to(ROOT)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'teacher_substeps':len(force),'teacher_iterations':configuration['iterations'],'source_final_check':'PASS','prescribed_body_ids':kin.tolist(),'dynamic_state_continuity_bitexact':True,'source_prescribed_history_matches_committed_output':True,'source_first_step_world_rebaseline_masks':resets,'asset_fields_bitexact':exact,'bounded_host_free_quaternion_roundoff':roundoff,'canonical_recorded_rest_preserved':True,'prescribed_inputs_and_history_explicit':True,'chunk_hashes_checked':True,'full_vbd_constructor_calls':0,'full_vbd_step_calls':0,'new_shared_optimizer_updates':0,'scope':'Data admission only; actual selected NN gradients, native behavior, full controller restart and realtime remain independent.'}
    report_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
