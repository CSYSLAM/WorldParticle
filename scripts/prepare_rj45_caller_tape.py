"""Qualify authored RJ45 records with causal caller and raw physical history.

CPU construction supplies descriptors only: SDF builds are suppressed and
all collision/integration is forbidden. Actual training needs real GPU SDFs.
"""
import argparse
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);parser.add_argument('--expected-frames',type=int,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.report.exists():raise FileExistsError('Preserve earlier qualified data')
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd import math3d as so3
    from wp_vbd.data import State
    from wp_vbd.official import make_example
    from wp_vbd.reference_tape import arrays,read_tape
    from wp_vbd.rod_joint_scene import export_model
    from wp_vbd.rod_contact_data import SOURCE_FIELDS,FINITE_SOURCE_FIELDS
    from wp_vbd.rod_contact_system import strain,valid_state
    from wp_vbd.joint_history import rod_strain_increment
    from wp_vbd.rigid_mesh_reference import mesh_identity,restore_mass_inputs,filter_pairs
    from wp_vbd.caller_program_io import from_example,to_values,from_values
    from wp_vbd.rigid_caller_program import SCHEMA,CallerMemory
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    manifest_path=args.trace/'manifest.json';manifest=json.loads(manifest_path.read_text())
    assert manifest['status']=='PHYSICAL_SUBSTEP_TRACE_READY' and manifest['vbd_instances']==1
    assert manifest['id']=='contacts.example_contacts_rj45_plug'
    assert manifest['frames_completed']==manifest['frames_requested']==args.expected_frames>0
    assert subprocess.check_output(['git','rev-parse','HEAD'],cwd='/home/oem/code/engine/newton',text=True).strip()==manifest['source_sha']=='25298aea1691e060c86f95e8d2b773c898c006d3'
    static=args.trace/manifest['static']['path'];assert hashlib.sha256(static.read_bytes()).hexdigest()==manifest['static']['sha256']
    with np.load(static,allow_pickle=False) as archive:
        recorded={k.removeprefix('solver0__model__'):archive[k].copy() for k in archive.files if k.startswith('solver0__model__')}
    calls={'vbd_construct':0,'vbd_step':0,'collision':0,'suppressed_sdf_build':0};sdf_requests=[]
    original=newton.solvers.SolverVBD.__init__
    @functools.wraps(original)
    def forbid_init(*unused,**kwargs):calls['vbd_construct']+=1;raise AssertionError('Only reference records, no VBD construction')
    def forbid_step(*unused,**kwargs):calls['vbd_step']+=1;raise AssertionError('No VBD stepping in qualification')
    def forbid_collision(*unused,**kwargs):calls['collision']+=1;raise AssertionError('Descriptor-only model must not collide')
    def skip_sdf(mesh,*values,**options):
        calls['suppressed_sdf_build']+=1;sdf_requests.append({'arguments':list(values),'options':options})
    newton.solvers.SolverVBD.__init__=forbid_init;newton.solvers.SolverVBD.step=forbid_step
    newton.CollisionPipeline.collide=forbid_collision;newton.Mesh.build_sdf=skip_sdf
    example,solvers,_=make_example(manifest['id'],reference=False,preserve_defaults=True)
    assert len(solvers)==1 and solvers[0][1]==manifest['solvers'][0]['configuration']
    model=example.model;n=model.body_count;dt=example.sim_dt;configuration=solvers[0][1]
    geometry=mesh_identity(model,newton,require_sdf=False);roundoff=restore_mass_inputs(model,recorded)
    source=arrays(model)
    for name,value in recorded.items():
        if name=='bvh_shape_enabled':
            count=model.bvh_shape_count_enabled;np.testing.assert_array_equal(np.sort(value[:count]),np.sort(source[name][:count]))
        else:assert np.array_equal(source[name],value),name
    source['joint_target_q_start']=model.joint_target_q_start.numpy().copy();source['shape_collision_filter_pairs']=filter_pairs(model)
    scene=export_model(model,configuration,manifest['id'],newton,dt,'cpu',sdf_mesh_contacts=True,massless_prescribed=True)
    assert not bool(scene.rod.rod_dahl_eps.any()),'Active Dahl needs its extended state replay'
    program=from_example(scene,example);program_values=to_values(program)
    reconstructed=to_values(from_values(scene,program_values))
    assert all(np.array_equal(v,reconstructed[k]) for k,v in program_values.items())
    def convert(pose,velocity):
        p=torch.as_tensor(pose.astype(np.float64));q=so3.normalize(p[:,3:]);x=p[:,:3]+so3.rotate(q,scene.body_com[:-1])
        v=torch.as_tensor(velocity.astype(np.float64))
        return State(torch.cat((x,x.new_zeros(1,3)))[None],torch.cat((q,q.new_tensor([[0.,0.,0.,1.]])))[None],
            torch.cat((v[:,:3],v.new_zeros(1,3)))[None],torch.cat((v[:,3:],v.new_zeros(1,3)))[None])
    first,_=next(read_tape(args.trace));current=convert(source['body_q'],first['input__body_qd'])
    memory=CallerMemory(torch.as_tensor(np.asarray(example._rest_pos).astype(np.float64))[None],torch.tensor([0]),torch.tensor([False]))
    states=[current];memories=[memory];duals=[];controls=[];candidate_counts=[];previous_pose=source['body_q'];previous_dual=None;generation=None
    spec=scene.joint_spec;valid=spec.dof_index.numpy()>=0;used=np.zeros(model.joint_dof_count,bool);used[spec.dof_index.numpy()[valid]]=True
    rod_ids=scene.rod.source_joint_ids.numpy();rod_q=source['joint_target_q_start'][rod_ids,None]+np.arange(7)
    rod_qd=source['joint_qd_start'][rod_ids,None]+np.arange(6)
    def dual(data,prefix):return np.concatenate((data[prefix+'__joint_lambda_lin'][rod_ids],data[prefix+'__joint_lambda_ang'][rod_ids]),-1).astype(np.float64)
    max_x=max_q=max_force=0.;count=0
    for data,timing in read_tape(args.trace):
        assert timing['sequence']==count and timing['solver_slot']==0 and timing['frame']==count//program.steps_per_sample and timing['dt']==dt
        assert bool(data['aux_before___update_rigid_history'])
        g=int(data['contacts_before__contact_generation'][0]);assert generation is None or g==generation+1;generation=g
        assert np.array_equal(data['aux_before__body_q_prev'],previous_pose),'Physical history was externally edited'
        before=dual(data,'aux_before')
        if previous_dual is None:duals.append(before)
        else:assert np.array_equal(before,previous_dual)
        prepared,force,torque=program.prepare(current,memory,scene.gravity,example.pick_stiffness,example.pick_damping,torch.tensor([-1]))
        incoming=data['input__body_q'];expected_q=so3.normalize(torch.tensor(incoming[:,3:],dtype=torch.float64))[None]
        origin=prepared.x-so3.rotate(prepared.q,scene.body_com)
        max_x=max(max_x,float(np.abs(origin[0,:n].numpy()-incoming[:,:3]).max()))
        max_q=max(max_q,float(so3.log(so3.multiply(prepared.q[:,:n],so3.conjugate(expected_q))).norm(dim=-1).max()))
        np.testing.assert_array_equal(torch.cat((prepared.v[:,:n],prepared.omega[:,:n]),-1)[0].numpy(),data['input__body_qd'])
        max_force=max(max_force,float(np.abs(torch.cat((force[:,:n],torque[:,:n]),-1)[0].numpy()-data['input__body_f']).max()))
        assert bool(valid_state(scene,prepared).all())
        assert not data['control__joint_f'][~used].any()
        for name,indices in [('joint_target_q',rod_q),('joint_target_qd',rod_qd)]:
            assert np.array_equal(data['control__'+name][indices],source[name][indices]),'Unrepresented rod target change'
        command=[]
        for name,index in [('joint_target_q',spec.target_index.numpy()),('joint_target_qd',spec.dof_index.numpy()),('joint_f',spec.dof_index.numpy())]:
            value=np.zeros(index.shape,np.float64);value[valid]=data['control__'+name][index[valid]];command.append(value)
        controls.append(command)
        for name,value in data.items():
            if name.startswith('model_input__') and name.removeprefix('model_input__') in source:
                assert np.array_equal(value,source[name.removeprefix('model_input__')]),'Unrepresented mutable material or asset: '+name
        current=convert(data['output__body_q'],data['output__body_qd'])
        assert bool(valid_state(scene,current).all())
        memory=program.commit(current,memory,torch.tensor([True]));states.append(current);memories.append(memory)
        previous_pose=data['output__body_q'];previous_dual=dual(data,'aux_after');duals.append(previous_dual)
        candidate_counts.append(int(data['contacts_before__rigid_contact_count'][0]));count+=1
    assert count==manifest['physical_vbd_calls']==args.expected_frames*program.steps_per_sample
    assert max_x<1e-7 and max_q<5e-5 and max_force<1e-4
    # The original final check is stateless: it reads only final finite poses
    # and initial transforms. Apply the authored post-solve alignment first.
    aligned=program.pose.apply(current,align_previous_output=True,apply_followers=False)
    final_pose=torch.cat((aligned.x[:,:n]-so3.rotate(aligned.q[:,:n],scene.body_com[:n]),aligned.q[:,:n]),-1)[0].numpy().astype(np.float32)
    example.state_0.body_q.assign(final_pose);example.state_0.body_qd.assign(data['output__body_qd']);example.test_final()
    values={key:torch.cat([getattr(s,key) for s in states]).numpy() for key in ('x','q','v','omega')}
    with torch.no_grad():
        c=strain(scene,State(*(torch.as_tensor(values[k]) for k in ('x','q','v','omega'))))[...,3:]
        increment=torch.cat((torch.zeros_like(c[:1]),rod_strain_increment(c[1:],c[:-1])),0)
    kappa=c.numpy().copy();kappa[0]=0.
    values.update(history__rod_dual=np.asarray(duals),history__rod_kappa=kappa,history__rod_increment=increment.numpy(),
        history__rod_stress=np.zeros_like(kappa),history__rod_previous_x=values['x'].copy(),history__rod_previous_q=values['q'].copy(),
        history__rod_initialized=np.arange(count+1)>0,
        history__caller_target=torch.cat([m.target for m in memories]).numpy(),
        history__caller_phase=torch.cat([m.phase for m in memories]).numpy(),history__caller_align=torch.cat([m.align for m in memories]).numpy(),
        input_caller_stiffness=np.full(count,example.pick_stiffness,np.float64),input_caller_damping=np.full(count,example.pick_damping,np.float64),
        input_caller_picked_body=np.full(count,-1,np.int64))
    previous_angle=spec.coordinate_offset[None,:,3].clone();angles=[]
    with torch.no_grad():
        for state in states:
            angle=scene.joint_coordinates(state,previous_angle,jacobian=False).axis[...,3]
            previous_angle=torch.where(spec.angular_mode==1,angle,0.);angles.append(previous_angle[0].numpy().copy())
    values['history__joint_angle_prev']=np.asarray(angles)
    assert np.max(np.abs(np.diff(values['history__joint_angle_prev'],axis=0)))<np.pi-1e-6
    for i,name in enumerate(('input_joint_target','input_joint_target_rate','input_joint_effort')):values[name]=np.asarray([c[i] for c in controls])
    assert all(np.isfinite(v).all() for v in values.values())
    assert calls=={'vbd_construct':0,'vbd_step':0,'collision':0,'suppressed_sdf_build':3}
    metadata={**scene.metadata,'teacher_frames':args.expected_frames,'teacher_substeps':count,'teacher_iterations':configuration['iterations'],
        'reference_final_check':'PASS','newton_sha':manifest['source_sha'],'finite_rod_training_schema':'accepted_joint_controls_winding_v144',
        'prescribed_rod_training_schema':True,'zero_Dahl_prescribed_fresh_training_schema':True,'stationary_body_indices':[],
        'prescribed_body_indices':scene.prescribed_ids.tolist(),'rigid_caller_training_schema':SCHEMA,
        'rigid_mesh_reference_schema':'authored_sdf_mesh_identity_and_reference_mass_v216','mesh_geometry_identity':geometry,
        'source_constructor_sdf_requests':sdf_requests,'source_filter_provenance':'Pinned authored constructor, every numeric static field verified; legacy tape did not record filter-set wrapper.',
        'reference_mass_inputs_policy':'per_body_reduction_bound_1e-5_then_explicit_reference_inputs',
        'authored_contact_cadence_verified':'one_fresh_query_per_physical_call',
        'source_trace_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        'source_original_final_check_provenance':'Original stateless test_final evaluated on recorded final pose after authored post-solve alignment; no integration or accumulated metric reconstruction.',
        'history_provenance':'Raw accepted physical poses/velocities and prior numeric duals; zero-Dahl and winding reconstructed sequentially. Caller target uses only past accepted frame outputs. Model rollouts must substitute their predictions.',
        'teacher_target':'Original authored asset/material/controller and12-iteration budget at original dt,120 frames/2s; no converged-teacher or general mesh-force coverage claim.'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**{k:v.numpy() for k,v in scene.values.items()},
        **{'rod__'+k:v.numpy() for k,v in vars(scene.rod).items()},**{'joint__'+k:v.numpy() for k,v in vars(spec).items()},
        **{'source__'+k:source[k] for k in (*SOURCE_FIELDS,*FINITE_SOURCE_FIELDS) if k in source},
        **program_values,**values,metadata=np.asarray(json.dumps(metadata)))
    result={'status':'RJ45_AUTHORED_CAUSAL_TEACHER_QUALIFIED_FOR_MESH_PREFLIGHT','demo':manifest['id'],'path':str(args.output),
        'sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'teacher_frames':args.expected_frames,'teacher_substeps':count,
        'physical_seconds':count*dt,'body_count':n,'rod_count':len(rod_ids),'finite_joint_count':len(spec.owners),
        'source_original_final_check':'PASS','source_mesh_candidates_not_load_proof':sum(candidate_counts),
        'maximum_caller_origin_error_m':max_x,'maximum_caller_rotation_error_rad':max_q,'maximum_feedback_wrench_error':max_force,
        'mesh_mass_roundoff_restoration':roundoff,'descriptor_operations':calls,'caller_serialization_roundtrip_exact':True,
        'no_recorded_force_or_pin_channels_as_inputs':True,'new_optimizer_updates':0,'new_training_admissions':0,'full_vbd_prediction_calls':0,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'implementation_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/wp_vbd').glob('*.py')},
        'scope':'Complete existing authored120-frame/720-step records and original stateless final check. Descriptor-only CPU model never collides or integrates. Real GPU SDF loader, full shared backward, expanded development, optimization and native task quality remain independent gates.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='implementation_sha256'},indent=2))


if __name__=='__main__':main()
