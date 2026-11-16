"""Admit stateless rigid teacher tapes after asset, timing and task checks."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--demos',nargs='+',default=['contacts.example_balance_bird','contacts.example_domino_spiral']);args=p.parse_args()
    import numpy as np
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    from wp_vbd.reference_tape import read_tape
    from wp_vbd.rigid_contact_scene import export_model
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cuda:0')
    out=ROOT/'.local/rigid_contact_v21';reports=ROOT/'reports/rigid_contact_integration_v21';out.mkdir(parents=True,exist_ok=True)
    for identifier in args.demos:
        if identifier not in ('contacts.example_balance_bird','contacts.example_domino_spiral','basic.example_basic_shapes'):raise ValueError('Additional controllers/observables require their own stateless admission')
        prefix=identifier.split('.')[-1].removeprefix('example_');target=out/(prefix+'_authored.npz');report_path=reports/(prefix+'_tape_admission.json')
        if target.exists() or report_path.exists():raise FileExistsError('Keep tape admission evidence immutable')
        tape=ROOT/'.local/official_substeps_v1'/identifier;manifest_path=tape/'manifest.json';manifest=json.loads(manifest_path.read_text())
        assert manifest['status']=='PHYSICAL_SUBSTEP_TRACE_READY' and manifest['vbd_instances']==1
        assert manifest['source_sha']=='25298aea1691e060c86f95e8d2b773c898c006d3'
        example,solvers,authored=make_example(identifier,reference=False,preserve_defaults=True)
        config=solvers[0][1];assert config==manifest['solvers'][0]['configuration']
        static,meta=export_model(example.model,config,identifier,newton,example.sim_dt)
        source_static_path=tape/manifest['static']['path'];assert hashlib.sha256(source_static_path.read_bytes()).hexdigest()==manifest['static']['sha256']
        checked=[]
        with np.load(source_static_path) as recorded:
            for name in ('body_mass','body_inv_mass','body_inertia','body_inv_inertia','body_com','body_q','shape_type','shape_transform','shape_scale','shape_margin','shape_collision_radius','shape_material_ke','shape_material_kd','shape_material_mu'):
                current=getattr(example.model,name).numpy();previous=recorded['solver0__model__'+name]
                assert np.array_equal(current,previous),(identifier,name)
                checked.append(name)
        trajectory=[];forces=[];counts=[];previous_output=None;last=None
        def state(pose,velocity):
            pose=pose.astype(np.float64);q=pose[:,3:];com=static['body_com']
            rotated=com+2*np.cross(q[:,:3],np.cross(q[:,:3],com)+q[:,3:]*com)
            return {'x':pose[:,:3]+rotated,'q':q,'v':velocity[:,:3].astype(np.float64),'omega':velocity[:,3:].astype(np.float64)}
        for index,(values,timing) in enumerate(read_tape(tape)):
            assert timing['sequence']==index and timing['solver_slot']==0 and timing['dt']==example.sim_dt
            pose=values['input__body_q'];velocity=values['input__body_qd']
            if previous_output is None:trajectory.append(state(pose,velocity))
            else:
                assert np.array_equal(pose,previous_output[0]) and np.array_equal(velocity,previous_output[1]),'Authored controller changes state outside this stateless force-only contract'
            joint_force=values.get('control__joint_f')
            if joint_force is not None:assert not np.any(joint_force),'Free-joint control forces need explicit integration ownership'
            for key,value in values.items():
                if key.startswith('model_input__'):
                    current=getattr(example.model,key.removeprefix('model_input__'),None)
                    if hasattr(current,'numpy'):assert np.array_equal(value,current.numpy()),'Mutable model parameter requires an explicit input channel: '+key
            count=int(values['contacts_before__rigid_contact_count'][0]);capacity=len(values['contacts_before__rigid_contact_shape0']);assert 0<=count<=capacity
            previous_output=(values['output__body_q'],values['output__body_qd']);last=previous_output
            trajectory.append(state(*previous_output));forces.append(values['input__body_f']);counts.append(count)
        assert len(forces)==manifest['physical_vbd_calls']
        example.state_0.body_q.assign(last[0]);example.state_0.body_qd.assign(last[1]);example.sim_time=len(forces)*example.sim_dt
        example.test_final()
        arrays={key:np.stack([value[key] for value in trajectory]) for key in trajectory[0]}
        assert all(np.isfinite(v).all() for v in arrays.values())
        meta.update(teacher_iterations=config['iterations'],teacher_frames=manifest['frames_completed'],teacher_substeps=len(forces),reference_final_check='PASS',native_preserve_defaults=True,newton_sha=manifest['source_sha'],teacher_target='Original authored finite iteration budget; comparable task behavior, no convergence or exact-pose claim.',source_trace_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest())
        np.savez_compressed(target,**static,**arrays,force=np.asarray(forces),source_contact_count=np.asarray(counts),metadata=json.dumps(meta))
        result={'status':'STATELESS_TEACHER_DATA_ADMITTED','demo':identifier,'teacher_path':str(target.relative_to(ROOT)),'data_sha256':hashlib.sha256(target.read_bytes()).hexdigest(),'physical_substeps':len(forces),'original_terminal_assertion':'PASS','static_fields_bitexact':checked,'chunk_hashes_checked':True,'substep_state_continuity_bitexact':True,'full_vbd_constructor_calls':0,'full_vbd_step_calls':0,'actual_shared_optimizer_updates':0,'source_trace_manifest_sha256':meta['source_trace_manifest_sha256'],'scope':'Only training-data eligibility for the implemented rigid-only stateless contact contract. Actual shared optimizer sampling and complete learned inference tasks are separate gates.'}
        report_path.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
