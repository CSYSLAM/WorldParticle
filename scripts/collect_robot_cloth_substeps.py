"""Collect full, independently simulated VBD drapes for coarse-step supervision."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--limit',type=int,default=12)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd.franka_cloth_corner_scene import build,controller
    from wp_vbd.robot_cloth_data import SCHEMA,digest,tensors
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.mixed_contact_data import STATIC_FIELDS
    from wp_vbd.surface_joint_data import JOINT_FIELDS
    from wp_vbd.cloth_drape_metrics import drape_metrics
    from wp_vbd.reference_mimic_relaxation import ReferenceMimicRelaxation
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache')
    wp.init();wp.set_device(a.device)
    cases=[('edge',300.,.0015,2.,'training'),('corner',300.,.0015,2.,'training'),
           ('edge',220.,.001,2.5,'training'),('corner',420.,.0025,2.,'training'),
           ('edge',420.,.0025,2.5,'training'),('corner',220.,.001,2.5,'training'),
           ('edge',340.,.0018,2.25,'development'),('corner',260.,.0013,2.25,'development'),
           ('edge',260.,.002,2.25,'training'),('edge',380.,.0013,2.,'training'),
           ('corner',300.,.003,2.25,'training'),('corner',340.,.0035,2.25,'development')]
    manifest={'schema':SCHEMA,'newton_sha':newton_revision(),'status':'RUNNING','scenes':[],
              'teacher_dt':1/720,'target_student_dt':1/180,'rejected':[],
              'scope':'Native offline teacher only; original and local corner folds, full robot dynamics.'}
    if (a.output/'manifest.json').exists():
        manifest=json.loads((a.output/'manifest.json').read_text())
        if manifest['newton_sha']!=newton_revision():raise ValueError('Cannot resume another Newton revision')
        manifest.setdefault('rejected',[])
        manifest['status']='RUNNING'
    def write():
        target=a.output/'manifest.json';tmp=target.with_suffix('.tmp')
        tmp.write_text(json.dumps(manifest,indent=2)+'\n');tmp.replace(target)
    write()
    for index,(fold,membrane,bending,friction,split) in enumerate(cases[:a.limit]):
        name=f'robot_cloth_{index:02d}_{fold}';path=a.output/(name+'.npz')
        existing=next((r for r in manifest['scenes'] if r['dataset_id']==name),None)
        if existing:
            if not path.exists() or digest(path)!=existing['sha256']:raise ValueError('Existing teacher changed')
            continue
        if any(r['dataset_id']==name for r in manifest['rejected']):continue
        if path.exists():raise FileExistsError(path)
        options=dict(substeps=12,grasp_style='overhead',fold_profile=fold,
                     membrane=membrane,bending=bending,friction=friction)
        e=build(wp,newton,device=a.device,**options)
        source={}
        for field in set(STATIC_FIELDS+JOINT_FIELDS+('joint_mimic_coeffs','joint_target_mode')):
            array=getattr(e.model,field,None)
            if array is not None:source['source__'+field]=array.numpy().copy()
        for shape,item in enumerate(e.model.shape_source):
            for field in ('vertices','indices','points'):
                array=getattr(item,field,None)
                if array is None:continue
                array=array.numpy() if hasattr(array,'numpy') else np.asarray(array)
                if not array.dtype.hasobject:source[f'source_geometry{shape}__{field}']=array.copy()
        cfg=dict(iterations=32,integrate_with_external_rigid_solver=False,
            particle_enable_self_contact=False,rigid_joint_linear_ke=1e5,rigid_joint_angular_ke=1e5,
            rigid_joint_linear_kd=0.,rigid_joint_angular_kd=0.,
            rigid_body_contact_buffer_size=512,rigid_body_particle_contact_buffer_size=1024)
        bound=inspect.signature(newton.solvers.SolverVBD).bind(e.model,**cfg);bound.apply_defaults()
        configuration={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
        scene=export_model(e.model,configuration,'research.franka_cloth_corner',newton,1/720,a.device)
        source_q=e.model.joint_q.numpy().copy()
        archive={k:[] for k in ('particle_position','particle_velocity','body_pose','body_velocity',
                               'joint_position','input_joint_target_raw','input_joint_rate_raw','input_joint_effort_raw')}
        q_read=wp.empty_like(e.model.joint_q);qd_read=wp.empty_like(e.model.joint_qd)
        def record():
            newton.eval_ik(e.model,e.state_0,q_read,qd_read)
            for key,arr in [('particle_position',e.state_0.particle_q),('particle_velocity',e.state_0.particle_qd),
                            ('body_pose',e.state_0.body_q),('body_velocity',e.state_0.body_qd),('joint_position',q_read)]:
                archive[key].append(arr.numpy().copy())
        start=time.perf_counter();record();adapter=ReferenceMimicRelaxation(1.,projection_interval=32)
        with adapter:
            solver=newton.solvers.SolverVBD(e.model,**cfg)
            pipeline=newton.CollisionPipeline(e.model,deterministic=True,rigid_contact_max=8192,
                                             enable_rigid_soft_full_surface_contact=True)
            contacts=pipeline.contacts()
            for frame in range(168):
                controller(e,frame/60,wp)
                for sub in range(12):
                    for key,arr in [('input_joint_target_raw',e.control.joint_target_q),
                                    ('input_joint_rate_raw',e.control.joint_target_qd),('input_joint_effort_raw',e.control.joint_f)]:
                        archive[key].append(arr.numpy().copy())
                    e.state_0.clear_forces();e.state_1.clear_forces();pipeline.collide(e.state_0,contacts)
                    solver.step(e.state_0,e.state_1,e.control,contacts,1/720)
                    e.state_0,e.state_1=e.state_1,e.state_0;record()
                if frame%60==59:print(json.dumps({'case':name,'frame':frame+1,'wall_seconds':time.perf_counter()-start}),flush=True)
        archive={k:np.asarray(v) for k,v in archive.items()}
        if any(not np.isfinite(v).all() for v in archive.values()):raise ValueError('Nonfinite native teacher')
        positions=archive['particle_position'];faces=e.model.tri_indices.numpy();rest=e.model.particle_q.numpy()
        hold=[drape_metrics(rest,x,faces,e.corner_id) for x in positions[round(2.4*720)::12]]
        shape_pass=all(r['panel_width_ratio']>=.55 and r['panel_normal_coherence']>=.60 for r in hold)
        masses=e.model.particle_mass.numpy()
        if not (masses>0).all():raise ValueError('No cloth pins allowed')
        minimum_hold_z=float(positions[round(2.*720):,:,2].min())
        lift=float(positions[-1,e.corner_id,2]-positions[0,e.corner_id,2])
        edges=np.unique(np.sort(np.concatenate([faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]]]),axis=-1),axis=0)
        rest_length=np.linalg.norm(rest[edges[:,0]]-rest[edges[:,1]],axis=-1)
        lengths=np.linalg.norm(positions[:,edges[:,0]]-positions[:,edges[:,1]],axis=-1)/rest_length
        triangles=positions[:,faces];reference=rest[faces]
        area=np.linalg.norm(np.cross(triangles[:,:,1]-triangles[:,:,0],triangles[:,:,2]-triangles[:,:,0]),axis=-1)
        rest_area=np.linalg.norm(np.cross(reference[:,1]-reference[:,0],reference[:,2]-reference[:,0]),axis=-1)
        area_ratio=area/rest_area
        followers=e.model.joint_mimic_joint.numpy()
        follower_ids=np.flatnonzero(followers>=0);starts=e.model.joint_q_start.numpy()
        coeff=e.model.joint_mimic_coeffs.numpy()[follower_ids]
        jq=archive['joint_position']
        mimic_error=np.max(np.abs(jq[:,starts[follower_ids]]-coeff[:,0]-coeff[:,1]*jq[:,starts[followers[follower_ids]]]))
        mesh_pass=bool(lengths.max()<1.6 and lengths.min()>.4 and area_ratio.min()>.25 and mimic_error<1e-5)
        if not shape_pass or lift<.22 or minimum_hold_z<.205 or not mesh_pass:
            rejection=dict(dataset_id=name,fold_profile=fold,split=split,shape=hold[-1],
                corner_lift_m=lift,minimum_hold_z=minimum_hold_z,maximum_stretch=float(lengths.max()),
                minimum_stretch=float(lengths.min()),maximum_mimic_error=float(mimic_error),
                minimum_area_ratio=float(area_ratio.min()),
                training_admitted=False,reason='Native teacher task, broad-panel drape or mesh screen failed')
            manifest['rejected'].append(rejection);write()
            print(json.dumps(rejection),flush=True)
            del solver,pipeline,contacts,scene,e
            continue
        with torch.no_grad():states=tensors(scene,archive,a.device)
        meta={**scene.metadata,'training_schema':SCHEMA,'newton_sha':newton_revision(),
              'builder_sha256':digest(ROOT/'src/wp_vbd/franka_cloth_corner_scene.py'),
              'build_options':options,'dt':1/720,'reference_final_check':'PASS','corner_id':e.corner_id,
              'solver_configuration':configuration,'native_mimic_policy':'final iteration only, every 32 opportunities',
              'teacher_states':len(positions),'native_steps':len(positions)-1,
              'full_drape_qualified':True,'split':split,'is_official_demo':False}
        values={k:v.detach().cpu().numpy() for k,v in scene.values.items()}
        values.update({k:v.detach().cpu().numpy() for k,v in states.items()})
        values.update(source)
        values['source__joint_q']=source_q
        for prefix,spec in [('joint__',scene.joint_spec),('mimic__',scene.mimic_spec)]:
            values.update({prefix+k:v.detach().cpu().numpy() for k,v in vars(spec).items()})
        np.savez_compressed(path,metadata=json.dumps(meta),physical_time=np.arange(len(positions))/720,**values)
        proof={'status':'QUALIFIED_ROBOT_CLOTH_NATIVE_DATA','sha256':digest(path),
               'native_steps':len(positions)-1,'full_state_and_actual_inputs':True,
               'minimum_hold_z':minimum_hold_z,'corner_lift_m':lift,'hold_shape':hold,
               'maximum_stretch':float(lengths.max()),'minimum_stretch':float(lengths.min()),
               'minimum_area_ratio':float(area_ratio.min()),'maximum_area_ratio':float(area_ratio.max()),
               'maximum_mimic_coordinate_error':float(mimic_error),
               'wall_seconds':time.perf_counter()-start,'mimic_projection_calls':adapter.calls}
        path.with_suffix('.qualification.json').write_text(json.dumps(proof,indent=2)+'\n')
        manifest['scenes'].append(dict(dataset_id=name,path=str(path.resolve()),loader='robot_cloth',
            reference_dt=1/720,split=split,sha256=proof['sha256'],fold_profile=fold))
        write();print(json.dumps({'case':name,'status':'QUALIFIED','shape':hold[-1]}),flush=True)
        del solver,pipeline,contacts,scene,e,states,values
    manifest['status']='COMPLETE';write()


if __name__=='__main__':main()
