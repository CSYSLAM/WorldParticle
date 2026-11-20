"""Generate independent randomized native-VBD trajectories, not reused windows."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--per-family',type=int,default=33)
    p.add_argument('--samples',type=int,default=126,help='Recorded 180-Hz intervals; each contains ten native fine steps')
    p.add_argument('--device',default='cuda:0');p.add_argument('--families',nargs='*')
    p.add_argument('--resume',action='store_true')
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=a.resume)
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd.augmentation_data import FAMILIES,SCHEMA,specification,build,export_scene
    from wp_vbd.gpu_ownership import claim_cuda
    from wp_vbd import math3d as so3
    from wp_vbd.data import State
    from wp_vbd.elastic_rod_configuration import record_configuration
    from wp_vbd.general_joint_export import read_controls
    ownership=claim_cuda(a.device,ROOT)
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(a.device)
    if a.device.startswith('cuda'):
        torch.cuda.set_stream(torch.cuda.ExternalStream(wp.get_stream(a.device).cuda_stream,device=a.device))
    families=a.families or FAMILIES
    report={'status':'RUNNING','newton_sha':newton_revision(),'schema':SCHEMA,'scenes':[],
            'rejected':[],'native_vbd_steps':0,'new_optimizer_updates':0,'seed':20261119,
            'counting':'One independently initialized and simulated asset/material/control trajectory per scene'}
    if a.resume and (a.output/'collection.json').exists():report=json.loads((a.output/'collection.json').read_text())
    existing={r['dataset_id'] for r in report['scenes']};began=time.perf_counter()
    def persist():
        tmp=a.output/'collection.tmp';tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(a.output/'collection.json')
    def collect(family,seed,dataset):
        spec=specification(family,seed);model,cfg,provenance=build(spec,None,newton,wp,a.device)
        meta={'schema':SCHEMA,'family':family,'dataset_id':dataset,'dt':1/180,
              'native_dt':1/1800,'specification':spec,'configuration':record_configuration(cfg),
              'seed':seed,'newton_sha':report['newton_sha']}
        scene=export_scene(model,cfg,meta,newton,wp,a.device)
        solver=newton.solvers.SolverVBD(model,**cfg)
        pipeline=newton.CollisionPipeline(model,deterministic=True,rigid_contact_max=4096,
            soft_contact_max=8192,enable_rigid_soft_full_surface_contact=True,publish_capacity=False)
        contacts=pipeline.contacts();incoming=model.state();outgoing=model.state();control=model.control()
        n=model.particle_count;b=model.body_count;steps=0
        def fine_chunk():
            nonlocal incoming,outgoing
            for _ in range(10):
                incoming.clear_forces();pipeline.collide(incoming,contacts)
                solver.step(incoming,outgoing,control,contacts,1/1800)
                incoming,outgoing=outgoing,incoming
        # Warm kernels before capture; recorded start is this accepted state.
        fine_chunk();steps+=10
        if a.device.startswith('cuda'):
            with wp.ScopedCapture(device=model.device) as capture:fine_chunk()
            graph=capture.graph
        else:graph=None
        T=a.samples+1
        pq=torch.empty(T,n,3,device=a.device);pv=torch.empty_like(pq)
        bq=torch.empty(T,b,7,device=a.device);bv=torch.empty(T,b,6,device=a.device)
        counts=torch.zeros(T,3,device=a.device,dtype=torch.long)
        J=len(scene.joint_spec.owners)
        targets=torch.zeros(T,J,6,device=a.device,dtype=torch.float64)
        rates=torch.zeros_like(targets);effort=torch.zeros_like(targets)
        dual=torch.zeros(T,len(scene.rod.joints) if hasattr(scene,'rod') else 0,6,device=a.device,dtype=torch.float64)
        source_bodies=wp.to_torch(model.shape_body).long()
        slots=torch.arange(contacts.soft_contact_max,device=a.device)
        drive=wp.to_torch(control.joint_target_q) if control.joint_target_q is not None else None
        amplitude=float(np.random.default_rng(seed+1).uniform(.052,.072))
        if family=='joint_soft':
            size=next(o['geometry']['size'][2] for o in spec['objects'] if o['family']=='softbody')
            amplitude=.118-(size+.006)+size*float(np.random.default_rng(seed+2).uniform(.1,.22))
        def record(i):
            if n:
                pq[i].copy_(wp.to_torch(incoming.particle_q));pv[i].copy_(wp.to_torch(incoming.particle_qd))
            if b:
                bq[i].copy_(wp.to_torch(incoming.body_q));bv[i].copy_(wp.to_torch(incoming.body_qd))
            counts[i,0]=wp.to_torch(contacts.rigid_contact_count)[0]
            counts[i,1]=wp.to_torch(contacts.soft_contact_count)[0]
            shape=wp.to_torch(contacts.soft_contact_shape).long().clamp(0,len(source_bodies)-1)
            counts[i,2]=((slots<counts[i,1])&(source_bodies[shape]>=0)).sum()
            if hasattr(scene,'rod'):
                ids=scene.rod.source_joint_ids
                dual[i].copy_(torch.cat((wp.to_torch(solver.joint_lambda_lin)[ids],wp.to_torch(solver.joint_lambda_ang)[ids]),-1))
        record(0)
        for i in range(a.samples):
            if family in ('joint','joint_soft'):
                # Smooth press/retract, held during each recorded interval.
                drive[0]=-amplitude*np.sin(np.pi*i/a.samples)**2
            for out,value in zip((targets,rates,effort),read_controls(scene.joint_spec,control)):out[i].copy_(value[0])
            if graph is not None:wp.capture_launch(graph)
            else:fine_chunk()
            steps+=10;record(i+1)
        for out in (targets,rates,effort):out[-1].copy_(out[-2])
        x=torch.zeros(T,scene.n,3,device=a.device,dtype=torch.float64)
        q=torch.zeros(T,scene.n,4,device=a.device,dtype=torch.float64);q[...,3]=1
        v=torch.zeros_like(x);omega=torch.zeros_like(x)
        x[:,:n]=pq;v[:,:n]=pv
        if b:
            q[:,n:n+b]=so3.normalize(bq[...,3:].double())
            x[:,n:n+b]=bq[...,:3]+so3.rotate(q[:,n:n+b],scene.body_com[n:n+b])
            v[:,n:n+b]=bv[...,:3];omega[:,n:n+b]=bv[...,3:]
        state=State(x,q,v,omega)
        if hasattr(scene,'rod'):
            from wp_vbd import elastic_rod_system as physics,rod_contact_system as rod
            if bool(scene.rod.rod_dahl_eps.any()|scene.rod.rod_dahl_tau.any()):raise ValueError('Generator requires explicit elastic rod history')
            history=physics.initial_history(scene,state)
            history.update(rod_dual=dual,rod_kappa=rod.strain(scene,state)[...,3:],
                           rod_initialized=torch.ones(T,dtype=torch.bool,device=a.device))
        else:
            angle=scene.joint_coordinates(state,scene.joint_spec.coordinate_offset[None,:,3],jacobian=False).axis[...,3]
            history={'pin_x_prev':x[:,scene.fixed_particle_ids].clone(),
                     'pin_initialized':torch.ones(T,dtype=torch.bool,device=a.device),
                     'joint_angle_prev':torch.where(scene.joint_spec.angular_mode==1,angle,0.)}
        finite=all(bool(torch.isfinite(t).all()) for t in (x,q,v,omega))
        lower=1.;upper=1.
        if len(scene.tets):
            points=x[:,scene.tets];ds=torch.stack([points[:,:,k]-points[:,:,0] for k in (1,2,3)],-1)
            ratio=torch.linalg.det(ds@scene.tet_Dm);lower=float(ratio.min());upper=float(ratio.max())
        speed=float(v.norm(dim=-1).amax());maximum=counts.amax(0).cpu().tolist()
        if not finite or lower<.08 or upper>5 or speed>8:raise ValueError(f'Unreasonable native trajectory: finite={finite}, J=[{lower},{upper}], speed={speed}')
        if maximum[0]>=4096 or maximum[1]>=8192:raise ValueError('Native contact overflow')
        coupling=family in ('rigid_soft','rigid_cloth','joint_soft','multitype','cable_cloth')
        if coupling and not maximum[2]:raise ValueError('No observed dynamic rigid/deformable contact candidates')
        qualification={'status':'PASS','recorded_states_finite':finite,'minimum_tet_volume_ratio':lower,
            'maximum_tet_volume_ratio':upper,'maximum_speed_m_s':speed,'maximum_contact_counts':maximum,
            'scope':'Finite recorded states, bounded material volume and speed, no contact capacity overflow; discrete screening, not continuous collision proof.'}
        meta.update(qualification=qualification,native_vbd_steps=steps,record_stride_fine_steps=10)
        output={key:value.detach().cpu().numpy() for key,value in {'x':x,'q':q,'v':v,'omega':omega,
            **{'history__'+k:t for k,t in history.items()},'input_pin_x':x[:,scene.fixed_particle_ids],
            'input_joint_target':targets,'input_joint_target_rate':rates,'input_joint_effort':effort,
            'input_force':torch.zeros_like(x),'input_torque':torch.zeros_like(x)}.items()}
        path=a.output/(dataset+'.npz');np.savez_compressed(path,metadata=json.dumps(meta),**output)
        return {'dataset_id':dataset,'family':family,'seed':seed,'loader':'augmentation','path':str(path.resolve()),
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'qualification':qualification,
                'native_vbd_steps':steps,'reference_dt':1/180,'states':T,
                'split':'development' if int(dataset.rsplit('_',1)[-1])%11==10 else 'training'}
    for family in families:
        for index in range(a.per_family):
            dataset=f'aug_{family}_{index:03d}'
            if dataset in existing:continue
            for retry in range(8):
                seed=20261119+FAMILIES.index(family)*10000+index*17+retry
                try:
                    row=collect(family,seed,dataset);report['scenes'].append(row)
                    report['native_vbd_steps']+=row['native_vbd_steps'];break
                except Exception as error:
                    report['rejected'].append({'dataset_id':dataset,'seed':seed,'error':str(error)})
                    persist();print('REJECT',dataset,repr(error),flush=True)
                    if retry==7:raise
                finally:gc.collect()
            report['elapsed_seconds']=time.perf_counter()-began;persist()
            print(json.dumps({'qualified':len(report['scenes']),'dataset':dataset,'seconds':report['elapsed_seconds']}),flush=True)
    report['status']='COMPLETE';persist()


if __name__=='__main__':main()
