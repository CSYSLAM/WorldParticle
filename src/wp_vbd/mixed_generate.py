"""Batched offline teachers, with an explicit finite-penalty target contract."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .data import ROOT


def specifications(config):
    from .assets import load_asset_spec
    base=load_asset_spec(ROOT/config['asset'])
    specs=[]
    for i in range(10):
        s=deepcopy(base)
        friction={1:0.,2:.8,8:.3,9:.55}.get(i,.4)
        # VBD's particle-side coefficient is a global scalar. Keep it at .4
        # in the batch and change each world's shape coefficient instead.
        s['plane'].update(particle_mu=.4,ground_mu=friction**2/.4)
        for p in s['objects']:
            p['friction']=.4
            if p['family']=='softbody':
                factor={3:.5,4:2.,8:1.25,9:.75}.get(i,1.)
                p['material']['mu']*=factor;p['material']['lmbd']*=factor
            if p['family']=='cloth' and i in (5,8,9):
                factor={5:.4,8:.8,9:1.15}[i]
                for key in ('mu','lmbd','bending'):p['material'][key]*=factor
            if p['family']=='rod' and i in (6,7,8,9):
                factor={6:.25,7:4.,8:1.75,9:2.}[i]
                p['material']['bending']*=factor;p['material']['twist']*=factor
        specs.append(s)
        for attachment in s.get('attachments',[]):attachment['stiffness']*={1:.25,2:4.,3:.5,4:2.,8:1.5,9:.75}.get(i,1.)
    return specs


def make_teacher(model,iterations,newton,wp):
    pipeline=newton.CollisionPipeline(model,deterministic=True,soft_contact_gap=.04)
    solver=newton.solvers.SolverVBD(model,iterations=iterations,collision_pipeline=pipeline,particle_enable_self_contact=False,rigid_soft_enable_dat=False,deterministic=wp.DeterministicMode.RUN_TO_RUN,rigid_compliant_alm=False,rigid_contact_hard=False,rigid_avbd_beta=0.,rigid_avbd_alpha=0.,rigid_joint_linear_ke=2e4,rigid_joint_angular_ke=2e3,friction_epsilon=.01)
    for j,jt in enumerate(model.joint_type.numpy()):
        if int(jt)!=int(newton.JointType.FREE):solver.set_joint_constraint_mode(j,False)
    return solver


def generate(config, *, specs=None):
    begin=time.perf_counter()
    sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=config['newton_root'],text=True).strip()
    if sha!=config['newton_sha']:raise RuntimeError('Teacher source pin differs')
    sys.path.insert(0,config['newton_root'])
    import warp as wp
    import newton
    from .mixed_demo import build_mixed
    from .mixed_state import export_mixed
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cuda:0')
    specs=specifications(config) if specs is None else specs
    builder=newton.ModelBuilder(gravity=tuple(specs[0]['gravity']))
    scenes=[];offsets=[]
    for spec in specs:
        sub,groups=build_mixed(spec,'cuda:0',newton,wp,builder_only=True)
        static_model=sub.finalize(device='cuda:0')
        static,metadata=export_mixed(static_model,groups,spec,newton)
        scenes.append((static,metadata))
        offsets.append((builder.particle_count,builder.body_count,static_model.particle_count,static_model.body_count))
        builder.begin_world();builder.add_builder(sub);builder.end_world()
    builder.color();model=builder.finalize(device='cuda:0')
    model.soft_contact_ke=5000.;model.soft_contact_kd=0.;model.soft_contact_mu=.4
    out=ROOT/config['data_dir'];out.mkdir(parents=True,exist_ok=True)
    report_dir=ROOT/config['report_dir'];report_dir.mkdir(parents=True,exist_ok=True)
    def simulate(iterations):
        solver=make_teacher(model,iterations,newton,wp)
        a,b=model.state(),model.state();control=model.control()
        def step():
            a.clear_forces();solver.step(a,b,control,None,config['dt'])
            for key in ('particle_q','particle_qd','body_q','body_qd'):wp.copy(getattr(a,key),getattr(b,key))
        def snapshot():return [getattr(a,key).numpy().copy() for key in ('particle_q','particle_qd','body_q','body_qd')]
        states=[snapshot()]
        with wp.ScopedCapture(device='cuda:0') as capture:step()
        for i in range(config['steps']):
            wp.capture_launch(capture.graph);states.append(snapshot())
            if (i+1)%60==0:print(f'teacher {iterations}: {i+1}/{config["steps"]} ({time.perf_counter()-begin:.1f}s)',flush=True)
        return [np.stack([row[k] for row in states]) for k in range(4)]
    coarse=simulate(config['teacher_iterations']);fine=simulate(config['convergence_iterations'])
    manifest={'status':'RUNNING','config':config,'source_sha':sha,'contract':config['contract'],'scenes':[],'convergence':[],'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    for i,((static,metadata),(p0,b0,np_,nb)) in enumerate(zip(scenes,offsets)):
        def convert(sequence):
            pq,pv,bq,bv=sequence
            q=bq[:,b0:b0+nb,3:];com=static['body_com'][np_:np_+nb]
            rotate=lambda quat,v:v+2*np.cross(quat[...,:3],np.cross(quat[...,:3],v)+quat[...,3:]*v)
            x_body=bq[:,b0:b0+nb,:3]+rotate(q,com)
            t=len(pq);n=len(static['rest_x'])
            result={key:np.zeros((t,n,4 if key=='q' else 3),np.float32) for key in ('x','q','v','omega')}
            result['q'][...,3]=1
            result['x'][:,:np_]=pq[:,p0:p0+np_];result['v'][:,:np_]=pv[:,p0:p0+np_]
            result['x'][:,np_:np_+nb]=x_body;result['q'][:,np_:np_+nb]=q
            result['v'][:,np_:np_+nb]=bv[:,b0:b0+nb,:3];result['omega'][:,np_:np_+nb]=bv[:,b0:b0+nb,3:]
            return result
        trajectory,refined=convert(coarse),convert(fine)
        error=float(np.linalg.norm(trajectory['x']-refined['x'],axis=-1).max()/.4)
        p=trajectory['x'][:,static['tets']];f=(p[:,:,1:]-p[:,:,:1]).swapaxes(-1,-2)@static['tet_Dm']
        determinant=float(np.linalg.det(f).min())
        q=trajectory['q'][:,static['site_owner']];local=static['site_local'];point=trajectory['x'][:,static['site_owner']]+local+2*np.cross(q[...,:3],np.cross(q[...,:3],local)+q[...,3:]*local)
        clearance=float((point[...,2]-static['site_radius']-static['plane_height']).min())
        passed=error<=config['convergence_tolerance_over_L'] and determinant>=.05 and clearance>=0 and all(np.isfinite(x).all() for x in trajectory.values())
        split=config.get('fixed_split','train' if i<8 else 'validation' if i==8 else 'test')
        metadata.update(index=i,split=split,source_sha=sha)
        path=out/f'mixed_{i:02d}.npz'
        np.savez_compressed(path,**static,**trajectory,metadata=np.asarray(json.dumps(metadata)))
        manifest['scenes'].append({'index':i,'split':split,'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'minimum_detF':determinant,'minimum_physical_plane_clearance':clearance,'status':'PASS' if passed else 'FAIL'})
        manifest['convergence'].append({'index':i,'iterations':[config['teacher_iterations'],config['convergence_iterations']],'maximum_position_difference_over_L':error,'status':'PASS' if passed else 'FAIL'})
    manifest['status']='PASS' if all(row['status']=='PASS' for row in manifest['scenes']) else 'FAIL'
    manifest['wall_seconds']=time.perf_counter()-begin
    (report_dir/'data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
    if manifest['status']!='PASS':raise RuntimeError('Teacher geometric or convergence check failed; data retained')
    return manifest
