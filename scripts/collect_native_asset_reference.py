"""CPU-only offline VBD reference for an authored asset; never model inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--asset',type=Path,required=True)
    parser.add_argument('--set',action='append',default=[],help='Existing asset.path=JSON physical parameter override')
    parser.add_argument('--frames',type=int,default=60);parser.add_argument('--iterations',type=int,default=10)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.frames<=0 or args.iterations<=0:parser.error('Positive reference budgets required')
    if any(p.exists() for p in (args.report,args.report.with_suffix('.npz'),args.report.with_suffix('.asset.npz'))):raise FileExistsError(args.report)
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd.native_assets import read,build
    from wp_vbd.reference_tape import arrays
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    source=Path('/home/oem/code/engine/newton');commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    if commit!='25298aea1691e060c86f95e8d2b773c898c006d3' or subprocess.run(['git','diff','--quiet'],cwd=source).returncode:
        raise ValueError('Pinned clean tracked Newton source required')
    model,configuration,provenance=build(read(args.asset,args.set),newton,wp,device='cpu');configuration['iterations']=args.iterations
    from wp_vbd.elastic_rod_configuration import record_configuration
    requested_configuration=record_configuration(configuration)
    # The pinned native solver's self-contact branch assumes a particle
    # detector exists. A zero-particle model has no such interaction.
    inactive_particle_self_contact_disabled=not model.particle_count and configuration['particle_enable_self_contact']
    if inactive_particle_self_contact_disabled:configuration={**configuration,'particle_enable_self_contact':False}
    static=arrays(model);args.report.parent.mkdir(parents=True,exist_ok=True);descriptor=args.report.with_suffix('.asset.npz')
    np.savez_compressed(descriptor,metadata=json.dumps({'schema':'native_shared_asset_static_v250',
        'scene_contract':'offline_native_VBD_reference','provenance':provenance}),**{'source__'+k:v for k,v in static.items()})
    solver=newton.solvers.SolverVBD(model,**configuration)
    after=arrays(model);changed=[key for key in static if key not in after or not np.array_equal(static[key],after[key])]
    rigid_capacity=max(8192,16*model.shape_count**2)
    soft_capacity=max(8192,8*(model.particle_count+model.tri_count+model.edge_count)*model.shape_count)
    pipeline=newton.CollisionPipeline(model,deterministic=True,rigid_contact_max=rigid_capacity,
        soft_contact_max=soft_capacity,enable_rigid_soft_full_surface_contact=True,publish_capacity=False)
    contacts=pipeline.contacts();incoming=model.state();outgoing=model.state();control=model.control()
    trajectory=[];steps=0;failure=None;began=time.perf_counter();maximum_counts={'rigid':0,'soft':0}
    def record():
        particles=incoming.particle_q.numpy().copy() if incoming.particle_q is not None else np.empty((0,3),np.float32)
        bodies=incoming.body_q.numpy().copy() if incoming.body_q is not None else np.empty((0,7),np.float32)
        trajectory.append((particles,bodies))
    record()
    try:
        for _ in range(args.frames*provenance['substeps']):
            incoming.clear_forces();pipeline.collide(incoming,contacts)
            for kind,capacity in [('rigid',rigid_capacity),('soft',soft_capacity)]:
                count=int(getattr(contacts,kind+'_contact_count').numpy()[0]);maximum_counts[kind]=max(maximum_counts[kind],count)
                if count>=capacity:raise RuntimeError('Reference collision buffer reached capacity: '+kind)
            solver.step(incoming,outgoing,control,contacts,provenance['dt']);steps+=1;incoming,outgoing=outgoing,incoming
            record()
            if not all(np.isfinite(value).all() for value in trajectory[-1]):raise RuntimeError('Nonfinite offline reference')
    except (RuntimeError,ValueError,AssertionError,AttributeError) as error:failure=type(error).__name__+': '+str(error)
    np.savez_compressed(args.report.with_suffix('.npz'),particle_q=np.stack([v[0] for v in trajectory]),
        body_q=np.stack([v[1] for v in trajectory]),physical_substep=np.arange(len(trajectory)))
    result={'status':'OFFLINE_NATIVE_ASSET_REFERENCE_GENERATED' if failure is None else 'OFFLINE_NATIVE_ASSET_REFERENCE_INCOMPLETE',
        'recording_kind':'offline_native_VBD_reference','reference_only':True,'asset':provenance,
        'scene_contract':'offline_native_VBD_reference','physical_substeps':steps,'physical_dt':provenance['dt'],
        'physical_seconds':steps*provenance['dt'],'seconds':time.perf_counter()-began,'iterations':args.iterations,
        'failure':failure,'source_constructor_changed_numeric_fields':changed,
        'requested_native_solver_configuration':requested_configuration,
        'effective_native_solver_configuration':record_configuration(configuration),
        'inactive_particle_self_contact_disabled':bool(inactive_particle_self_contact_disabled),
        'collision_capacities':{'rigid':rigid_capacity,'soft':soft_capacity},'maximum_observed_contacts':maximum_counts,
        'static_asset_descriptor':str(descriptor),'static_asset_descriptor_sha256':digest(descriptor),
        'offline_vbd_reference_calls':{'construct':1,'step':steps},'full_vbd_prediction_calls':{'construct':0,'step':0},
        'neural_model_constructed_or_loaded':False,'new_optimizer_updates':0,'new_training_admissions':0,
        'newton_commit':commit,'script_sha256':digest(Path(__file__)),
        'scope':'Explicit offline native reference benchmark. No NN model is loaded or corrected. This pose-only recording lacks qualified accepted-history training data and does not certify source convergence, force equivalence, complete nonpenetration or realtime.'}
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ('status','physical_substeps','seconds','failure','source_constructor_changed_numeric_fields','offline_vbd_reference_calls')},indent=2))
    if failure:raise SystemExit(1)


if __name__=='__main__':main()
