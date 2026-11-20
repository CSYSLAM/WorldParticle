"""Offline native VBD teacher for the actual five-family research asset."""
import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import source_root
NEWTON=source_root()
sys.path[:0]=[str(ROOT/'src'),str(NEWTON)]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parameters',type=Path)
    parser.add_argument('--asset',type=Path,help='Static native JSON asset; mutually exclusive with legacy --parameters')
    parser.add_argument('--set',action='append',default=[],dest='overrides',help='Existing asset.path=JSON physical parameter override')
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--frames',type=int,default=300)
    parser.add_argument('--substeps',type=int)
    parser.add_argument('--dt',type=float)
    parser.add_argument('--duty-cycle',type=float,default=.25)
    parser.add_argument('--self-topology',choices=('original','reciprocal-union'),
                        help='Explicit source contact profile for this new authored asset')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    if args.device=='cpu':os.environ['CUDA_VISIBLE_DEVICES']=''
    import math
    if args.asset and (args.parameters or args.self_topology):parser.error('JSON asset embeds its own physical and reciprocal topology definition')
    if args.overrides and not args.asset:parser.error('--set requires a native JSON asset')
    authored=None
    if args.asset:
        from wp_vbd.native_assets import read
        authored=read(args.asset,args.overrides)
        simulation=authored.get('simulation',{})
        for key,default in (('dt',1/600),('substeps',10)):
            value=simulation.get(key,default)
            if getattr(args,key) is not None and getattr(args,key)!=value:parser.error('JSON asset defines '+key+'; use an explicit asset override')
            setattr(args,key,value)
    else:
        args.dt=1/600 if args.dt is None else args.dt
        args.substeps=10 if args.substeps is None else args.substeps
        args.self_topology=args.self_topology or 'original'
    if args.frames<=0 or args.substeps<=0 or not math.isfinite(args.dt) or args.dt<=0:
        parser.error('Positive frame/substep counts and finite positive dt required')
    if not math.isfinite(args.duty_cycle) or not 0<args.duty_cycle<=1:parser.error('Duty cycle must lie in (0,1]')
    if args.output.exists() or args.report.exists():raise FileExistsError('Preserve teacher collection artifacts')
    parameters=None if authored is not None else json.loads((args.parameters or ROOT/'configs/native_five_families.json').read_text())
    if parameters is not None and set(parameters)!={'friction','elastic_scale','mass_scale'}:
        parser.error('Complete five-family physical parameter file required')
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(args.device,ROOT)
    import numpy as np
    import warp as wp
    import newton
    from wp_vbd.elastic_rod_demo import build_asset
    from wp_vbd.official import solver_configuration_record
    from wp_vbd.elastic_rod_configuration import reciprocal_configuration,record_configuration
    from wp_vbd.reference_tape import (TapeWriter,arrays,auxiliary,contact_arrays,self_contact_arrays,
                                       MUTABLE_MODEL_FIELDS,TRACE_FEATURE_SET)
    from wp_vbd.gpu_budget import GpuBudget
    budget=GpuBudget(args.device,12.)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(args.device)
    configuration={k:v.default for k,v in inspect.signature(newton.solvers.SolverVBD).parameters.items()
                   if k not in ('model','collision_pipeline')}
    configuration.update(particle_enable_self_contact=True,particle_self_contact_margin=.05,
                         particle_self_contact_gap=.02,deterministic=wp.DeterministicMode.RUN_TO_RUN)
    source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=NEWTON,text=True).strip()
    if source_sha!='25298aea1691e060c86f95e8d2b773c898c006d3':raise ValueError('Pinned Newton source changed')
    writer=TapeWriter(args.output,chunk_steps=16,max_bytes=1024**3)
    report={'id':'research.native_five_families','status':'RUNNING','source_sha':source_sha,
        'frames_requested':args.frames,'frames_completed':0,'substeps_per_frame':args.substeps,
        'dt':args.dt,'device':args.device,'vbd_instances':0,'reference_step_calls':0,
        'duty_cycle':args.duty_cycle,
        'trace_feature_set':TRACE_FEATURE_SET,'physical_parameters':parameters,
        'configuration':solver_configuration_record(configuration),'training_admission':False,
        'new_optimizer_updates':0,'new_training_admissions':0,'neural_prediction_calls':0,
        'full_vbd_prediction_calls':0,'collision_refresh_policy':'fresh_analytic_shape_query_each_substep_native_self_schedule',
        'shape_full_surface':True,'failure':None,
        'collector_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'factory_sha256':hashlib.sha256((ROOT/'src/wp_vbd/elastic_rod_demo.py').read_bytes()).hexdigest(),
        'tape_module_sha256':hashlib.sha256((ROOT/'src/wp_vbd/reference_tape.py').read_bytes()).hexdigest(),
        'self_topology_policy':'asset_defined_reciprocal_union' if authored is not None else args.self_topology,
        'scope':'Native VBD offline teacher only, with default iterations and explicit authored asset/physical inputs/contact profile. No neural model loaded. Recorded trace is not training admission, source convergence or learned quality.'}
    begin=time.perf_counter()
    try:
        if authored is not None:
            from wp_vbd.native_assets import build
            from wp_vbd.native_asset_training_definition import capture
            asset,configuration,description=build(authored,newton,wp,device=args.device)
            if not asset.particle_count or not description['has_native_rods']:
                raise ValueError('This training trace requires elastic particles and native rods')
            report['asset_definition']=capture(description,configuration)
            report['factory_sha256']=report['asset_definition']['builder_sha256']
            report['source_self_contact_definition']=description['source_topology']
        else:asset,description=build_asset(newton,wp,device=args.device,**parameters)
        if authored is None and args.self_topology=='reciprocal-union':
            configuration,evidence=reciprocal_configuration(asset,configuration)
            report['source_self_contact_definition']=evidence
        report['configuration']=record_configuration(configuration)
        pipeline=newton.CollisionPipeline(asset,deterministic=True,rigid_contact_max=4096,
            soft_contact_max=8192,enable_rigid_soft_full_surface_contact=True,publish_capacity=False)
        # Explicit external geometry buffers: the native API forbids passing
        # contacts to a solver that also owns its own collision pipeline.
        solver=newton.solvers.SolverVBD(asset,**configuration)
        report.update(vbd_instances=1,asset=description,native_solver_iterations=solver.iterations,
            contact_storage={'rigid':4096,'soft':8192,'policy':'Explicit storage only; overflow rejects, no contact truncation.'})
        static=arrays(asset,prefix='solver0__model__')
        static.update(arrays(solver,names=('joint_constraint_start','joint_constraint_dim','joint_material_k',
            'joint_penalty_kd','joint_is_hard','joint_rest_angle','joint_rod_rest_kb_local',
            'joint_rod_rest_twist','joint_dahl_eps_max','joint_dahl_tau'),prefix='solver0__definition__'))
        static_path=args.output/'static.npz';np.savez_compressed(static_path,**static)
        report['static']={'path':'static.npz','sha256':hashlib.sha256(static_path.read_bytes()).hexdigest(),'fields':len(static)}
        native=asset.state();output=asset.state();control=asset.control();contacts=pipeline.contacts()
        for frame in range(args.frames):
            for _ in range(args.substeps):
                started=time.perf_counter()
                native.clear_forces();pipeline.collide(native,contacts)
                fields={}
                def collect(owner,prefix,function=arrays):
                    fields.update({prefix+'__'+k:v for k,v in function(owner).items()})
                collect(native,'input');collect(control,'control')
                fields.update({'model_input__'+k:v for k,v in arrays(asset,names=MUTABLE_MODEL_FIELDS).items()})
                collect(solver,'aux_before',auxiliary);collect(contacts,'contacts_before',contact_arrays)
                collect(solver,'self_contact_before',self_contact_arrays)
                report['reference_step_calls']+=1
                solver.step(native,output,control,contacts,args.dt)
                collect(output,'output');collect(solver,'aux_after',auxiliary)
                collect(contacts,'contacts_after',contact_arrays);collect(solver,'self_contact_after',self_contact_arrays)
                if any(not np.isfinite(v).all() for k,v in fields.items()
                       if k.startswith(('input__','output__')) and v.dtype.kind=='f'):
                    raise ValueError('Nonfinite native reference state')
                writer.append(fields,{'solver_slot':0,'sequence':writer.steps,'frame':frame,'dt':args.dt,
                                      'observables_requested':[],'controller':'constant_physical_parameters_and_fixed_pins'})
                native,output=output,native
                if args.device.startswith('cuda'):
                    wp.synchronize_device();delay=(time.perf_counter()-started)*(1/args.duty_cycle-1)
                    while delay>0:
                        part=min(delay,.25);time.sleep(part);delay-=part
            report['frames_completed']=frame+1;budget.observe()
            if (frame+1)%10==0:
                writer.manifest(report)
                print(f'frames={frame+1} reference_steps={writer.steps}',flush=True)
        report['status']='PHYSICAL_SUBSTEP_TRACE_READY'
    except Exception as error:
        report.update(status='REFERENCE_COLLECTION_FAILED',failure=str(error),traceback=traceback.format_exc())
    report['seconds']=time.perf_counter()-begin;report['recorded_reference_steps']=writer.steps
    report['gpu_resource_budget']=budget.report()
    result=writer.manifest(report)
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('chunks','configuration')},indent=2))
    if result['status']!='PHYSICAL_SUBSTEP_TRACE_READY':raise SystemExit(1)


if __name__=='__main__':main()
