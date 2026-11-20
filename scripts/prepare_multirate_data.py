"""Collect native-demo references and index autonomous long-step training windows."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

from wp_vbd.multirate_window import reference_grid,command_events,long_step_window


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(8*1024**2),b''):
            h.update(part)
    return h.hexdigest()


def collect(plan_path,output):
    """Run the registered existing collectors/converters with structured arguments."""
    plan=json.loads(plan_path.read_text())
    from wp_vbd.newton_source import source_root,newton_revision
    if newton_revision()!=plan['newton_sha']:
        raise ValueError('Collection plan and active Newton source revision differ')
    environment={**os.environ,'WP_VBD_NEWTON_ROOT':str(source_root())}
    jobs=plan.get('jobs',[])
    if not jobs:
        raise ValueError('Collection plan requires at least one job')
    for job in jobs:
        if not job['id'] or job['id'] in ('.','..') or Path(job['id']).name!=job['id']:
            raise ValueError('Collection job IDs must be plain filenames')
        commands=job.get('commands',[])
        if not commands:
            raise ValueError('Each collection job requires collector/converter commands')
        for index,command in enumerate(commands):
            script=(ROOT/command['script']).resolve()
            if script.parent!=ROOT/'scripts' or not script.name.startswith(('collect_','generate_','prepare_')):
                raise ValueError('Use existing repository data collection or conversion scripts')
            arguments=command.get('args',[])
            if not all(isinstance(v,str) for v in arguments):
                raise ValueError('Collector arguments must be strings')
            arguments=[value.replace('{output}',str(output.resolve())) for value in arguments]
            # Load the selected Newton before legacy scripts add their historical paths.
            bootstrap=("import sys,runpy;sys.path.insert(0,"+repr(str(ROOT/'src'))+");"
                       "from wp_vbd.newton_source import activate_newton_source;"
                       "activate_newton_source();import newton;"
                       "sys.argv=sys.argv[1:];runpy.run_path(sys.argv[0],run_name='__main__')")
            log=output/(job['id']+f'_{index}.log')
            if log.exists():
                raise FileExistsError(log)
            with log.open('w') as stream:
                subprocess.run([sys.executable,'-u','-c',bootstrap,str(script),*arguments],
                               cwd=ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
    scenes=[]
    for job in jobs:
        if 'scene' not in job:
            continue
        row=dict(job['scene'])
        for key in ('path','validation_path','qualification'):
            if key not in row:
                continue
            path=Path(row[key].replace('{output}',str(output.resolve())))
            if not path.is_absolute():
                path=ROOT/path
            row[key]=str(path.resolve())
            hash_key={'path':'sha256','validation_path':'validation_sha256','qualification':'qualification_sha256'}[key]
            row[hash_key]=digest(path)
        scenes.append(row)
    return {'newton_sha':plan['newton_sha'],'scenes':scenes,
            'collection_plan':str(plan_path.resolve()),'collection_plan_sha256':digest(plan_path)}


def window_catalog(path,row,targets,horizon,validation,maximum_hops=None):
    grid,metadata=reference_grid(path,row.get('dt'))
    events=command_events(path,len(grid.times),hard_only=True)
    starts=row.get('validation_starts') if validation else row.get('qualified_train_starts')
    if not starts:
        if metadata.get('data_layout')=='teacher_queried_windows_v1':
            starts=[a for a,b in grid.segments]
        elif validation:
            starts=[int(.8*(len(grid.times)-1))]
        else:
            lo,hi=row.get('train_start_range',(0,int(.6*(len(grid.times)-1))))
            starts=list(range(lo,hi,max(1,(hi-lo)//64)))
    windows=[]
    rejected=[]
    patterns=[(h,)*horizon for h in targets]
    if len(targets)>1:
        patterns.append(tuple(targets[i%len(targets)] for i in range(horizon)))
    for start in starts:
        for durations in patterns:
            try:
                plan=long_step_window(grid,int(start),durations,events=events)
                if maximum_hops is not None and len(plan['strides'])>maximum_hops:
                    raise ValueError(f'Control events require {len(plan["strides"])} student hops, exceeding budget {maximum_hops}')
            except ValueError as error:
                rejected.append({'start':int(start),'target_durations':list(durations),'reason':str(error)})
                continue
            windows.append(plan)
    steps=np.concatenate([np.diff(grid.times[a:b+1]) for a,b in grid.segments])
    original_dt=metadata.get('dt',row.get('dt'))
    return {
        'reference_dt':None if original_dt is None else float(original_dt),
        'reference_steps':len(grid.times)-1,
        'reference_dt_min':float(steps.min()),'reference_dt_max':float(steps.max()),
        'original_demo':metadata.get('official_demo',row.get('demo')),
        'continuous_segments':[list(segment) for segment in grid.segments],
        'hard_event_count':len(events),'windows':windows,'unavailable_windows':rejected,
        'control_span_policy':'One student call per requested duration; integrate known drive/wrench inputs and use endpoint prescribed poses',
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,help='Existing demo data manifest; optional when collection jobs declare scene rows')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--collection-plan',type=Path,help='Optional collector/converter jobs to execute before indexing')
    parser.add_argument('--target-dts',type=float,nargs='+',default=[1/60,1/30])
    parser.add_argument('--horizon',type=int,default=3)
    parser.add_argument('--maximum-hops',type=int,help='Exclude windows needing more actual student steps after control-event splitting')
    args=parser.parse_args()
    if args.horizon<1 or any(not math.isfinite(h) or h<=0 for h in args.target_dts):
        parser.error('Positive finite target timesteps and rollout horizon required')
    if args.maximum_hops is not None and args.maximum_hops<args.horizon:
        parser.error('Actual student hop budget must cover the requested horizon')
    if args.manifest is None and args.collection_plan is None:
        parser.error('Provide a reference manifest or a collection plan')
    args.output.mkdir(parents=True,exist_ok=False)
    source=json.loads(args.manifest.read_text()) if args.manifest else None
    if args.collection_plan:
        generated=collect(args.collection_plan,args.output)
        if source is None:
            source=generated
        else:
            if generated['newton_sha']!=source['newton_sha']:
                raise ValueError('Collected and existing references use different Newton revisions')
            source={**source,'scenes':[*source['scenes'],*generated['scenes']]}
        (args.output/'source_manifest.json').write_text(json.dumps(source,indent=2)+'\n')
    if not source['scenes']:
        raise ValueError('The source manifest or collection jobs must declare reference scene rows')
    identifiers=[row['dataset_id'] for row in source['scenes']]
    if len(set(identifiers))!=len(identifiers):
        raise ValueError('Dataset IDs must be unique')
    rows=[]
    excluded=[]
    for original in source['scenes']:
        row=dict(original)
        path=Path(row['path'])
        if not path.is_absolute():
            path=ROOT/path
        if digest(path)!=row['sha256']:
            raise ValueError('Reference data changed: '+str(path))
        for label in ('qualification',):
            if label in row:
                proof=Path(row[label])
                if not proof.is_absolute():proof=ROOT/proof
                if digest(proof)!=row[label+'_sha256']:
                    raise ValueError('Reference qualification changed: '+str(proof))
        training=window_catalog(path,row,args.target_dts,args.horizon,False,args.maximum_hops)
        validation_path=Path(row.get('validation_path',row['path']))
        if not validation_path.is_absolute():
            validation_path=ROOT/validation_path
        if digest(validation_path)!=row.get('validation_sha256',row['sha256']):
            raise ValueError('Development reference changed')
        development=window_catalog(validation_path,row,args.target_dts,args.horizon,True,args.maximum_hops)
        if validation_path.resolve()==path.resolve():
            dev_ranges=[(w['start'],w['endpoint_indices'][-1]) for w in development['windows']]
            training['windows']=[w for w in training['windows'] if not any(
                w['start']<=b and w['endpoint_indices'][-1]>=a for a,b in dev_ranges)]
        if row['loader']=='surface':
            # An accepted small teacher step does not certify a macro hop.
            # In particular the authored Franka adapter used to inherit its
            # native 3 mm envelope even for 1/30 s targets, making those
            # endpoints impossible irrespective of learned weights.
            with np.load(path,allow_pickle=False) as archive:
                metadata=json.loads(str(archive['metadata']))
            budget=row.get('detection_motion_budget')
            if budget is None and metadata.get('training_schema')=='authored_franka_substep_reference_v310':
                budget=.003
            if budget is not None:
                from wp_vbd.multirate_motion import audit_endpoint_motion
                audits={'training':audit_endpoint_motion(path,training['windows'],budget),
                        'development':audit_endpoint_motion(validation_path,development['windows'],budget)}
                if any(audit.get('supported') and audit['unreachable_macro_hops'] for audit in audits.values()):
                    raise ValueError(f'{row["dataset_id"]}: macro teacher endpoints exceed the {budget:g} m '
                        'collision-query envelope. Set an explicit detection_motion_budget covering the '
                        'reference motion and regenerate candidates; do not truncate teacher targets.')
                row['macro_motion_envelope_audit']=audits
        row['multirate']={'training':training,'development':development}
        row['path']=str(path.resolve())
        if 'validation_path' in row:
            row['validation_path']=str(validation_path.resolve())
        if not training['windows'] or not development['windows']:
            excluded.append({'dataset_id':row['dataset_id'],'reason':'No disjoint continuous long-step train/development windows',
                             'training_windows':len(training['windows']),'development_windows':len(development['windows'])})
            continue
        rows.append(row)
    result={
        'schema':'shared_vbd_multirate_data_v1','status':'INDEXED' if rows else 'NO_ELIGIBLE_WINDOWS',
        'newton_sha':source['newton_sha'],
        'source_manifest':str((args.output/'source_manifest.json' if args.collection_plan else args.manifest).resolve()),
        'source_manifest_sha256':digest(args.output/'source_manifest.json' if args.collection_plan else args.manifest),
        'target_dts':args.target_dts,
        'horizon':args.horizon,'maximum_hops':args.maximum_hops,'scenes':rows,'excluded':excluded,
        'reference_timing':'Per-demo original timestamps, no uniform teacher-step requirement',
        'scope':'Window construction and source-byte audit; source qualifications remain authoritative',
    }
    (args.output/'data_manifest.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'status':result['status'],'datasets':len(rows),'excluded':excluded},indent=2))
    if not rows:
        raise SystemExit(1)


if __name__=='__main__':
    main()
