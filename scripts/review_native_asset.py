"""Inspect actual native asset inference without requiring a teacher trajectory."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-report',type=Path,required=True);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    run=json.loads(args.run_report.read_text())
    reference=run['status']=='OFFLINE_NATIVE_ASSET_REFERENCE_GENERATED' and run.get('reference_only') is True
    if (run['status']!='NATIVE_ASSET_INFERENCE_EXECUTED' and not reference) or run['failure'] is not None:raise ValueError('Complete explicitly labeled native recording required')
    descriptor=Path(run['static_asset_descriptor'])
    if digest(descriptor)!=run['static_asset_descriptor_sha256']:raise ValueError('Captured static asset changed')
    with np.load(descriptor,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']));source={k.removeprefix('source__'):archive[k] for k in archive.files if k.startswith('source__')}
    if metadata['schema']!='native_shared_asset_static_v250' or metadata['provenance']!=run['asset'] or metadata['scene_contract']!=run['scene_contract']:raise ValueError('Bound native source descriptor required')
    trajectory=args.run_report.with_suffix('.npz')
    with np.load(trajectory,allow_pickle=False) as archive:points=archive['particle_q'].astype(float);pose=archive['body_q'].astype(float);steps=archive['physical_substep']
    if len(points)!=run['physical_substeps']+1 or not np.array_equal(steps,np.arange(len(points))):raise ValueError('Complete native state records required')
    if not np.array_equal(points[0],source.get('particle_q',np.empty((0,3)))) or not np.array_equal(pose[0],source.get('body_q',np.empty((0,7)))):raise ValueError('Initial asset and recording differ')
    from wp_vbd.native_geometry_review import measure
    measured=measure(source,points,pose,run['physical_dt'])
    result={'status':'NATIVE_ASSET_GEOMETRY_AND_DEFORMATION_MEASURED','physical_seconds':run['physical_seconds'],
        'recording_kind':'offline_native_VBD_reference' if reference else 'independent_native_model_prediction',
        'physical_substeps':run['physical_substeps'],'asset_name':run['asset']['name'],**measured,
        'full_vbd_prediction_calls':run['full_vbd_prediction_calls'],'teacher_trajectory_reads':1 if reference else 0,
        'offline_vbd_reference_calls':run.get('offline_vbd_reference_calls',{'construct':0,'step':0}),
        'new_optimizer_updates':0,'new_training_admissions':0,'quality_accepted':False,
        'run_report_sha256':digest(args.run_report),'trajectory_sha256':digest(trajectory),'descriptor_sha256':digest(descriptor),
        'script_sha256':digest(Path(__file__)),'measurement_sha256':digest(ROOT/'src/wp_vbd/native_geometry_review.py'),
        'scope':('Offline geometry/deformation measurement of explicitly labeled native VBD reference. ' if reference else
                 'Offline geometry/deformation measurement of actual independent inference; no teacher trajectory is used. ')+
                'Partial sampled checks and finite states do not certify complete nonpenetration, visual plausibility, material/asset generalization or realtime.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status','physical_substeps','tet_deformation','cloth_deformation','strict_transverse_triangle_crossings')},indent=2))


if __name__=='__main__':main()
