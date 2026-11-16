"""Inspect recorded physical surface pairs without running any dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-report',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    run=json.loads(args.run_report.read_text());reference=run.get('reference_only') is True
    expected='OFFLINE_NATIVE_ASSET_REFERENCE_GENERATED' if reference else 'NATIVE_ASSET_INFERENCE_EXECUTED'
    if run['status']!=expected or run['failure'] is not None:raise ValueError('Complete explicitly labeled recording required')
    descriptor=Path(run['static_asset_descriptor']);trajectory=args.run_report.with_suffix('.npz')
    if digest(descriptor)!=run['static_asset_descriptor_sha256']:raise ValueError('Recorded asset changed')
    with np.load(descriptor,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']));source={k.removeprefix('source__'):archive[k] for k in archive.files if k.startswith('source__')}
    if metadata['schema']!='native_shared_asset_static_v250' or metadata['provenance']!=run['asset'] or metadata['scene_contract']!=run['scene_contract']:
        raise ValueError('Captured source does not match the run')
    with np.load(trajectory,allow_pickle=False) as archive:
        points=archive['particle_q'].astype(float);pose=archive['body_q'].astype(float);steps=archive['physical_substep']
    if len(points)!=run['physical_substeps']+1 or not np.array_equal(steps,np.arange(len(points))):raise ValueError('Full contiguous recording required')
    if not np.array_equal(points[0],source.get('particle_q',np.empty((0,3)))) or not np.array_equal(pose[0],source.get('body_q',np.empty((0,7)))):
        raise ValueError('Recorded initial source differs')
    from wp_vbd.native_pair_geometry import measure
    measured=measure(source,points,pose)
    soft_profile=None;tets=source.get('tet_indices',np.empty((0,4),int))
    if len(tets):
        rest=source['particle_q'].astype(float);dm=(rest[tets[:,1:]]-rest[tets[:,:1]]).transpose(0,2,1)
        current=(points[:,tets[:,1:]]-points[:,tets[:,:1]]).transpose(0,1,3,2)
        deformation=current@np.linalg.inv(dm);j=np.linalg.det(deformation)
        k,cell=np.unravel_index(j.argmin(),j.shape);weights=np.abs(np.linalg.det(dm));volume=(j*weights).sum(-1)/weights.sum()
        soft_profile={'minimum_cell_volume_ratio':float(j[k,cell]),'minimum_cell':int(cell),
            'minimum_cell_step':int(k),'minimum_cell_physical_time_s':float(k*run['physical_dt']),
            'minimum_total_signed_volume_ratio':float(volume.min()),'minimum_total_volume_step':int(volume.argmin()),
            'final_total_signed_volume_ratio':float(volume[-1]),
            'scope':'Total over all captured tets, and a separately identified worst local cell. A local near-collapse is not the global object volume ratio.'}
    result={'status':'NATIVE_RECORDED_PRIMITIVE_PAIRS_MEASURED','recording_kind':
        'offline_native_VBD_reference' if reference else ('independent_neural_prediction' if run['learned_corrections_enabled'] else 'independent_physical_control'),
        'physical_substeps':run['physical_substeps'],'physical_seconds':run['physical_seconds'],**measured,
        'soft_volume_profile':soft_profile,'run_report_sha256':digest(args.run_report),'trajectory_sha256':digest(trajectory),
        'descriptor_sha256':digest(descriptor),'script_sha256':digest(Path(__file__)),
        'module_sha256':digest(ROOT/'src/wp_vbd/native_pair_geometry.py'),
        'full_vbd_prediction_calls':run['full_vbd_prediction_calls'],
        'offline_vbd_reference_calls':run.get('offline_vbd_reference_calls',{'construct':0,'step':0}),
        'offline_reference_trajectory_reads':int(reference),'teacher_trajectory_reads_during_prediction':0,
        'new_optimizer_updates':0,'new_training_admissions':0,
        'scope':'Offline observer of previously recorded physical states. No inference, dynamics modification, geometry projection, candidate-row reuse or tolerance based on contact margins. Sampled pair checks do not certify continuous collision avoidance, all containment or physical plausibility.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    nonjoint=[x for x in result['rigid_primitive_pairs'] if not x['direct_joint_connection']]
    print(json.dumps({'status':result['status'],'recording_kind':result['recording_kind'],
        'triangle_shape_overlap_steps':sum(x['recorded_steps_with_strict_overlap'] for x in result['shape_triangle_overlaps']),
        'nonjoint_rigid_pair_overlap_steps':sum(x['recorded_steps_with_strict_overlap'] for x in nonjoint),
        'soft_volume_profile':soft_profile},indent=2))


if __name__=='__main__':main()
