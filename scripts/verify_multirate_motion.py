"""Check macro endpoint reachability and the actual surface query adapter."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    from wp_vbd.multirate_motion import surface_travel,audit_endpoint_motion
    # A stationary rigid COM can still rotate its surface outside the
    # envelope. Normalize quaternions and handle their double cover.
    q=np.array([[0.,0.,0.,1.]])
    rotated=np.array([[0.,0.,np.sin(.2),np.cos(.2)]])
    moved=surface_travel(np.zeros((1,3)),q,np.array([[.002,0.,0.]]),rotated,np.array([.01]))
    np.testing.assert_allclose(moved,[.006],rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(surface_travel(np.zeros((1,3)),q*2,np.zeros((1,3)),-q*3,np.array([.1])),[0.],atol=1e-15)
    with tempfile.TemporaryDirectory() as temporary:
        path=Path(temporary)/'reference.npz'
        x=np.zeros((3,2,3));x[1,0,0]=.005;x[2,0,0]=.006;x[:,1,0]=np.arange(3)
        quat=np.broadcast_to(q,(3,2,4)).copy()
        np.savez_compressed(path,x=x,q=quat,body_radius=[.01,.01],movable=[True,False])
        windows=[{'start':0,'strides':[1,1],'durations':[1/60,1/60]}]
        before=audit_endpoint_motion(path,windows,.003);after=audit_endpoint_motion(path,windows,.012)
        assert before['macro_hops']==2 and before['unreachable_macro_hops']==1
        assert after['unreachable_macro_hops']==0
        assert before['violations'][0]['start']==0 and before['violations'][0]['endpoint']==1
        np.testing.assert_allclose(after['maximum_teacher_surface_travel_m'],.005)
        native=Path(temporary)/'native.npz';np.savez_compressed(native,particle_position=x)
        assert not audit_endpoint_motion(native,windows,.012)['supported']

    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.multirate_data import load_reference
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    manifest=json.loads(args.manifest.read_text())
    assert manifest['newton_sha']==newton_revision()
    row=next(row for row in manifest['scenes'] if row['dataset_id']=='franka_block_mu5000_lambda5000_grasp_v310')
    with tempfile.TemporaryDirectory() as temporary:
        temporary=Path(temporary)
        original=dict(row);original.pop('detection_motion_budget',None)
        source=temporary/'source.json'
        source.write_text(json.dumps({'newton_sha':manifest['newton_sha'],'scenes':[original]}))
        rejected=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_multirate_data.py'),
            '--manifest',str(source),'--output',str(temporary/'invalid')],capture_output=True,text=True)
        assert rejected.returncode!=0 and 'macro teacher endpoints exceed' in rejected.stderr
        source.write_text(json.dumps({'newton_sha':manifest['newton_sha'],'scenes':[row]}))
        accepted=subprocess.run([sys.executable,str(ROOT/'scripts/prepare_multirate_data.py'),
            '--manifest',str(source),'--output',str(temporary/'valid')],capture_output=True,text=True)
        assert accepted.returncode==0,accepted.stderr
        prepared=json.loads((temporary/'valid/data_manifest.json').read_text())['scenes'][0]
        assert prepared['detection_motion_budget']==.012
    def forbidden(*unused,**options):raise AssertionError('Runtime data checks must not execute native dynamics')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=forbidden;cls.step=forbidden
    scene,states=load_reference(row,wp,newton,'cpu')
    assert scene.geometry_oracle.motion_budget==row['detection_motion_budget']==.012
    assert scene.metadata['detection_motion_budget']==.012
    from wp_vbd.mixed_experiment import select
    chosen=select(states,slice(0,1));snapshot=scene.geometry_oracle.query(chosen)
    assert snapshot is not None
    assert audit_endpoint_motion(row['path'],row['multirate']['training']['windows'],.003)['unreachable_macro_hops']>0
    audit=audit_endpoint_motion(row['path'],row['multirate']['training']['windows'],.012)
    assert audit['unreachable_macro_hops']==0
    result={'status':'PASS','translation_and_surface_rotation_units':True,'quaternion_sign_and_scale':True,
            'static_entities_excluded':True,'unsupported_layout_not_certified':True,
            'metadata_and_real_query_envelopes_match':True,'actual_query_succeeds':True,
            'indexer_rejects_unreachable_macro_targets':True,
            'indexer_accepts_corrected_query_envelope':True,
            'all_training_macro_endpoints_reachable':True,'native_dynamics_calls':0,
            'macro_hops_checked':audit['macro_hops'],'maximum_teacher_surface_travel_m':audit['maximum_teacher_surface_travel_m']}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
