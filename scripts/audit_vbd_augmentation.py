"""Check independent trajectory counts, integrity, splits, and native provenance."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    data=json.loads(a.collection.read_text());assert data['status']=='COMPLETE'
    counts=Counter();seeds=set();specs=set();traces=set();splits=Counter();calls=0
    for row in data['scenes']:
        path=Path(row['path']);assert hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256']
        with np.load(path,allow_pickle=False) as archive:
            meta=json.loads(str(archive['metadata']));x=archive['x'];q=archive['q']
            assert len(x)==row['states']==91 and len(q)==len(x)
            for key in archive.files:
                if key!='metadata':assert np.isfinite(archive[key]).all(),(path,key)
            assert np.max(np.abs(np.linalg.norm(q,axis=-1)-1))<1e-8
            assert len(archive['input_joint_target'])==len(x)
            fingerprint=hashlib.sha256(x.tobytes()+q.tobytes()).hexdigest()
        assert fingerprint not in traces;traces.add(fingerprint)
        assert row['seed'] not in seeds;seeds.add(row['seed'])
        shape=hashlib.sha256(json.dumps(meta['specification'],sort_keys=True).encode()).hexdigest()
        assert shape not in specs;specs.add(shape)
        assert meta['native_dt']==1/1800 and meta['dt']==1/180
        assert meta['record_stride_fine_steps']==10 and row['native_vbd_steps']==910
        assert not meta['configuration']['integrate_with_external_rigid_solver']
        assert meta['qualification']['status']=='PASS'
        counts[row['family']]+=1;splits[row['split']]+=1;calls+=row['native_vbd_steps']
    assert len(counts)==10 and all(value==33 for value in counts.values())
    assert splits=={'training':300,'development':30}
    result={'status':'PASS','independent_trajectories':len(traces),'unique_seeds':len(seeds),
            'unique_asset_parameter_control_specs':len(specs),'families':dict(counts),'splits':dict(splits),
            'native_vbd_physical_steps':calls,'total_recorded_physical_seconds':len(traces)*.5,
            'each_trajectory_seconds':.5,'cross_solver_coupling_used':False,
            'native_reference_integrity_verified':True,'labels_interpolated':False,
            'scope':'Independent native-VBD trajectories, not a count of distinct topology templates or reused windows. Geometry/volume screening is discrete, not a continuous collision proof.'}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
