"""Measure fixed-h reference refinement without certifying physical accuracy."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def load(path):
    with np.load(path,allow_pickle=False) as values:
        return {key:values[key].copy() for key in values.files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lower',type=Path,required=True)
    parser.add_argument('--higher',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    low,high=load(args.lower),load(args.higher)
    metadata=[json.loads(str(v['metadata'])) for v in (low,high)]
    assert metadata[0]['official_demo']==metadata[1]['official_demo']
    assert metadata[0]['dt']==metadata[1]['dt'] and low['x'].shape==high['x'].shape
    shared=[key for key in low if key not in ('metadata','x','q','v','omega') and not key.startswith('history__')]
    identical={key:bool(np.array_equal(low[key],high[key])) for key in shared}
    assert all(identical.values()),{key:value for key,value in identical.items() if not value}
    mobile=low['movable'];length=float(low['reference_length'])
    position=np.linalg.norm(low['x'][:,mobile].astype(np.float64)-high['x'][:,mobile],axis=-1)
    q1,q2=low['q'][:,mobile].astype(np.float64),high['q'][:,mobile].astype(np.float64)
    dot=np.abs((q1*q2).sum(-1)/(np.linalg.norm(q1,axis=-1)*np.linalg.norm(q2,axis=-1)))
    angle=2*np.arccos(dot.clip(0,1))
    errors={}
    for key in ('v','omega','history__dual','history__kappa','history__increment','history__stress'):
        difference=low[key].astype(np.float64)-high[key]
        errors[key]={'maximum_absolute_difference':float(np.abs(difference).max()),'rms_difference':float(np.sqrt(np.mean(difference**2)))}
    result={
        'status':'REFERENCE_REFINEMENT_MEASURED','demo':metadata[0]['official_demo'],
        'duration_seconds':(len(low['x'])-1)*metadata[0]['dt'],'physical_steps':len(low['x'])-1,
        'same_h':metadata[0]['dt'],'identical_assets_materials_and_current_inputs':identical,
        'maximum_position_difference_m':float(position.max()),'maximum_position_difference_over_L':float(position.max()/length),
        'rms_position_difference_m':float(np.sqrt(np.mean(position**2))),
        'maximum_orientation_difference_rad':float(angle.max()),'rms_orientation_difference_rad':float(np.sqrt(np.mean(angle**2))),
        'refinement_limits':{'position_over_L':.002,'orientation_rad':.002},
        'within_refinement_limits':bool(position.max()/length<.002 and angle.max()<.002),
        'other_channels':errors,
        'source_files':{key:{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()} for key,path in (('lower',args.lower),('higher',args.higher))},
        'training_admission_granted':False,
        'scope':'Full-duration fixed-h iteration refinement only. Agreement between FP32 solver budgets does not establish continuum/physical accuracy or independent force consistency. No audit episode enters training through this script.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
