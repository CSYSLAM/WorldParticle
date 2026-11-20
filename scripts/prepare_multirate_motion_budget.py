"""Write a new manifest with explicit, reachable macro collision envelopes."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--current-budget', type=float, default=.003)
    parser.add_argument('--motion-budget', type=float, default=.012,
                        help='Explicit new query envelope; increase candidates and retain all runtime checks')
    args = parser.parse_args()
    from wp_vbd.multirate_motion import audit_endpoint_motion
    if args.output.exists() or args.report.exists():
        raise FileExistsError('Preserve existing manifests and audits; choose new output paths')
    data = json.loads(args.manifest.read_text()); result = copy.deepcopy(data)
    chosen = set(args.datasets); found = set(); reports = []
    for row in result['scenes']:
        if row['dataset_id'] not in chosen: continue
        if row['loader'] != 'surface':
            raise ValueError('This correction requires the surface/joint query adapter')
        found.add(row['dataset_id']); before = {}; after = {}
        for split, path_key in (('training','path'), ('development','validation_path')):
            path = Path(row.get(path_key,row['path']))
            if not path.is_absolute(): path = ROOT/path
            hash_key = 'validation_sha256' if path_key=='validation_path' and 'validation_path' in row else 'sha256'
            with path.open('rb') as stream: actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != row[hash_key]: raise ValueError('Reference checksum changed: '+str(path))
            windows = row['multirate'][split]['windows']
            before[split] = audit_endpoint_motion(path, windows, row.get('detection_motion_budget',args.current_budget))
            after[split] = audit_endpoint_motion(path, windows, args.motion_budget)
            if not after[split]['supported'] or after[split]['unreachable_macro_hops']:
                raise ValueError('Requested envelope does not cover teacher endpoints for '+row['dataset_id'])
        row['detection_motion_budget'] = args.motion_budget
        row['macro_motion_envelope_audit'] = after
        reports.append({'dataset':row['dataset_id'],'before':before,'after':after})
    if found != chosen: raise ValueError('Unknown datasets: '+', '.join(sorted(chosen-found)))
    with args.manifest.open('rb') as stream: parent_hash = hashlib.file_digest(stream,'sha256').hexdigest()
    result['motion_budget_parent'] = {'path':str(args.manifest.resolve()),'sha256':parent_hash}
    report = {'datasets':reports,'w1_used':False,'optimizer_updates':0,
              'scope':'Candidate-envelope correction. No material, control, time grid or iteration budget changes.'}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.report.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps([{'dataset':r['dataset'],
        'previous_unreachable':{k:v['unreachable_macro_hops'] for k,v in r['before'].items()},
        'corrected_unreachable':{k:v['unreachable_macro_hops'] for k,v in r['after'].items()}} for r in reports],indent=2))


if __name__=='__main__': main()
