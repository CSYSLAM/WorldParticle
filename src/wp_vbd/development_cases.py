"""Explicit material/topology development cases without dropping legacy caps."""
from collections import OrderedDict


def case_key(row):
    value=row.get('development_case',row.get('demo') or 'coupled_replay')
    if not isinstance(value,str) or not value:raise ValueError('Nonempty development case key required')
    return value


def validation_rows(rows):
    """Legacy first-per-demo behavior, plus every explicitly declared window.

    Training rows and dataset IDs are preserved. The expanded rows are only
    views for development, not extra optimizer samples or additional demos.
    """
    legacy={case_key(row) for row in rows if 'development_windows' not in row}
    seen=set();result=[]
    for row in rows:
        if 'development_case' in row:raise ValueError('Case keys are derived from dataset and window, not supplied by training rows')
        if 'development_windows' not in row:
            key=case_key(row)
            if key not in seen:result.append(row);seen.add(key)
            continue
        windows=row['development_windows'];dataset=row.get('dataset_id')
        if not isinstance(dataset,str) or not dataset or not isinstance(windows,list) or not windows:
            raise ValueError('Dataset identity and explicit nonempty development windows required')
        if any(type(i) is not int or i<0 for i in windows) or len(set(windows))!=len(windows):
            raise ValueError('Distinct nonnegative integer development windows required')
        if row.get('validation_starts')!=windows:raise ValueError('Declared development and validation windows must agree')
        if set(windows)&set(row.get('qualified_train_starts',[])):raise ValueError('Training and development windows overlap')
        count=row.get('teacher_substeps')
        if type(count) is not int or any(i+2>count for i in windows):raise ValueError('Complete two-hop development windows required')
        for index in windows:
            key=f'{row.get("demo") or "coupled_replay"}::{dataset}@{index}'
            if key in seen or key in legacy:raise ValueError('Development case identity collision: '+key)
            result.append({**row,'development_case':key,'validation_starts':[index]});seen.add(key)
    return result


def comparison_metric(row,index,objects,policy):
    result={'demo':row.get('demo') or 'coupled_replay','index':index,'objects':objects,
        'baseline_physical_overrides':policy,
        'comparison_baseline':'old32_matched_physical_policy' if policy else 'old32_original_physical_policy'}
    if 'development_case' in row:result.update(development_case=case_key(row),dataset_id=row['dataset_id'])
    return result


def object_errors(scene,outputs,target):
    import torch
    from . import math3d as so3
    result=[]
    for obj in torch.unique(scene.object_id[scene.movable]).tolist():
        mask=(scene.object_id==obj)&scene.movable;values=[]
        for current in outputs:
            dx=(current.x[:,mask]-target.x[:,mask])/scene.length
            dq=so3.log(so3.multiply(current.q[:,mask],so3.conjugate(target.q[:,mask])))*scene.body[mask][None,:,None]
            values.append(float((1e4*dx.square()+100*dq.square()).mean()))
        result.append({'object':obj,'old32':values[0],'new648':values[1],'within_cap':values[1]<=values[0]+.2*max(values[0],.1)})
    return result


def equal_dataset_score(metrics):
    groups=OrderedDict()
    for row in metrics:
        value=sum(1+(o['new648']-o['old32'])/max(o['old32'],.1) for o in row['objects'])/len(row['objects'])
        groups.setdefault(row.get('dataset_id',case_key(row)),[]).append(value)
    if not groups:raise ValueError('Nonempty development metrics required')
    return sum(sum(group)/len(group) for group in groups.values())/len(groups)


def verify_case_metrics(rows,metrics):
    expected=validation_rows(rows);actual={case_key(row):row for row in metrics}
    if len(actual)!=len(metrics) or set(actual)!={case_key(row) for row in expected}:
        raise ValueError('Every declared development dataset/window must appear exactly once')
    for row in expected:
        if 'development_case' not in row:continue
        measured=actual[case_key(row)]
        if measured.get('dataset_id')!=row['dataset_id'] or measured.get('demo')!=row['demo'] or measured.get('index')!=row['validation_starts'][0]:
            raise ValueError('Development case must bind its actual dataset, demo and physical window')
    return True
