"""Predeclared per-object regression caps and balanced validation selection."""
import math
from . import math3d as so3

def object_scores(scene,predicted,target):
    result={}
    for index,group in enumerate(scene.metadata['groups']):
        mask=(scene.object_id==index)&scene.movable
        # Static group membership is built once before graph capture. This
        # report helper runs outside the captured inference/training callable.
        if not bool(mask.any()):continue
        position=(predicted.x[:,mask]-target.x[:,mask])/scene.length
        rotation=so3.log(so3.multiply(predicted.q[:,mask],so3.conjugate(target.q[:,mask])))*scene.body[mask][None,:,None]
        result[group['name']]=float((1e4*position.square()+100*rotation.square()).mean())
    return result

def capped_selection(metrics,reference,*,maximum_relative_increase=.05,absolute_floor=1e-4):
    """Average normalized object changes, with every object separately capped."""
    rows=metrics['scenarios'];original={r['dataset']:r for r in reference['scenarios']}
    assert set(original)=={r['dataset'] for r in rows}
    details=[];scenarios=[];eligible=metrics['accepted']
    for row in rows:
        old=original[row['dataset']]['objects'];new=row['objects'];assert set(old)==set(new)
        ratios=[]
        for name,value in new.items():
            base=old[name];ratio=1+(value-base)/max(base,absolute_floor)
            accepted=math.isfinite(value) and ratio<=1+maximum_relative_increase
            eligible=eligible and accepted;ratios.append(ratio)
            details.append({'dataset':row['dataset'],'object':name,'score':value,'reference':base,'normalized_change_ratio':ratio,'within_cap':accepted})
        scenarios.append(sum(ratios)/len(ratios))
    return sum(scenarios)/len(scenarios),{'eligible':bool(eligible),'maximum_relative_increase':maximum_relative_increase,'absolute_floor':absolute_floor,
        'aggregation':'mean per scenario of mean per-object normalized changes','objects':details}
