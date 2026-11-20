"""Paired long robot development windows, with causal student-owned history."""
import argparse
import functools
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ['CUDA_VISIBLE_DEVICES']=''


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--parent',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--hops',type=int,default=8)
    a=p.parse_args()
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.multirate_data import load_reference,advance
    from wp_vbd.multirate_window import inputs_for_span
    from wp_vbd.mixed_experiment import select
    from wp_vbd.multirate_objective import RolloutObjective
    torch.set_num_threads(1);wp.init();wp.set_device('cpu')
    def guard(original):
        @functools.wraps(original)
        def forbidden(*args,**kwargs):raise AssertionError('Native dynamics forbidden in development window evaluation')
        return forbidden
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=guard(cls.__init__);cls.step=guard(cls.step)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=saved['model_config'];models={}
    for label,path in [('parent_at_same_physical_budget',a.parent),('trained',a.checkpoint)]:
        state=torch.load(path,map_location='cpu',weights_only=True)
        model=MixedSolver(config).eval();model.load_state_dict(state['model'],strict=True);models[label]=model
    results=[]
    with torch.no_grad():
        for row in json.loads(a.manifest.read_text())['scenes']:
            if row['loader']!='surface':continue
            scene,states=load_reference(row,wp,newton,'cpu');dt=float(scene.metadata['dt'])
            scene.metadata.update(compact_numerical_diagnostics=True,elastic_shape_response=True,
                joint_tree_coarse_modes='scalar_tree_twelve_columns_v307')
            if scene.metadata.get('training_schema')=='authored_franka_substep_reference_v310':
                scene.metadata['separate_coarse_ray_step']=False
            objective=RolloutObjective(scene)
            plans=row['multirate']['development']['windows']
            for index in sorted(set((0,len(plans)//2,len(plans)-1))):
                plan=plans[index];start=plan['start'];stride=min(4,sum(plan['strides'])//a.hops)
                if stride<1:continue
                item={'dataset':row['dataset_id'],'teacher_start':start,'dt':stride*dt,'hops':a.hops,'variants':[]}
                for name,model in models.items():
                    current=select(states,slice(start,start+1));values=[];details=[]
                    for hop in range(a.hops):
                        j=start+hop*stride;actions=inputs_for_span(states,j,stride,durations=[dt]*stride)
                        current,info=advance(model,row,scene,current,actions,dt*stride,wp)
                        if not bool(info['accepted'].all()):
                            details.append({'rejected_at_hop':hop});break
                        target=select(states,slice(j+stride,j+stride+1))
                        value,terms,metrics=objective(current,target,info)
                        values.append(float(value));details.append({k:float(v) for k,v in metrics.items()})
                    item['variants'].append({'name':name,'mean_loss':sum(values)/a.hops if len(values)==a.hops else None,
                        'final_metrics':details[-1] if details else {},'completed_hops':len(values)})
                results.append(item);print(json.dumps(item),flush=True)
    means={name:sum(v['mean_loss'] for row in results for v in row['variants'] if v['name']==name and v['mean_loss'] is not None)/
        max(1,sum(v['name']==name and v['mean_loss'] is not None for row in results for v in row['variants'])) for name in models}
    report={'status':'PAIRED_ROBOT_DEVELOPMENT_WINDOW_REVIEW','device':'cpu','means':means,'windows':results,
        'native_dynamics_calls':0,'optimizer_updates':0,'teacher_used_only_for_initial_state_and_supervision':True,
        'scope':'Disjoint development intervals of existing robot/hand teacher trajectories, not independent controls/assets or full autonomous-task acceptance.'}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
