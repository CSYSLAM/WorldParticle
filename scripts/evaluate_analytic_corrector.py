"""Paired held-out window ablations at equal analytic/correction round budgets."""
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
    p.add_argument('--collection',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--initial-checkpoint',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    p.add_argument('--analytic-elastic-postsolve',action=argparse.BooleanOptionalAction,default=None)
    p.add_argument('--analytic-predictor-rounds',type=int,choices=(1,2,3))
    a=p.parse_args()
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.augmentation_data import FAMILIES,load_reference,advance
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.mixed_experiment import select
    from wp_vbd.multirate_objective import RolloutObjective
    from wp_vbd.multirate_window import inputs_for_span
    torch.set_num_threads(1);wp.init();wp.set_device('cpu')
    def block(original):
        @functools.wraps(original)
        def forbidden(*args,**kwargs):raise AssertionError('Native dynamics forbidden in window comparison')
        return forbidden
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=block(cls.__init__);cls.step=block(cls.step)
    saved=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    if a.analytic_elastic_postsolve is not None:saved['model_config']['analytic_elastic_postsolve']=a.analytic_elastic_postsolve
    if a.analytic_predictor_rounds is not None:saved['model_config']['analytic_predictor_rounds']=a.analytic_predictor_rounds
    model=MixedSolver(saved['model_config']).eval();model.load_state_dict(saved['model'],strict=True)
    initial=MixedSolver(saved['model_config']).eval()
    parent=torch.load(a.initial_checkpoint,map_location='cpu',weights_only=True)
    shared_parent=bool((parent.get('model_config') or parent['config']['model']).get('shared_terminal_velocity',False))
    retired=('elastic_shape_head.','contact_tangent_head.')+(() if shared_parent else ('terminal_velocity.',))
    state={k:v for k,v in parent['model'].items() if not k.startswith(retired)}
    missing,extra=initial.load_state_dict(state,strict=False)
    expected=set() if shared_parent else {'terminal_velocity.shared_readout.weight','terminal_velocity.shared_readout.bias'}
    assert set(missing)==expected and not extra
    collection=json.loads(a.collection.read_text());results=[]
    with torch.no_grad():
        for family in FAMILIES:
            row=next(r for r in collection['scenes'] if r['split']=='development' and r['family']==family)
            scene,states=load_reference(row['path'],wp,newton,'cpu')
            objective=RolloutObjective(scene);stride=(1,2,3)[row['seed']%3]
            start=min(len(states['x'])-2*stride-1,len(states['x'])//2)
            comparisons=[]
            for label,network,learned in (('analytic_only_same_two_rounds',model,False),
                                        ('initial_shared_corrector',initial,True),('trained_shared_corrector',model,True)):
                current=select(states,slice(start,start+1));losses=[];details=[]
                for hop in range(2):
                    j=start+hop*stride;dt=stride/180
                    actions=inputs_for_span(states,j,stride,durations=[1/180]*stride)
                    current,info=advance(network,scene,current,actions,dt,wp,learned=learned)
                    if not bool(info['accepted'].all()):
                        details.append({'rejected':True});break
                    target=select(states,slice(j+stride,j+stride+1));value,terms,metrics=objective(current,target,info)
                    losses.append(float(value));details.append({'terms':{k:float(v) for k,v in terms.items()},
                        'metrics':{k:float(v) for k,v in metrics.items()}})
                comparisons.append({'variant':label,'loss':sum(losses)/2 if len(losses)==2 else None,'hops':details})
            results.append({'dataset':row['dataset_id'],'family':family,'start':start,'dt':stride/180,'comparisons':comparisons})
            print(family,[(c['variant'],c['loss']) for c in comparisons],flush=True)
    means={name:sum(c['loss'] for r in results for c in r['comparisons'] if c['variant']==name and c['loss'] is not None)/
           sum(c['variant']==name and c['loss'] is not None for r in results for c in r['comparisons'])
           for name in ('analytic_only_same_two_rounds','initial_shared_corrector','trained_shared_corrector')}
    rounds=int(saved['model_config'].get('analytic_predictor_rounds',0))+int(saved['model_config'].get('unroll',1))
    report={'status':'PAIRED_HELD_OUT_WINDOW_REVIEW','device':'cpu','cases':results,'mean_loss':means,
            'checkpoint_step':saved['step'],'checkpoint_path':str(a.checkpoint.resolve()),
            'analytic_elastic_postsolve':bool(saved['model_config'].get('analytic_elastic_postsolve',False)),
            'physical_response_rounds_per_hop':rounds,
            'native_dynamics_calls':0,'optimizer_updates':0,
            'legacy_variant_name_note':'analytic_only_same_two_rounds is retained as an identifier; actual round count is physical_response_rounds_per_hop',
            'scope':f'Fixed independent teacher-state initial windows, two autonomous hops each, equal {rounds} physical response rounds per hop. CPU arithmetic; not a full-scene or GPU-speed benchmark.'}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
