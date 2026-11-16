"""Isolate real Adam updates using the normal trainer's exact window functions."""
import copy
import gc
import hashlib
import json
import math
from pathlib import Path
import time
import traceback


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def normalize_unit_gradient(parameters,original_norm):
    """Balance sample gradients after the existing norm-one clipping.

    A 1e-6 floor bounds amplification; the inherited Adam moments remain intact.
    """
    factor=1./max(min(float(original_norm),1.),1e-6)
    for parameter in parameters:
        if parameter.grad is not None:parameter.grad.mul_(factor)
    return factor


def run_diagnostic(*,args,report,model,optimizer,parameters,rows,load,advance,release,budget):
    import torch
    from wp_vbd.mixed_experiment import select,loss,clip_gradient_norm_fp64
    from wp_vbd.training_window import transfer_window
    plan=json.loads(args.diagnostic_plan.read_text());path=Path(plan['report'])
    if path.exists():raise FileExistsError(path)
    if plan.get('modes')!=['clip','unit']:raise ValueError('Explicit isolated clip/unit comparison required')
    source=torch.load(args.resume_from,weights_only=True,map_location='cpu',mmap=True)
    source_adam_steps={key:float(value['step']) for key,value in source['optimizer']['state'].items()}
    original_shas={str(p):digest(p) for p in [args.resume_from,args.resume_report,args.manifest,args.diagnostic_plan]}
    named=[(name,p) for name,p in model.named_parameters() if p.requires_grad]
    before=torch.cat([source['model'][name].reshape(-1) for name,p in named])
    by_id={row['dataset_id']:row for row in rows};results=[];gradients=[];began=time.perf_counter()
    result={'status':'ISOLATED_ACTUAL_ADAM_DIRECTION_RUNNING','device':args.device,'source_step':source['step'],
        'main_training_optimizer_calls':0,'isolated_optimizer_calls':0,'results':results,'failure':None,
        'source_artifacts_sha256':original_shas,'config':model.config,'resource_budget':report['resource_budget'],
        'normal_training_windows_and_advance_functions_reused':True,'source_adam_moments_restored_for_every_trial':True,
        'scope':'Independent updates on copies of real weights and Adam. No main updates, new data admission, convergence or native rollout claim.'}
    path.parent.mkdir(parents=True,exist_ok=True)
    def persist():
        result['seconds']=time.perf_counter()-began
        temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');temporary.replace(path)
    def forward(case):
        row=by_id[case['dataset_id']];index=case['index'];validation=case.get('validation',False)
        scene,states=load(row,validation);physical=index
        if ((row['loader']=='mixed' and args.host_mixed_trajectories)
            or (row['loader'] in ('self','elastic_rod') and args.host_self_trajectories)
            or (row['loader']=='rod' and args.host_rod_trajectories)):
            states=transfer_window(states,index,2,args.device);physical=0
        current=select(states,slice(physical,physical+1));objective=current.x.new_zeros(());diagnostics=[]
        for offset in range(2):
            current,d=advance(model,row,scene,states,current,physical+offset)
            if not bool(d['accepted'].all()):raise RuntimeError('Diagnostic window rejected: '+str(case))
            objective=objective+(loss(scene,current,select(states,slice(physical+offset+1,physical+offset+2)))
                +d.get('contact_repair_supervision',0.))*(1/120/scene.metadata['dt'])**4/2
            diagnostics.append({k:d[k] for k in ['active_soft_cloth_self_rows','active_dynamic_particle_body_rows',
                'rod_rows_with_nonzero_wrench','finite_joint_rows_with_nonzero_wrench','contact_repair_calls'] if k in d})
        if not bool(torch.isfinite(objective)):raise RuntimeError('Nonfinite diagnostic loss')
        return objective,diagnostics
    persist()
    try:
        for case in plan['cases']:
            for mode in plan['modes']:
                model.load_state_dict(source['model']);optimizer.load_state_dict(copy.deepcopy(source['optimizer']))
                initial_steps=[float(optimizer.state[p]['step']) for p in parameters]
                for group in optimizer.param_groups:group['lr']=args.learning_rate
                model.train();optimizer.zero_grad(set_to_none=True)
                objective,physical_rows=forward(case);value=float(objective.detach());objective.backward()
                norm=clip_gradient_norm_fp64(parameters,1.)
                gradient=torch.cat([(p.grad.detach().cpu() if p.grad is not None else torch.zeros_like(p,device='cpu')).reshape(-1) for p in parameters])
                if mode=='clip':gradients.append(gradient.clone())
                factor=normalize_unit_gradient(parameters,norm) if mode=='unit' else 1.
                active=[p.grad is not None for p in parameters]
                optimizer.step();result['isolated_optimizer_calls']+=1
                if any(float(optimizer.state[p]['step'])!=old+int(used) for p,old,used in zip(parameters,initial_steps,active)):
                    raise RuntimeError('Adam must advance active parameters once and retain inactive clocks')
                if source_adam_steps!={key:float(value['step']) for key,value in source['optimizer']['state'].items()}:
                    raise RuntimeError('An isolated trial mutated its in-memory source Adam clocks')
                delta=torch.cat([p.detach().cpu().reshape(-1) for p in parameters])-before
                directional_derivative=float(torch.dot(gradient.double(),delta.double()))
                optimizer.zero_grad(set_to_none=True);del objective;gc.collect()
                model.eval()
                with torch.no_grad():after,_=forward(case)
                item={**case,'gradient_mode':mode,'gradient_norm_before_clip':float(norm),
                    'gradient_extra_scale':factor,'loss_before':value,'loss_after':float(after),
                    'in_memory_source_adam_unchanged':True,'active_adam_clocks_advanced_once':True,
                    'inactive_adam_clocks_unchanged':True,'parameter_elements_with_grad':sum(p.numel() for p,used in zip(parameters,active) if used),
                    'loss_change_percent':100*(float(after)/max(value,1e-30)-1),
                    'clipped_gradient_dot_actual_parameter_delta':directional_derivative,
                    'parameter_delta_l2':float(delta.double().norm()),'physical_rows':physical_rows}
                if args.device.startswith('cuda'):
                    torch.cuda.synchronize();item['process_gpu_mib']=budget.observe(force=True)
                results.append(item);persist();print(json.dumps({'isolated_adam_direction':item}),flush=True)
                del gradient,delta,after;release()
        norms=[float(g.double().norm()) for g in gradients]
        result['pairwise_gradient_cosines']=[[float(torch.dot(a.double(),b.double()))/max(norms[i]*norms[j],1e-30)
            for j,b in enumerate(gradients)] for i,a in enumerate(gradients)]
        result['pairwise_case_order']=plan['cases'];result['status']='ISOLATED_ACTUAL_ADAM_DIRECTION_COMPLETE'
    except Exception:
        result.update(status='ISOLATED_ACTUAL_ADAM_DIRECTION_FAILED',failure=traceback.format_exc());raise
    finally:
        result['source_artifacts_unchanged']=all(digest(p)==sha for p,sha in original_shas.items())
        result['gpu_resource_budget']=budget.report();persist()
