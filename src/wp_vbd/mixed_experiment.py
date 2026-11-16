"""Verification, shared training and independent mixed rollouts."""

import hashlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import torch

from . import math3d as so3
from .data import ROOT,State
from .mixed_state import MixedScene,tensor_scene
from .mixed_solver import MixedSolver,apply_update,geometry_scale
from .mixed_physics import joint_coordinates,joint_system,mass_whitening,predictor,residual_system
from .runtime import CapturedStepper


def load_sample(path,device='cuda:0'):
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        dynamic=('x','q','v','omega','input_x','input_q','input_force','input_torque')
        states={key:torch.as_tensor(archive[key].copy(),device=device) for key in archive.files if key in dynamic or key.startswith('history__')}
        static={key:archive[key].copy() for key in archive.files if key not in (*states,'metadata')}
    return tensor_scene(static,metadata,device),states


def select(states,indices):
    history={key.removeprefix('history__'):value[indices] for key,value in states.items() if key.startswith('history__')}
    return State(*(states[key][indices] for key in ('x','q','v','omega')),history or None)


def external_inputs(states,indices):
    return {key:states['input_'+key][indices] for key in ('force','torque') if 'input_'+key in states}


def scene_config(config,scene):
    """The physical step belongs to each authored scene, not the batch."""
    return {**config,'dt':float(scene.metadata.get('dt',config['dt']))}


def prescribe(scene,state,position,rotation):
    ids=scene.kinematic_ids
    return State(state.x.index_copy(1,ids,position),state.q.index_copy(1,ids,rotation),state.v,state.omega,state.history)


def controlled_start(scene,current,states,step):
    if 'input_x' not in states:return current
    return prescribe(scene,current,states['input_x'][step:step+1],states['input_q'][step:step+1])


def configure(config):
    torch.set_num_threads(2);torch.manual_seed(config['seed']);torch.use_deterministic_algorithms(True)


def clip_gradient_norm_fp64(parameters,maximum):
    gradients=[p.grad for p in parameters if p.grad is not None]
    norm=torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(g.double()) for g in gradients]))
    if not bool(torch.isfinite(norm)):raise RuntimeError('Nonfinite gradient or FP64 gradient norm')
    scale=(maximum/(norm+1e-12)).clamp_max(1.)
    for g in gradients:g.mul_(scale.to(g.dtype))
    return norm


def verify(config):
    configure(config);sys.modules['newton']=None
    scene,states=load_sample(ROOT/config['data_dir']/'mixed_00.npz')
    model=MixedSolver(config['model']).cuda()
    current=select(states,torch.tensor([0,40,120],device='cuda'))
    predicted,diagnostic=model(scene,current,config['dt'])
    objective=predicted.x.square().mean()+predicted.q.square().mean();objective.backward()
    finite=all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
    assert finite,'Nonfinite new-model gradient'
    assert bool(diagnostic['accepted'].all()),diagnostic
    assert torch.equal(predicted.x[:,~scene.movable],current.x[:,~scene.movable])
    # Independent scalar-energy differentiation checks the entire coupled
    # joint/rod generalized residual, not just its local block construction.
    values={key:value.double() if value.is_floating_point() else value for key,value in scene.values.items()}
    double=SimpleNamespace(**values,n=scene.n,rod_rest=scene.rod_rest.double())
    x=current.x[:1].double()+.001*torch.randn_like(current.x[:1].double())
    q=so3.normalize(so3.multiply(so3.exp(.05*torch.randn_like(x)),current.q[:1].double()))
    state=State(x,q,current.v[:1].double(),current.omega[:1].double())
    gradient,blocks,_,_=joint_system(double,state)
    direction=torch.randn(1,scene.n,6,device='cuda',dtype=torch.float64)
    def energy(alpha):
        xx=x+alpha*direction[...,:3];qq=so3.multiply(so3.exp(alpha*direction[...,3:]),q)
        c=joint_coordinates(double,xx[:,double.joints],qq[:,double.joints])
        return .5*(c.square()*double.joint_stiffness).sum()
    epsilon=1e-6;fd=(energy(epsilon)-energy(-epsilon))/(2*epsilon);exact=(gradient*direction).sum()
    derivative=float((fd-exact).abs()/exact.abs().clamp_min(1))
    assert derivative<1e-6,derivative
    balance=float(gradient[...,:3].sum(1).abs().max())
    moment=float((torch.linalg.cross(x,gradient[...,:3])+gradient[...,3:]).sum(1).abs().max())
    assert balance<1e-8 and moment<1e-8,(balance,moment)
    assert float(blocks[...,3:,:3].abs().max())>0
    start=select(states,slice(0,1));proposal=.1*torch.randn(1,scene.n,6,device='cuda')*scene.dof
    proposal[...,2]-=.3*scene.movable
    scale,valid=geometry_scale(scene,start,proposal)
    minimum=1.
    for alpha in torch.linspace(0,1,33):
        candidate=apply_update(scene,start,proposal,scale*alpha)
        point=candidate.x[:,scene.site_owner]+so3.rotate(candidate.q[:,scene.site_owner],scene.site_local)
        minimum=min(minimum,float((point[...,2]-scene.site_radius-scene.plane_height).min()))
    assert bool(valid.all()) and minimum>=-1e-7,(minimum,valid)
    tx,tq=predictor(scene,start,config['dt']);r,d,extra=residual_system(scene,start,start,tx,tq,config['dt']);w=mass_whitening(scene,start,config['dt'])
    from .mixed_solver import relation_inputs
    jac,relation=relation_inputs(scene,start,start,extra,config['dt'],torch.ones_like(start.x[...,0]))
    root=model.response(scene.layout,torch.zeros_like(r),w,torch.eye(6,device='cuda').expand_as(d),jac,torch.randn(1,scene.n,24,device='cuda'),relation)
    assert torch.count_nonzero(root)==0
    report={'status':'PASS','architecture':model.architecture_id,'finite_forward_backward':finite,'fixed_dof_exact':True,'joint_rod_energy_directional_derivative_error':derivative,'joint_rod_internal_force_balance_error':balance,'joint_rod_internal_angular_balance_error':moment,'coupled_6D_cross_block_max':float(blocks[...,3:,:3].abs().max()),'sampled_rotational_plane_path_minimum_clearance':minimum,'zero_residual_update':float(root.abs().max()),'newton_import_blocked':True,'scope':'Finite elastic, primitive plane contract only; no ALM-history, general collision or whole-stage acceptance.'}
    path=ROOT/config['report_dir']/'verification.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return report


def loss(scene,predicted,target):
    position=(predicted.x-target.x)/scene.length
    rotation=so3.log(so3.multiply(predicted.q,so3.conjugate(target.q)))*scene.body[None,:,None]
    return ((1e4*position.square()+100*rotation.square())*scene.movable[None,:,None]).mean()


def cyclic_shape_loss(scene,predicted,target):
    """Free-ring size and nonplanarity, independent of post-buckling phase."""
    values=[];details=[]
    for plan in scene.rod_cycle_plans:
        def spectrum(x):
            points=x[:,plan.ids].double();points=points-points.mean(1,keepdim=True)
            covariance=points.transpose(-1,-2)@points/len(plan.ids)
            return torch.linalg.eigvalsh(covariance).clamp_min(0).sqrt()
        a,b=spectrum(predicted.x),spectrum(target.x)
        values.append(1e4*((a-b)/scene.length).square().mean())
        details.append({'component_root':plan.root,'predicted_principal_rms_m':a[0].tolist(),'reference_principal_rms_m':b[0].tolist()})
    return float(torch.stack(values).mean()),details


def rollout(model,scene,states,config,*,captured=True,learned=True,steps=None):
    if scene.metadata.get('data_layout')=='teacher_queried_windows_v1':raise ValueError('Disjoint teacher windows cannot be concatenated into an autonomous rollout')
    current=select(states,slice(0,1));sequence=[current.detach()];accepted=True
    graph=CapturedStepper(model,scene,current,config['dt']) if captured and learned else None
    with torch.no_grad():
        for step in range(config['steps'] if steps is None else steps):
            current=controlled_start(scene,current,states,step)
            inputs=external_inputs(states,slice(step,step+1))
            predicted,diagnostic=graph.step(current,**inputs) if graph is not None else model(scene,current,config['dt'],learned=learned,**inputs)
            if not bool(diagnostic['accepted'].all()):accepted=False;break
            current=predicted.detach();sequence.append(current)
    return sequence,accepted


@torch.no_grad()
def validation(model,scene,states,config):
    config=scene_config(config,scene)
    validation_steps=config['training'].get('validation_steps_by_demo',{}).get(scene.metadata.get('official_demo'),config['training']['validation_steps'])
    model.eval();indices=torch.linspace(0,len(states['x'])-2,min(12,len(states['x'])-1),device='cuda').long()
    if scene.metadata.get('data_layout')=='teacher_queried_windows_v1':
        if not scene.metadata['qualified_windows']:raise ValueError('Unqualified query labels')
        valid=scene.qualified_step_indices
        indices=valid[torch.linspace(0,len(valid)-1,min(12,len(valid)),device=scene.device).long()]
        if config['training'].get('validation_batch_size')!=1:raise ValueError('Queried-window validation requires the explicit window path')
    if config['training'].get('validation_batch_size')==1:
        graph=CapturedStepper(model,scene,select(states,slice(0,1)),config['dt'])
        single=0.
        for i in indices.tolist():
            one=controlled_start(scene,select(states,slice(i,i+1)),states,i)
            single+=float(loss(scene,graph.step(one,**external_inputs(states,slice(i,i+1)))[0],select(states,slice(i+1,i+2))))/len(indices)
        windows=scene.metadata.get('validation_windows') or config['training'].get('validation_windows_by_demo',{}).get(scene.metadata.get('official_demo'))
        if windows:
            values=[];details=[];object_values=[]
            for first in windows['starts']:
                current=select(states,slice(first,first+1));ok=True
                for step in range(first,min(first+windows['steps'],len(states['x'])-1)):
                    current=controlled_start(scene,current,states,step)
                    current,diagnostic=graph.step(current,**external_inputs(states,slice(step,step+1)))
                    if not bool(diagnostic['accepted'].all()):ok=False;break
                value=float(loss(scene,current,select(states,slice(step+1,step+2)))) if ok else 1e6
                values.append(value);details.append({'start':first,'steps':step-first+1,'accepted':ok,'endpoint_loss':value})
                if config['training'].get('validation_selection'):
                    from .validation_policy import object_scores
                    object_values.append(object_scores(scene,current,select(states,slice(step+1,step+2))) if ok else {})
            model.train()
            score=sum(values)/len(values)
            report={'one_step_loss':single,'rollout_loss':score,'accepted':all(v['accepted'] for v in details),'validation_mode':'predeclared_teacher_initialized_windows','windows':details}
            if object_values:
                names=sorted(g['name'] for i,g in enumerate(scene.metadata['groups']) if bool(((scene.object_id==i)&scene.movable).any()))
                report['objects']={name:sum(row.get(name,1e6) for row in object_values)/len(object_values) for name in names}
            return single+score,report
        current=select(states,slice(0,1));ok=True
        count=min(validation_steps,len(states['x'])-1)
        for step in range(count):
            current=controlled_start(scene,current,states,step)
            current,diagnostic=graph.step(current,**external_inputs(states,slice(step,step+1)))
            if not bool(diagnostic['accepted'].all()):ok=False;break
        target=select(states,slice(step+1,step+2))
        pose_error=float(loss(scene,current,target)) if ok else 1e6
        error=pose_error;extra={}
        if ok and config['training'].get('validation_cyclic_shape') and getattr(scene,'rod_cycle_plans',[]):
            error,shapes=cyclic_shape_loss(scene,current,target)
            extra={'rollout_pose_loss':pose_error,'rollout_metric':'free_ring_covariance_spectrum','ring_shapes':shapes}
        report={'one_step_loss':single,'rollout_loss':error,'accepted':ok,'rollout_steps':step+1,**extra}
        if config['training'].get('validation_selection'):
            from .validation_policy import object_scores
            report['objects']=object_scores(scene,current,target)
            if not ok:report['objects']={key:1e6 for key in report['objects']}
        model.train();return single+error,report
    predicted,d=model(scene,select(states,indices),config['dt'],**external_inputs(states,indices))
    single=float(loss(scene,predicted,select(states,indices+1)))
    seq,ok=rollout(model,scene,states,config,steps=min(validation_steps,len(states['x'])-1))
    error=float(loss(scene,seq[-1],select(states,slice(len(seq)-1,len(seq))))) if ok else 1e6
    model.train()
    return single+error,{'one_step_loss':single,'rollout_loss':error,'accepted':ok,'rollout_steps':len(seq)-1}


def copy_static(destination,source):
    if destination is source:raise ValueError('A mutable training carrier cannot be a source scene')
    if destination.material_damping_flags!=source.material_damping_flags:raise ValueError('Material damping paths differ between training graphs')
    if destination.metadata.get('contract')!=source.metadata.get('contract') or destination.metadata['box_sites']!=source.metadata['box_sites']:raise ValueError('Physical contact contracts differ between training graphs')
    if not torch.equal(destination.layout.incidence,source.layout.incidence) or not torch.equal(destination.layout.pool,source.layout.pool):raise ValueError('Training graph requires matching explicit topology')
    if not torch.equal(destination.layout.relation_kind,source.layout.relation_kind):raise ValueError('Relation laws differ between training graphs')
    for key in ('dof','body','kind','object_id','site_owner'):
        if not torch.equal(getattr(destination,key),getattr(source,key)):raise ValueError(f'Training graph layout differs: {key}')
    for key,value in destination.values.items():
        value.copy_(source.values[key])
        objective=getattr(destination.objective_scene,key)
        if objective is not value:objective.copy_(source.values[key])
    destination.inverse_inertia_sqrt.copy_(source.inverse_inertia_sqrt)
    destination.rod_rest.copy_(source.rod_rest);destination.objective_scene.rod_rest.copy_(source.rod_rest)
    destination.origins.copy_(source.origins)


class TrainingGraph:
    def __init__(self,model,scene,states,config,optimizer):
        self.config=config;self.scene=scene;self.model=model
        batch=config['training']['batch_size'];horizon=config['training']['horizon']
        initial_indices=torch.arange(batch,device='cuda')
        if scene.metadata.get('data_layout')=='teacher_queried_windows_v1':
            if horizon>scene.metadata['qualified_window_horizon']:raise ValueError('Training horizon exceeds qualified teacher windows')
            initial_indices=scene.qualified_starts[initial_indices%len(scene.qualified_starts)]
        self.input=select(states,initial_indices)
        self.targets=[select(states,initial_indices+i+1) for i in range(horizon)]
        self.drives=[(states['input_x'][initial_indices+i].clone(),states['input_q'][initial_indices+i].clone()) for i in range(horizon)] if 'input_x' in states else None
        self.external=[{key:value.clone() for key,value in external_inputs(states,initial_indices+i).items()} for i in range(horizon)]
        self.optimizer=optimizer
        def forward_backward():
            optimizer.zero_grad(set_to_none=False)
            current=self.input;total=current.x.new_zeros(());accepted=torch.ones(batch,device='cuda',dtype=torch.bool)
            for offset,target in enumerate(self.targets):
                if self.drives is not None:current=prescribe(scene,current,*self.drives[offset])
                current,d=model(scene,current,config['dt'],**self.external[offset])
                time_scale=(config.get('loss_reference_dt',config['dt'])/config['dt'])**4 if config['training'].get('normalize_acceleration_loss',False) else 1.
                total=total+loss(scene,current,target)*time_scale/horizon;accepted=accepted&d['accepted']
            total.backward();return total,accepted
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):forward_backward()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):self.loss,self.accepted=forward_backward()
        torch.cuda.current_stream().wait_stream(stream)

    def run(self,scene,states,indices):
        if scene.metadata.get('data_layout')=='teacher_queried_windows_v1' and not bool(torch.isin(indices,scene.qualified_starts).all()):raise ValueError('Training indices cross a teacher-query window boundary')
        copy_static(self.scene,scene)
        self.input.copy_(select(states,indices))
        for offset,target in enumerate(self.targets):
            target.copy_(select(states,indices+offset+1))
            if self.drives is not None:
                self.drives[offset][0].copy_(states['input_x'][indices+offset]);self.drives[offset][1].copy_(states['input_q'][indices+offset])
            for key,value in self.external[offset].items():value.copy_(states['input_'+key][indices+offset])
        self.graph.replay()
        return self.loss,self.accepted


def training_layout_signature(scene):
    digest=hashlib.sha256()
    digest.update(str([(k,str(v.dtype),tuple(v.shape)) for k,v in sorted(scene.values.items())]).encode())
    digest.update(str(scene.length).encode())
    digest.update(str(scene.tet_floor).encode())
    digest.update(str(scene.metadata.get('dt')).encode())
    digest.update(str(scene.material_damping_flags).encode())
    digest.update(str([plan.signature for plan in scene.bending_plans]).encode())
    digest.update(json.dumps({'contract':scene.metadata.get('contract'),'box_sites':scene.metadata['box_sites']},sort_keys=True).encode())
    if scene.metadata.get('history_contract'):
        digest.update(str((scene.metadata['history_contract'],scene.metadata['joint_alpha'],scene.metadata['joint_gamma'],scene.rod_enabled_dahl)).encode())
    tensors=[scene.layout.incidence,scene.layout.relation_kind,scene.layout.pool,scene.layout.pool_component,scene.dof,scene.kind,scene.object_id,scene.site_owner,scene.rest_x,scene.rest_q,scene.body_com,scene.anchor_p,scene.anchor_c,scene.anchor_qp,scene.anchor_qc]
    for value in tensors:digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def static_scene_fingerprint(scene):
    digest=hashlib.sha256()
    for key,value in sorted(scene.values.items()):
        digest.update(key.encode());digest.update(value.cpu().numpy().tobytes())
    return digest.hexdigest()


def train(config):
    configure(config);sys.modules['newton']=None
    report_dir=ROOT/config['report_dir'];manifest_path=report_dir/'data_manifest.json';manifest=json.loads(manifest_path.read_text())
    if manifest['status']!='PASS' or json.loads((report_dir/'verification.json').read_text())['status']!='PASS':raise RuntimeError('Passing teacher and physics verification required')
    samples=[];validations=[]
    for row in manifest['scenes']:
        path=ROOT/row['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Dataset changed')
        if row['split']=='test':continue
        sample=load_sample(path)
        sample[0].dataset_label=row.get('dataset_id',str(row['index']))
        if row['split']=='train':samples.append(sample)
        else:validations.append(sample)
    if not samples or not validations:raise ValueError('Training and validation scenes are required')
    model=MixedSolver(config['model']).cuda().train();cfg=config['training']
    base_step=0
    if config.get('initialize_from') and config.get('resume_from'):raise ValueError('Choose initialization or resume, not both')
    if config.get('initialize_from'):
        base=torch.load(config['initialize_from'],weights_only=True)
        incompatible=model.load_state_dict(base['model'],strict=False)
        expected=[name for name in model.state_dict() if name not in base['model']]
        if any(not name.startswith(('coarse_gain.','response.material_film.')) for name in expected):raise ValueError('Initialization cannot invent untrained missing modules')
        if sorted(incompatible.missing_keys)!=sorted(expected) or incompatible.unexpected_keys:
            raise ValueError(f'Initialization keys differ beyond the declared coarse head: {incompatible}')
    if config.get('resume_from'):
        base=torch.load(config['resume_from'],weights_only=True)
        if base['architecture']!=model.architecture_id:raise ValueError('Resume architecture differs')
        model.load_state_dict(base['model']);base_step=int(base['step'])
    if config.get('initialize_reset_coarse_head'):
        if config.get('resume_from') or model.coarse_gain is None:raise ValueError('Coarse head reset is initialization only')
        torch.nn.init.zeros_(model.coarse_gain[-1].weight);torch.nn.init.zeros_(model.coarse_gain[-1].bias)
    prefixes=cfg.get('trainable_prefixes')
    if prefixes:
        for name,p in model.named_parameters():p.requires_grad_(any(name.startswith(prefix) for prefix in prefixes))
        if not any(p.requires_grad for p in model.parameters()):raise ValueError('No trainable parameters match the declared curriculum')
    frozen_parameters={name:p.detach().clone() for name,p in model.named_parameters() if not p.requires_grad}
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=1e-4)
    directory=ROOT/config['checkpoint_dir'];directory.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();history=[];best=float('inf')
    selected_sample_counts={scene.dataset_label:0 for scene,_ in samples}
    def save(step,name):
        path=directory/name;temporary=path.with_suffix('.tmp')
        torch.save({'architecture':model.architecture_id,'model':model.state_dict(),'config':config,'step':base_step+step,'data_manifest_sha256':hashlib.sha256(manifest_path.read_bytes()).hexdigest()},temporary)
        temporary.replace(path)
    save(0,'refinement_initial.pt' if base_step else 'initial.pt')
    training_report='training_refinement.json' if base_step else 'training.json'
    def validate_all():
        rows=[]
        for scene,states in validations:
            value,detail=validation(model,scene,states,config)
            print('validation scene',scene.dataset_label,value,'accepted',detail['accepted'],flush=True)
            rows.append({'dataset':scene.dataset_label,'score':value,**detail})
        if len(rows)==1:return rows[0]['score'],{k:v for k,v in rows[0].items() if k not in ('dataset','score')}
        return sum(row['score'] for row in rows)/len(rows),{'aggregation':'mean_per_validation_scene','scenarios':rows,'accepted':all(row['accepted'] for row in rows)}
    score,metrics=validate_all()
    selection=cfg.get('validation_selection');reference_metrics=metrics
    if selection:
        from .validation_policy import capped_selection
        score,selection_detail=capped_selection(metrics,reference_metrics,**selection)
        if not selection_detail['eligible']:raise RuntimeError('Initial validation must pass before capped model selection')
        metrics={**metrics,'selection':selection_detail}
    best=score;save(0,'shared.pt')
    history.append({'step':0,'validation_score':score,'validation':metrics});print('initial validation',score,metrics,flush=True)
    graphs={};gradient_pointers=None
    source_fingerprints={id(scene):static_scene_fingerprint(scene) for scene,_ in samples}
    for scene,states in samples:
        scene.training_signature=training_layout_signature(scene)
        if scene.training_signature not in graphs:
            carrier=MixedScene({key:value.clone() for key,value in scene.values.items()},scene.metadata)
            if any(value.numel() and value.data_ptr()==scene.values[key].data_ptr() for key,value in carrier.values.items()):raise RuntimeError('Training carrier aliases source tensors')
            graphs[scene.training_signature]=TrainingGraph(model,carrier,states,scene_config(config,scene),optimizer)
            pointers=[None if p.grad is None else p.grad.data_ptr() for p in model.parameters()]
            if gradient_pointers is not None and pointers!=gradient_pointers:raise RuntimeError('Training graphs do not share stable gradient storage')
            gradient_pointers=pointers
    print('captured training layouts',len(graphs),flush=True)
    rng=random.Random(config['seed'])
    index_rng=None if 'batch_seed' not in cfg else torch.Generator(device='cuda').manual_seed(cfg['batch_seed'])
    sample_counts={scene.dataset_label:0 for scene,_ in samples}
    report={'status':'RUNNING','architecture':model.architecture_id,'config':config,'history':history,'best_validation_score':best,'training_graph_count':len(graphs),'shared_gradient_storage_verified':True,'sampled_training_updates':sample_counts.copy(),'selected_checkpoint_sampled_updates':selected_sample_counts.copy(),'completed_updates':0,'selected_checkpoint_step':base_step,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'trainable_parameter_names':[name for name,p in model.named_parameters() if p.requires_grad]}
    if cfg.get('record_sample_history'):report['optimization_samples']=[]
    def write_progress(status,completed):
        report.update(status=status,seconds=time.perf_counter()-start,completed_updates=completed,sampled_training_updates=sample_counts.copy(),selected_checkpoint_sampled_updates=selected_sample_counts.copy(),best_validation_score=best)
        path=report_dir/training_report;temporary=path.with_suffix('.tmp')
        temporary.write_text(json.dumps(report,indent=2)+'\n');temporary.replace(path)
    write_progress('RUNNING',0)
    for step in range(1,cfg['steps']+1):
        weights=[cfg.get('official_sampling_weight',1.) if s.metadata.get('official_demo') else 1. for s,_ in samples]
        if cfg.get('ensure_initial_coverage') and step<=len(samples):sample,states=samples[step-1]
        else:sample,states=rng.choices(samples,weights=weights,k=1)[0] if cfg.get('official_sampling_weight') else rng.choice(samples)
        if sample.metadata.get('data_layout')=='teacher_queried_windows_v1':
            indices=sample.qualified_starts[torch.randint(len(sample.qualified_starts),(cfg['batch_size'],),device='cuda',generator=index_rng)]
        else:indices=torch.randint(len(states['x'])-cfg['horizon'],(cfg['batch_size'],),device='cuda',generator=index_rng)
        objective,accepted=graphs[sample.training_signature].run(sample,states,indices)
        if not bool(accepted.all()) or not bool(torch.isfinite(objective)):
            save(step-1,'rejected.pt')
            failure={'step':step,'completed_updates':step-1,'indices':indices.cpu().tolist(),'accepted':accepted.cpu().tolist(),'scene':sample.metadata['index'],'dataset':sample.dataset_label,'loss':float(objective)}
            (report_dir/'training_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            write_progress('FAILED',step-1)
            raise RuntimeError(f'Training proposal rejected at {step}; keep data and diagnose: {failure}')
        try:
            norm=clip_gradient_norm_fp64(model.parameters(),cfg['gradient_clip']) if cfg.get('fp64_gradient_norm') else torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip'],error_if_nonfinite=True)
        except RuntimeError:
            save(step-1,'gradient_failure.pt')
            failure={'step':step,'completed_updates':step-1,'indices':indices.cpu().tolist(),'dataset':sample.dataset_label,'loss':float(objective),'nonfinite_gradients':[name for name,p in model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]}
            (report_dir/'gradient_failure.json').write_text(json.dumps(failure,indent=2)+'\n')
            write_progress('FAILED',step-1)
            raise
        optimizer.step()
        sample_counts[sample.dataset_label]+=1
        if cfg.get('record_sample_history'):report['optimization_samples'].append({'update':base_step+step,'dataset':sample.dataset_label,'indices':indices.cpu().tolist(),'gradient_norm_before_clipping':float(norm)})
        if step%20==0:
            write_progress('RUNNING',step)
            print(f'{step}/{cfg["steps"]} loss={float(objective):.6g} grad={float(norm):.4g} seconds={time.perf_counter()-start:.1f}',flush=True)
        if step%cfg['validation_interval']==0 or step==cfg['steps']:
            score,metrics=validate_all()
            eligible=True
            if selection:
                score,selection_detail=capped_selection(metrics,reference_metrics,**selection)
                eligible=selection_detail['eligible'];metrics={**metrics,'selection':selection_detail}
            history.append({'step':step,'loss':float(objective),'validation_score':score,'validation':metrics,'seconds':time.perf_counter()-start})
            save(step,'last.pt')
            if eligible and score<best:
                best=score;save(step,'shared.pt');report['selected_checkpoint_step']=base_step+step
                selected_sample_counts=sample_counts.copy()
            write_progress('RUNNING',step);print('validation',step,score,metrics,flush=True)
        if time.perf_counter()-start>cfg['max_seconds']:break
    report.update(status='COMPLETED' if step==cfg['steps'] else 'BUDGET_REACHED',seconds=time.perf_counter()-start,checkpoint_sha256=hashlib.sha256((directory/'shared.pt').read_bytes()).hexdigest())
    report['completed_updates']=step
    report['selected_checkpoint_sampled_updates']=selected_sample_counts.copy()
    report['selected_checkpoint_step']=int(torch.load(directory/'shared.pt',weights_only=True)['step'])
    report['base_checkpoint_step']=base_step
    for scene,_ in samples:
        if static_scene_fingerprint(scene)!=source_fingerprints[id(scene)]:raise RuntimeError('Training mutated an immutable source scene')
    report['training_source_static_values_preserved']=True
    for name,p in model.named_parameters():
        if name in frozen_parameters and not torch.equal(p,frozen_parameters[name]):raise RuntimeError('Training changed a frozen parameter: '+name)
    report['frozen_parameters_preserved']=True
    if config.get('initialize_from'):
        report['initialization_checkpoint_step']=base['step']
        report['initialization_checkpoint_sha256']=hashlib.sha256(Path(config['initialize_from']).read_bytes()).hexdigest()
    (report_dir/training_report).write_text(json.dumps(report,indent=2)+'\n');return report


def evaluate(config):
    configure(config);sys.modules['newton']=None
    checkpoint_path=ROOT/config['checkpoint_dir']/'shared.pt';checkpoint=torch.load(checkpoint_path,weights_only=True)
    model=MixedSolver(config['model']).cuda().eval();model.load_state_dict(checkpoint['model'])
    scene,states=load_sample(ROOT/config['data_dir']/'mixed_09.npz')
    start=time.perf_counter();sequence,accepted=rollout(model,scene,states,config)
    predicted={key:torch.cat([getattr(s,key) for s in sequence],0) for key in ('x','q','v','omega')}
    records=[]
    for group in scene.metadata['groups']:
        obj=next(i for i,g in enumerate(scene.metadata['groups']) if g['name']==group['name']);mask=(scene.object_id==obj)&scene.movable
        error=(predicted['x'][:,mask]-states['x'][:len(sequence),mask]).norm(dim=-1)/scene.length
        rotation=so3.log(so3.multiply(predicted['q'][:,mask],so3.conjugate(states['q'][:len(sequence),mask]))).norm(dim=-1)
        records.append({'family':group['family'],'position_rmse_over_L':float(error.square().mean().sqrt()),'position_p95_over_L':float(torch.quantile(error.flatten(),.95)),'angular_rmse_radians':float(rotation.square().mean().sqrt()) if group['family'] in ('joint','rigid','rod') else None})
    report={'status':'ROLLOUT_EXECUTED' if accepted and len(sequence)==config['steps']+1 else 'FAIL','architecture':model.architecture_id,'checkpoint_step':checkpoint['step'],'checkpoint_sha256':hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(),'newton_import_blocked':True,'same_checkpoint_all_types':True,'steps':len(sequence)-1,'seconds':time.perf_counter()-start,'objects':records,'contract':config['contract'],'scope':config['scope']}
    directory=ROOT/config['report_dir'];np.savez_compressed(directory/'test_prediction.npz',**{key:value.cpu().numpy() for key,value in predicted.items()})
    (directory/'evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report


def evaluate_frozen(config):
    configure(config);sys.modules['newton']=None
    directory=ROOT/config['report_dir'];snapshot=json.loads((ROOT/'configs/mixed_response_frozen_test.json').read_text())
    manifest=json.loads((directory/'frozen_test/data_manifest.json').read_text())
    if manifest['status']!='PASS':raise RuntimeError('Frozen reference verification failed')
    path=ROOT/config['checkpoint_dir']/'shared.pt';checkpoint=torch.load(path,weights_only=True)
    expected=hashlib.sha256((directory/'data_manifest.json').read_bytes()).hexdigest()
    if checkpoint['data_manifest_sha256']!=expected:raise RuntimeError('Training manifest changed')
    model=MixedSolver(checkpoint['config']['model']).cuda().eval();model.load_state_dict(checkpoint['model'])
    class Physical(torch.nn.Module):
        def forward(self,*args,**kwargs):return model(*args,learned=False,**kwargs)
    cases=[]
    for row in manifest['scenes']:
        data=ROOT/row['path']
        if hashlib.sha256(data.read_bytes()).hexdigest()!=row['sha256']:raise RuntimeError('Frozen case changed')
        scene,states=load_sample(data);reference_start=select(states,slice(0,1))
        predictions={};metrics={}
        for name,solver in [('learned',model),('physical_only',Physical())]:
            graph=CapturedStepper(solver,scene,reference_start,config['dt']);current=reference_start;sequence=[current]
            accepted=True;times=[];root_count=0
            for i in range(config['steps']):
                torch.cuda.synchronize();start=time.perf_counter();candidate,diag=graph.step(current);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                if not bool(diag['accepted'].all()):accepted=False;break
                current=candidate;sequence.append(current);root_count+=int(diag['numerical_root_entities'].sum())
            predicted={key:torch.cat([getattr(s,key) for s in sequence],0) for key in ('x','q','v','omega')}
            objects=[]
            for obj,group in enumerate(scene.metadata['groups']):
                mask=(scene.object_id==obj)&scene.movable
                error=(predicted['x'][:,mask]-states['x'][:len(sequence),mask]).norm(dim=-1)/scene.length
                angle=so3.log(so3.multiply(predicted['q'][:,mask],so3.conjugate(states['q'][:len(sequence),mask]))).norm(dim=-1)
                objects.append({'family':group['family'],'position_rmse_over_L':float(error.square().mean().sqrt()),'position_p95_over_L':float(torch.quantile(error.flatten(),.95)),'angular_rmse_radians':float(angle.square().mean().sqrt()) if group['family'] in ('joint','rigid','rod') else None})
            sites=predicted['x'][:,scene.site_owner]+so3.rotate(predicted['q'][:,scene.site_owner],scene.site_local)
            clearance=float((sites[...,2]-scene.site_radius-scene.plane_height).min())
            p=predicted['x'][:,scene.tets];deformation=(p[:,:,1:]-p[:,:,:1]).transpose(-1,-2)@scene.tet_Dm
            det=float(torch.linalg.det(deformation).min())
            index=next(g['body_start']+scene.metadata['particle_count'] for g in scene.metadata['groups'] if g['family']=='rigid')
            displacement=float(predicted['x'][-1,index,0]-predicted['x'][0,index,0])
            reference=float(states['x'][-1,index,0]-states['x'][0,index,0])
            metrics[name]={'accepted':accepted,'steps':len(sequence)-1,'objects':objects,'minimum_physical_plane_clearance':clearance,'minimum_detF':det,'free_box_x_displacement':displacement,'reference_free_box_x_displacement':reference,'synchronized_step_p50_ms':1000*float(np.median(times)),'numerical_root_entity_count':root_count}
            predictions[name]=predicted
        # Initial state only; future arrays are deliberately made unusable.
        poisoned={key:value.clone() for key,value in states.items()}
        for value in poisoned.values():value[1:]=float('nan')
        sequence,ok=rollout(model,scene,poisoned,config,steps=3)
        error=float((torch.cat([s.x for s in sequence],0)-predictions['learned']['x'][:len(sequence)]).abs().max())
        assert ok and error<=1e-6
        np.savez_compressed(directory/f'frozen_{row["index"]:02d}_prediction.npz',**{key:value.cpu().numpy() for key,value in predictions['learned'].items()})
        cases.append({'index':row['index'],'effective_friction':float(scene.site_mu.max()),'metrics':metrics,'future_teacher_poison_max_error':error})
        print(json.dumps(cases[-1],indent=2),flush=True)
    passed=all(c['metrics']['learned']['accepted'] and c['metrics']['learned']['steps']==240 and c['metrics']['learned']['minimum_physical_plane_clearance']>=-1e-7 and c['metrics']['learned']['minimum_detF']>=.05 for c in cases)
    zero,rough=cases
    friction_order=zero['metrics']['learned']['free_box_x_displacement']>rough['metrics']['learned']['free_box_x_displacement']
    report={'status':'FROZEN_ROLLOUT_EXECUTED' if passed and friction_order else 'FAIL','checkpoint_step':checkpoint['step'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'same_checkpoint_all_cases_and_types':True,'newton_import_blocked':True,'friction_response_order_correct':friction_order,'cases':cases,'scope':config['scope'],'formal_demo_acceptance':False}
    (directory/'frozen_evaluation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2));return report
