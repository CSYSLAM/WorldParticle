"""Fit material amplitudes on cached shared-backbone features, then export.

Phase I freezes the parent trunk. Loss measures actual projected displacement
in a mass metric; Gram matrices are OFFLINE training labels, never inputs to
the deployed material readout. Validation windows are not unseen assets.
"""
import argparse,hashlib,json,os,random,shutil,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,nargs='+',required=True)
    p.add_argument('--checkpoint',type=Path,default=ROOT/'model_releases/shared_vbd_v24_cloth40/model.pt')
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--updates',type=int,default=400);p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--hidden',type=int,default=256);p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--validation-interval',type=int,default=20);p.add_argument('--patience',type=int,default=5)
    a=p.parse_args()
    if min(a.updates,a.batch_size,a.validation_interval,a.patience)<1:p.error('Positive training budgets required')
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    import torch
    from wp_vbd.shared_material_correction import SharedMaterialCorrection
    torch.set_num_threads(1);torch.manual_seed(3101);random.seed(3101);torch.use_deterministic_algorithms(True)
    parent=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config=dict(parent.get('model_config') or parent['config']['model'])
    width=config['protected_width'];head=SharedMaterialCorrection(width,hidden=a.hidden).to(a.device)
    rows=[];provenance=[]
    for directory in a.data:
        capture=json.loads((directory/'capture.json').read_text())
        if capture['status']!='COMPLETE':raise ValueError('Only complete material capture trajectories admitted')
        if capture['checkpoint_sha256']!=hashlib.sha256(a.checkpoint.read_bytes()).hexdigest():raise ValueError('Capture parent weights differ')
        path=directory/'examples.pt';data=torch.load(path,map_location='cpu',weights_only=True)
        provenance.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),scene=capture['scene'],samples=capture['samples']))
        for index,sample in enumerate(data['samples']):
            row={key:tensor[index] for key,tensor in data['tensors'].items()}
            row.update(sample);rows.append(row)
    if len(rows)<40:raise ValueError('At least 40 captured material states required')
    train=[];dev=[]
    for index,row in enumerate(rows):
        # 20 student steps per block (~0.11 seconds). Entire held-out blocks
        # provide a development signal, not independent-asset generalization.
        (dev if (row['step']//20)%5==3 else train).append(index)
    if not train or not dev:raise ValueError('Training and held-out temporal blocks required')
    keys=('features','gram','cross','norm')
    tensors={key:torch.stack([row[key] for row in rows]).to(a.device) for key in keys}
    scene_names=sorted({row['scene'] for row in rows});by_scene={name:[i for i in train if rows[i]['scene']==name] for name in scene_names}
    optimizer=torch.optim.AdamW(head.parameters(),lr=a.learning_rate,weight_decay=1e-6)
    generator=torch.Generator().manual_seed(3101)
    def values(indices):
        index=torch.as_tensor(indices,device=a.device)
        prediction=head(tensors['features'][index]).double()
        gram=tensors['gram'][index];cross=tensors['cross'][index];norm=tensors['norm'][index]
        error=(torch.einsum('bp,bpq,bq->b',prediction,gram,prediction)-2*(prediction*cross).sum(-1)+norm).clamp_min(0)
        loss=(error/(norm+1e-4)).mean()+1e-7*prediction.square().mean()
        return loss,error,norm
    def validate(indices):
        with torch.no_grad():
            loss,error,norm=values(indices)
            return dict(loss=float(loss),relative_squared_displacement_error=float(error.sum()/(norm.sum()+1e-20)),
                target_normalized_squared_displacement=float(norm.sum()),mean_normalized_error=float(error.mean()))
    record={'status':'RUNNING','phase':'frozen_backbone_shared_latent_material_distillation',
        'parent_checkpoint':str(a.checkpoint.resolve()),'parent_sha256':hashlib.sha256(a.checkpoint.read_bytes()).hexdigest(),
        'frozen_shared_width':width,'trainable_parameters':sum(p.numel() for p in head.parameters()),
        'backbone_updated':False,'samples':len(rows),'training_samples':len(train),'validation_samples':len(dev),
        'validation_scope':'Held-out contiguous blocks within the same full trajectories; NOT unseen assets or materials',
        'hyperparameters':dict(hidden=a.hidden,learning_rate=a.learning_rate,batch_size=a.batch_size,
            maximum_updates=a.updates,validation_interval=a.validation_interval,patience=a.patience),
        'datasets':provenance,'seed':3101,'initial_validation':validate(dev),'history':[],'best_update':0}
    best=float('inf');stale=0;best_state=None;started=time.perf_counter()
    def write():
        record['wall_seconds']=time.perf_counter()-started
        (a.output/'training.json').write_text(json.dumps(record,indent=2)+'\n')
    for update in range(1,a.updates+1):
        optimizer.zero_grad(set_to_none=True)
        chosen=[]
        for name in scene_names:
            candidates=by_scene[name]
            sample=torch.randint(len(candidates),(max(1,a.batch_size//len(scene_names)),),generator=generator).tolist()
            chosen.extend(candidates[i] for i in sample)
        loss,_,_=values(chosen)
        if not bool(torch.isfinite(loss)):raise ValueError('Nonfinite material distillation objective')
        loss.backward();torch.nn.utils.clip_grad_norm_(head.parameters(),5.);optimizer.step()
        if update%a.validation_interval==0 or update==a.updates:
            result=validate(dev);per_scene={name:validate([i for i in dev if rows[i]['scene']==name]) for name in scene_names}
            row=dict(update=update,training_loss=float(loss.detach()),validation=result,per_scene=per_scene)
            record['history'].append(row)
            score=result['loss']
            if score<best*(1-1e-3):
                best=score;stale=0;best_state={k:v.detach().cpu().clone() for k,v in head.state_dict().items()};record['best_update']=update
                state={'head':best_state,'optimizer':optimizer.state_dict(),'torch_rng':torch.get_rng_state(),
                    'cuda_rng':torch.cuda.get_rng_state() if str(a.device).startswith('cuda') else None,
                    'batch_generator':generator.get_state(),'update':update,'parent_sha256':record['parent_sha256']}
                torch.save(state,a.output/'training_state.pt')
            else:stale+=1
            write();print(json.dumps(row),flush=True)
            if stale>=a.patience:record['stop_reason']='development_loss_plateau';break
    if best_state is None:raise ValueError('No completed validation checkpoint')
    head.load_state_dict(best_state);record['final_validation']=validate(dev)
    record['final_per_scene']={name:validate([i for i in dev if rows[i]['scene']==name]) for name in scene_names}
    record['module_validation_passed']=bool(record['final_validation']['relative_squared_displacement_error']<.6
        and all(row['relative_squared_displacement_error']<1. for row in record['final_per_scene'].values()))
    config.update(shared_material_correction=True,shared_material_hidden=a.hidden,shared_material_use_learned=True)
    weights=dict(parent['model']);weights.update({'shared_material_head.'+k:v for k,v in best_state.items()})
    torch.save(dict(model=weights,model_config=config,step=record['best_update'],
        architecture='shared_material_corrector_v31',parent_backbone_step=parent['step'],
        parent_sha256=record['parent_sha256'],training_phase=record['phase']),a.output/'model.pt')
    recipe=json.loads((ROOT/'configs/robot_validation_cloth_ft.json').read_text())
    recipe.update(checkpoint=str(a.output/'model.pt'),checkpoint_sha256=hashlib.sha256((a.output/'model.pt').read_bytes()).hexdigest(),
        validation_status='EXPERIMENTAL_MATERIAL_HEAD_ROLLOUTS_REQUIRED')
    (a.output/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
    record.update(status='COMPLETE',completed_updates=update,stop_reason=record.get('stop_reason','maximum_updates'),
        checkpoint_sha256=recipe['checkpoint_sha256'],runtime_teacher_queries=0)
    write()
    snapshot=a.output/'implementation_snapshot'
    for source in [Path(__file__),*(ROOT/'src/wp_vbd').glob('*.py')]:
        target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
    (a.output/'split.json').write_text(json.dumps(dict(train=train,development=dev),indent=2)+'\n')
    try:
        import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
        history=record['history'];fig,axis=plt.subplots(figsize=(7,4))
        axis.plot([r['update'] for r in history],[r['training_loss'] for r in history],label='training')
        axis.plot([r['update'] for r in history],[r['validation']['loss'] for r in history],label='held temporal blocks')
        axis.set(xlabel='Head updates (parent trunk frozen)',ylabel='Mass-metric displacement loss');axis.legend();fig.tight_layout();fig.savefig(a.output/'loss.png',dpi=160);plt.close(fig)
    except ImportError:pass
    print(json.dumps({k:record[k] for k in ('status','completed_updates','best_update','initial_validation','final_validation')}),flush=True)


if __name__=='__main__':main()
