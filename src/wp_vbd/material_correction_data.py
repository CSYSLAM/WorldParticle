"""Offline material-response labels from unchanged accepted model rollouts."""
from pathlib import Path
import json
import torch

from .shared_material_correction import object_inputs,MODE_COUNT
from .protected_response import metric_action


class MaterialCorrectionRecorder:
    def __init__(self,output,*,scene,stride=2):
        if stride<1:raise ValueError('Positive capture stride required')
        self.output=Path(output);self.scene=scene;self.stride=stride
        self.calls=0;self.rows=[];self.maximum_oracle_error=0.

    def observer(self,model,scene,state,metric,coarse,residual,update,whitening,baseline,dt):
        index=self.calls;self.calls+=1
        if index%self.stride:return None
        latent=model.protected_response.last_latent
        if latent is None:raise ValueError('Teacher capture requires this step shared latent')
        def record(component,objects,effective,target,valid):
            if len(objects)!=1:raise ValueError('Phase-I labels require one material object per correction block')
            if not bool(valid.all()):raise ValueError('Invalid material teacher is not a training label')
            ids=objects[0]
            with torch.no_grad():
                features,basis,scale=object_inputs(scene,state,ids,latent,effective,update,whitening,baseline,dt)
                raw=update.new_zeros(len(update),scene.n,6,MODE_COUNT,dtype=torch.float64)
                raw=raw.index_copy(1,ids,torch.nn.functional.pad(basis,(0,0,0,3)))
                columns=(raw-coarse(metric_action(metric,raw)))[:,component]
                weights=scene.mass[component].double()[:,None]*columns.new_tensor([1,1,1,*([float(scene.length)**2]*3)])
                total=scene.mass[ids].double().sum()
                gram=torch.einsum('bnip,ni,bniq->bpq',columns,weights,columns)/total
                cross=torch.einsum('bnip,ni,bni->bp',columns,weights,target)/(total*scale[:,None])
                norm=torch.einsum('bni,ni,bni->b',target,weights,target)/(total*scale.square())
                eigen,frame=torch.linalg.eigh((gram+gram.transpose(-1,-2))/2)
                # This is an OFFLINE mass-metric representability check,
                # not the teacher's energy-system rank decision. Near-planar
                # cloth needs a tighter cutoff than the physical H solve.
                threshold=eigen.abs().amax(-1,keepdim=True).clamp_min(1e-30)*1e-12
                inverse=torch.where(eigen>threshold,eigen.clamp_min(1e-300).reciprocal(),0.)
                coefficient=(frame@(inverse[...,None]*(frame.transpose(-1,-2)@cross[...,None])))[...,0]
                error=(torch.einsum('bp,bpq,bq->b',coefficient,gram,coefficient)-2*(coefficient*cross).sum(-1)+norm).clamp_min(0)
                relative=error/(norm+1e-12)
                self.maximum_oracle_error=max(self.maximum_oracle_error,float(relative.max()))
                if not bool(torch.isfinite(gram).all()&torch.isfinite(cross).all()&torch.isfinite(norm).all()):
                    raise ValueError('Finite offline material regression required')
                if bool((relative>1e-4).any()):
                    torch.save(dict(gram=gram.cpu(),cross=cross.cpu(),norm=norm.cpu(),features=features.cpu(),
                        eigen=eigen.cpu(),scale=scale.cpu()),self.output/'basis_failure.pt')
                    raise ValueError('Material-frame basis cannot represent teacher response')
                for batch in range(len(features)):
                    self.rows.append(dict(features=features[batch].float().cpu(),gram=gram[batch].cpu(),
                        cross=cross[batch].cpu(),norm=norm[batch].cpu(),oracle=coefficient[batch].cpu(),
                        scale=scale[batch].cpu(),step=index,dt=dt,scene=self.scene,nodes=len(ids)))
                if len(self.rows)%30==0:
                    (self.output/'capture_progress.json').write_text(json.dumps(dict(samples=len(self.rows),
                        neural_step=index,maximum_oracle_error=self.maximum_oracle_error))+'\n')
        return record

    def finish(self,*,complete):
        self.output.mkdir(parents=True,exist_ok=True)
        report={'status':'COMPLETE' if complete and self.rows else 'FAILED','scene':self.scene,
            'observed_neural_material_stages':self.calls,'capture_stride':self.stride,'samples':len(self.rows),
            'maximum_mass_metric_oracle_relative_error':self.maximum_oracle_error,
            'oracle_diagnostic_only':True,'oracle_relative_eigenvalue_cutoff':1e-12,
            'teacher':'Current analytic material correction at current model states; no native VBD dynamics',
            'basis':'Material-frame 24-column polynomial basis with the original protected projection',
            'runtime_inputs':'Pooled shared latent and cheap local/proposal mode summaries; no energy Gram',
            'scope':'Module distillation on complete robot trajectories, not unseen-asset generalization.'}
        if self.rows:
            tensors={key:torch.stack([r[key] for r in self.rows]) for key in ('features','gram','cross','norm','oracle','scale')}
            torch.save({'schema':'shared_material_distillation_v31','tensors':tensors,
                'samples':[{k:v for k,v in row.items() if k not in tensors} for row in self.rows]},self.output/'examples.pt')
        (self.output/'capture.json').write_text(json.dumps(report,indent=2)+'\n')
        return report
