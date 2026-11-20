"""Teacher-relative drape supervision; no flat-sheet or rigid-following target."""
import torch

from .multirate_objective import RolloutObjective,robust


class ClothRolloutObjective:
    def __init__(self,scene):
        self.base=RolloutObjective(scene)
        self.scene=scene
        self.ids=torch.arange(scene.metadata['particle_count'],device=scene.device)
        self.faces=scene.triangles
        # Nonlocal distances expose a narrow folded bundle even if area is preserved.
        a,b=torch.triu_indices(len(self.ids),len(self.ids),offset=1,device=scene.device)
        distance=(scene.rest_x[a]-scene.rest_x[b]).norm(dim=-1)
        self.pair_a=a[distance>.06];self.pair_b=b[distance>.06]

    def __call__(self,predicted,target,diagnostic):
        value,terms,metrics=self.base(predicted,target,diagnostic)
        def normals(x):
            p=x[:,self.faces]
            n=torch.linalg.cross(p[:,:,1]-p[:,:,0],p[:,:,2]-p[:,:,0])
            return n/n.norm(dim=-1,keepdim=True).clamp_min(1e-12)
        normal_error=1-(normals(predicted.x)*normals(target.x)).sum(-1).clamp(-1,1)
        terms['drape_normals']=robust(normal_error,.2).mean()
        def pair_lengths(x):return (x[:,self.pair_a]-x[:,self.pair_b]).norm(dim=-1)
        pair_error=pair_lengths(predicted.x)-pair_lengths(target.x)
        terms['drape_nonlocal']=robust(pair_error,.005).mean()
        # Shape-changing kinetic energy is measured after subtracting cloth COM motion.
        mass=self.scene.mass[self.ids];weights=mass/mass.sum()
        def relative_v(state):
            v=state.v[:,self.ids]
            return v-(v*weights[None,:,None]).sum(1,keepdim=True)
        pv,tv=relative_v(predicted),relative_v(target)
        pk=(pv.square().sum(-1)*weights).sum(-1)
        tk=(tv.square().sum(-1)*weights).sum(-1)
        terms['drape_excess_kinetic']=robust((pk-tk).clamp_min(0),.03).mean()
        value=value+.3*terms['drape_normals']+.2*terms['drape_nonlocal']+.2*terms['drape_excess_kinetic']
        metrics.update(drape_pair_rms_m=pair_error.square().mean().sqrt(),
                       drape_normal_cosine_error=normal_error.mean(),
                       cloth_internal_speed_sq=pk.mean(),teacher_internal_speed_sq=tk.mean())
        return value,terms,metrics
