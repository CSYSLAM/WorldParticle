"""Sample the actual paper surface against the original gripper visual solids.

This is an offline geometry audit, not a contact force or state correction.
Unlike particle-only checks, it also samples triangle interiors. Reported
depths are sampled estimates; spacing bounds the resolution of this audit.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def surface_samples(vertices,faces,spacing):
    triangles=np.asarray(vertices,dtype=np.float64)[faces]
    lengths=np.linalg.norm(triangles-np.roll(triangles,1,axis=1),axis=-1).max(1)
    divisions=np.maximum(1,np.ceil(lengths/spacing).astype(int))
    parts=[]
    for n in np.unique(divisions):
        weights=np.asarray([(a/n,b/n) for a in range(n+1) for b in range(n+1-a)])
        t=triangles[divisions==n]
        parts.append((t[:,None,0]+weights[None,:,0,None]*(t[:,None,1]-t[:,None,0])+
                      weights[None,:,1,None]*(t[:,None,2]-t[:,None,0])).reshape(-1,3))
    return np.concatenate(parts)


class GripperBagClearance:
    def __init__(self,model,*,owners=(15,16,17),spacing=.002):
        import trimesh
        self.meshes=[];self.spacing=float(spacing)
        bodies=model.shape_body.numpy();poses=model.shape_transform.numpy();scales=model.shape_scale.numpy()
        for index,source in enumerate(model.shape_source):
            if source is None or int(bodies[index]) not in owners:continue
            local=Rotation.from_quat(poses[index,3:]).apply(np.asarray(source.vertices)*scales[index])+poses[index,:3]
            mesh=trimesh.Trimesh(vertices=local,faces=np.asarray(source.indices).reshape(-1,3),process=True)
            mesh.fix_normals(multibody=True)
            self.meshes.append((int(bodies[index]),index,mesh))
        if not self.meshes:raise ValueError('No original gripper visual meshes found')

    def measure(self,body_pose,particles,faces):
        import trimesh
        points=surface_samples(particles,faces,self.spacing)
        worst=0.;owner=None;count=0;rows=[];unresolved=[]
        for body,index,mesh in self.meshes:
            rotation=Rotation.from_quat(body_pose[body,3:])
            local=rotation.inv().apply(points-body_pose[body,:3])
            if not mesh.is_watertight:
                # Imported palm/wrist CAD contains some open surfaces. Do
                # not invent an inside sign or repair its topology. Prove
                # separation conservatively using triangle AABBs instead.
                triangles=rotation.inv().apply(particles-body_pose[body,:3])[faces]
                overlap=((triangles.max(1)>=mesh.bounds[0])&(triangles.min(1)<=mesh.bounds[1])).all(1)
                if overlap.any():unresolved.append(index)
                rows.append(dict(body=body,shape=index,watertight=False,
                                 possible_surface_overlap=bool(overlap.any())))
                continue
            selected=((local>=mesh.bounds[0])&(local<=mesh.bounds[1])).all(1)
            depth=trimesh.proximity.signed_distance(mesh,local[selected]) if selected.any() else np.empty(0)
            positive=depth>1e-7;maximum=max(0.,float(depth.max())) if len(depth) else 0.
            count+=int(positive.sum())
            rows.append(dict(body=body,shape=index,watertight=True,sampled_inside_count=int(positive.sum()),maximum_sampled_depth_m=maximum))
            if maximum>worst:worst=maximum;owner=body
        return dict(maximum_sampled_paper_inside_visual_depth_m=worst,worst_body=owner,
                    sampled_inside_count=count,surface_sample_count=len(points),meshes=rows,
                    unresolved_open_visual_shapes=unresolved)
