"""Contact numerical weights from owned material and mobility descriptors.

These weights condition the fixed response rounds; contact ke, damping and
friction remain independent physical inputs. No dynamics solver is imported.
"""
import torch


def body_structural_stiffness(scene):
    """Max enabled source linear-joint slot per body, with a zero world slot.

    Rod exports currently require every rod enabled. Their first three
    material entries contain shear/shear/stretch. Finite D6 joints retain
    two structural slots even when all three linear axes are free: their
    first slot is then zero, regardless of angular constraints or drives.
    Read current descriptors each step so material/control refreshes apply.
    """
    summary=torch.zeros_like(scene.mass)
    rows=[]
    rod=getattr(scene,'rod',None)
    if rod is not None:
        rows.append((rod.joints,rod.joint_stiffness[:,:3].amax(-1)))
    spec=getattr(scene,'joint_spec',None)
    if spec is not None:
        has_linear=(spec.dof_index[:,:3]>=0).sum(-1)<3
        rows.append((spec.owners,torch.where(spec.enabled&has_linear,spec.stiffness[:,0],0.)))
    for owners,stiffness in rows:
        for side in (0,1):
            # The explicit static world slot must not collect attached k.
            values=torch.where(owners[:,side]!=scene.world,stiffness,0.)
            summary=summary.scatter_reduce(0,owners[:,side].long(),values,reduce='amax',include_self=True)
    return summary


def series(a,b):
    """Stable series stiffness, zero if either participating scale is zero."""
    lo=torch.minimum(a,b);hi=torch.maximum(a,b)
    safe_hi=torch.where(hi>0,hi,1.)
    return torch.where((a>0)&(b>0),lo/(1+lo/safe_hi),0.)


def conditioned_supports(scene,contacts,normal_delassus,tangent_support,dt):
    from .rigid_contact_rows import numerical_weight_policy
    ids=contacts.owners;mobile=scene.movable[ids]
    structural=body_structural_stiffness(scene)[ids]
    w=normal_delassus
    inertial=1/(dt**2*torch.where(w>0,w,1.))
    single=torch.where((w>0)&mobile[None],inertial+structural[None],0.)
    a,b=single.unbind(-1)
    normal=torch.where((a>0)&(b>0),series(a,b),torch.maximum(a,b))
    s=torch.where(mobile,structural,0.)
    a,b=s.unbind(-1)
    pair=torch.where(mobile.all(-1),series(a,b),torch.maximum(a,b))
    return numerical_weight_policy(normal,tangent_support,pair[None],contacts.material[None])
