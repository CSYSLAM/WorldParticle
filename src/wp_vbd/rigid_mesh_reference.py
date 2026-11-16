"""Portable mesh identity and explicit reference mass-property inputs."""
import hashlib
import numpy as np

MASS_FIELDS=('body_mass','body_inv_mass','body_com','body_inertia','body_inv_inertia')


def mesh_identity(model,newton,*,require_sdf):
    result=[]
    for shape in np.flatnonzero(model.shape_type.numpy()==int(newton.GeoType.MESH)):
        mesh=model.shape_source[shape]
        if require_sdf and mesh.sdf is None:raise ValueError('Real authored mesh SDF required before a training collision oracle')
        values={}
        for name in ('vertices','indices'):
            value=np.ascontiguousarray(getattr(mesh,name))
            values[name]={'shape':list(value.shape),'dtype':value.dtype.str,'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
        result.append({'shape':int(shape),**values})
    if not result:raise ValueError('Mesh-reference contract requires actual mesh geometry')
    return result


def filter_pairs(model):
    pairs=np.asarray(model.shape_collision_filter_pairs.pairs_array()).reshape(-1,2)
    return np.ascontiguousarray(pairs[np.lexsort((pairs[:,1],pairs[:,0]))])


def restore_mass_inputs(model,recorded):
    """Validate per-body float32 reduction bounds, then input exact materials.

    Geometry must be independently identified before this function is used.
    The archived physical parameters are explicit training inputs. This is
    not a permission to replace changed geometry or arbitrary body properties.
    """
    differences={}
    for name in MASS_FIELDS:
        target=getattr(model,name);a=target.numpy();b=np.asarray(recorded[name])
        if a.shape!=b.shape or a.dtype!=b.dtype or not np.isfinite(a).all() or not np.isfinite(b).all():
            raise ValueError('Matched finite native reference mass field required: '+name)
        delta=np.abs(a.astype(np.float64)-b.astype(np.float64)).reshape(model.body_count,-1).max(-1)
        scale=np.abs(b.astype(np.float64)).reshape(model.body_count,-1).max(-1)
        if np.any(delta>1e-5*scale):raise ValueError('Mesh mass-property difference exceeds per-body reduction bound: '+name)
        if np.any(delta):differences[name]=float(np.max(np.divide(delta,scale,out=np.zeros_like(scale),where=scale>0)))
    # Validate the complete packet before changing any native property.
    for name in MASS_FIELDS:getattr(model,name).assign(recorded[name])
    return differences
