"""Declared complete rigid witness capacity, shared by scenes and queries."""
from collections import Counter


def capacity(model, scene, newton):
    count = max(1024, model.rigid_contact_max or 0)
    policy = 'source_capacity_with_1024_floor'
    if scene.metadata.get('contract') in ('rod_shape_contact_com_v40','rod_finite_joint_contact_com_v143'):
        types = Counter(model.shape_type.numpy().tolist())
        capsule, sphere, plane = (int(getattr(newton.GeoType, name)) for name in ('CAPSULE', 'SPHERE', 'PLANE'))
        if set(types) <= {capsule, sphere, plane}:
            c, s, p = (types[key] for key in (capsule, sphere, plane))
            count = max(count, c*(c-1)+c*s+s*(s-1)//2+2*c*p+s*p)
            policy = 'capsule_sphere_plane_all_pair_witness_bound'
    return count, policy
