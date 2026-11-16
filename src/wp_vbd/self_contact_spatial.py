"""Conservative CPU sphere-tree prefilter for the existing F64 AABB query.

The tree only removes spatially distant pairs before the original lower-bound
test. Physics, primitive ownership, filtering, port weights and path admission
remain with SelfContactQuery. Dense neighborhoods are processed in bounded
chunks and overflow still rejects the whole snapshot.
"""
import numpy as np
import torch

from .self_contact_geometry import vertex_face, edge_edge
from .self_contact_rows import safe_norm


def query_pairs(query, positions, kind, capacity, *, radius_buckets=8):
    if type(radius_buckets) is not int or radius_buckets<1:
        raise ValueError('A positive integer sphere-radius bucket count is required')
    from scipy.spatial import cKDTree
    left = torch.arange(query.particle_count,device=positions.device)[:, None] if kind == 9 else query.edges
    right = query.triangles if kind == 9 else query.edges
    if not len(left) or not len(right):
        return torch.empty(0, 2, dtype=torch.long,device=positions.device)
    source, target = positions[left], positions[right]
    lower, upper = source.amin(1), source.amax(1)
    low, high = target.amin(1), target.amax(1)
    center_a, center_b = lower/2+upper/2, low/2+high/2
    # Enclose each ENTIRE AABB, including corners outside the primitive.
    # Primitive-only spheres would drop some harmless old AABB candidates
    # and would therefore change the neural relation snapshot.
    radius_a = safe_norm((upper-lower)/2)
    radius_b = safe_norm((high-low)/2)
    magnitude = float(positions.abs().max().clamp_min(1.))
    outward = 256*torch.finfo(positions.dtype).eps*(magnitude+query.query_radius+float(radius_b.max()))
    radius = query.query_radius+radius_a+radius_b.max()+outward
    if not bool(torch.isfinite(center_a).all() & torch.isfinite(center_b).all() & torch.isfinite(radius).all()):
        raise ValueError('Self-contact spatial query exceeded its finite arithmetic domain')
    # The conservative tree is host geometry indexing. Postfiltering, F64
    # feature geometry and all physics stay on the owning Torch device.
    host_centers=center_a.cpu().numpy();host_targets=center_b.cpu().numpy()
    host_radius_a=radius_a.cpu().numpy();host_radius_b=radius_b.cpu().numpy()
    positive=host_radius_b[host_radius_b>0]
    if radius_buckets==1 or not len(positive) or positive.min()==positive.max():
        labels=np.zeros(len(right),dtype=np.int64)
    else:
        cuts=np.geomspace(positive.min(),positive.max(),num=radius_buckets)
        labels=np.searchsorted(cuts,host_radius_b,side='left')
    trees=[]
    for label in np.unique(labels):
        ids=np.flatnonzero(labels==label)
        tree=cKDTree(host_targets[ids],compact_nodes=True,balanced_tree=True)
        # Every target retains an enclosing AABB sphere. Grouping only replaces
        # the global maximum radius by a conservative maximum in this tree.
        bound=query.query_radius+host_radius_a+float(host_radius_b[ids].max())+outward
        trees.append((tree,ids,bound))
    keys = query.filter_keys[0 if kind == 9 else 1]
    right_world = query.world[right[:, 0]]
    responsive=getattr(query,'responsive_particles',None)
    found = []
    for begin in range(0, len(left), query.chunk_size):
        end = min(begin+query.chunk_size, len(left))
        left_parts=[];right_parts=[]
        for tree,ids,bound in trees:
            neighbors=tree.query_ball_point(host_centers[begin:end],bound[begin:end],
                                           eps=0.,workers=1,return_sorted=True)
            sizes=np.fromiter(map(len,neighbors),dtype=np.int64,count=len(neighbors))
            if not int(sizes.sum()):continue
            left_parts.append(np.repeat(np.arange(begin,end,dtype=np.int64),sizes))
            right_parts.append(ids[np.concatenate(neighbors).astype(np.int64,copy=False)])
        if not left_parts:
            continue
        host_a=np.concatenate(left_parts);host_b=np.concatenate(right_parts)
        # Restore the original left-then-right order before any geometry or
        # capacity accounting, so neural snapshots and reductions stay exact.
        order=np.lexsort((host_b,host_a))
        a=torch.from_numpy(host_a[order]).to(positions.device)
        b=torch.from_numpy(host_b[order]).to(positions.device)
        mask = (query.world[left[a, 0]] == right_world[b]) | (query.world[left[a, 0]] < 0) | (right_world[b] < 0)
        mask &= ~(left[a, :, None] == right[b, None, :]).any((-1, -2))
        if responsive is not None:
            mask &= responsive[left[a]].any(-1) | responsive[right[b]].any(-1)
        if kind == 10:
            mask &= a < b
        if len(keys):
            pair_ids = a*len(right)+b
            index = torch.searchsorted(keys, pair_ids).clamp_max(len(keys)-1)
            mask &= keys[index] != pair_ids
        a, b = a[mask], b[mask]
        # Retain exactly the old outward-rounded Euclidean AABB rule after
        # prefiltering; closest-point distance is not used to exclude pairs.
        gap = torch.maximum(low[b]-upper[a], lower[a]-high[b]).clamp_min(0.)
        scale = torch.maximum(source[a].abs().amax((1, 2)), target[b].abs().amax((1, 2))).clamp_min(1.)
        mask = safe_norm(gap) <= query.query_radius+128*torch.finfo(positions.dtype).eps*scale
        pairs = torch.stack((a[mask], b[mask]), -1)
        if query.rest_radius and len(pairs):
            owners = query._owners(kind, pairs)
            geometry = (vertex_face if kind == 9 else edge_edge)(query.rest[owners])
            pairs = pairs[geometry.distance >= query.rest_radius]
        capacity -= len(pairs)
        if capacity < 0:
            raise ValueError('Self-contact candidate capacity exceeded; no partial snapshot is admissible')
        found.append(pairs)
    return torch.cat(found) if found else torch.empty(0, 2, dtype=torch.long,device=positions.device)
