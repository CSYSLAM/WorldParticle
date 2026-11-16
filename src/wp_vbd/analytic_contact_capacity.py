"""Count a conservative primitive witness bound before allocating contacts.

This is only a geometry/capacity calculation. It does not drop broad-phase
pairs, alter shape gaps, move bodies or construct a dynamics solver.
"""
import numpy as np
import warp as wp


@wp.kernel(enable_backward=False)
def _count_expanded_aabb_pairs(lower:wp.array[wp.vec3],upper:wp.array[wp.vec3],
                              count:wp.array[wp.int64],invalid:wp.array[wp.int32]):
    i=wp.tid()
    lo=lower[i];hi=upper[i]
    for axis in range(3):
        if not wp.isfinite(lo[axis]) or not wp.isfinite(hi[axis]) or lo[axis]>hi[axis]:
            wp.atomic_add(invalid,0,1)
            return
    local=wp.int64(0)
    for j in range(i+1,lower.shape[0]):
        other_lo=lower[j];other_hi=upper[j]
        if (lo[0]<=other_hi[0] and hi[0]>=other_lo[0]
            and lo[1]<=other_hi[1] and hi[1]>=other_lo[1]
            and lo[2]<=other_hi[2] and hi[2]>=other_lo[2]):local+=wp.int64(1)
    wp.atomic_add(count,0,local)


def overlapping_pair_count(lower,upper,*,chunk=256):
    lower=np.asarray(lower);upper=np.asarray(upper)
    if lower.shape!=upper.shape or lower.ndim!=2 or lower.shape[1]!=3:
        raise ValueError('One lower/upper AABB per shape required')
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower>upper):
        raise ValueError('Finite ordered AABBs required')
    if type(chunk) is not int or chunk<1:raise ValueError('Positive AABB count chunk required')
    count=0;n=len(lower);columns=np.arange(n)[None]
    for begin in range(0,n,chunk):
        end=min(n,begin+chunk)
        # Native AABBs already contain margin + effective detection gap.
        # Inclusive comparisons match the pinned zero-cutoff broad phase.
        overlap=(lower[begin:end,None]<=upper[None]).all(-1)
        overlap&=(upper[begin:end,None]>=lower[None]).all(-1)
        overlap&=np.arange(begin,end)[:,None]<columns
        count+=int(np.count_nonzero(overlap))
    return count


class AnalyticAabbCapacity:
    def __init__(self,model,wp,newton):
        self.model=model;self.wp=wp
        allowed=[int(getattr(newton.GeoType,name)) for name in ('CAPSULE','SPHERE','PLANE')]
        if not np.isin(model.shape_type.numpy(),allowed).all():
            raise ValueError('Two-witness capacity proof admits only capsule/sphere/plane geometry')
        n=model.shape_count;device=model.device
        self.lower=wp.zeros(n,dtype=wp.vec3,device=device);self.upper=wp.zeros(n,dtype=wp.vec3,device=device)
        self.geometry=wp.zeros(n,dtype=wp.vec4,device=device);self.transforms=wp.zeros(n,dtype=wp.transform,device=device)
        self.counters=wp.zeros(1,dtype=wp.int32,device=device)
        self.generation=wp.zeros(1,dtype=wp.int32,device=device)
        self.pair_counter=wp.zeros(1,dtype=wp.int32,device=device)
        self.overlaps=wp.zeros(1,dtype=wp.int64,device=device)
        self.invalid=wp.zeros(1,dtype=wp.int32,device=device)

    def measure(self,native_state):
        from newton._src.sim.collide import compute_shape_aabbs
        m=self.model;wp=self.wp
        wp.launch(compute_shape_aabbs,dim=m.shape_count,inputs=[native_state.body_q,
            m.shape_transform,m.shape_body,m.shape_type,m.shape_scale,m.shape_collision_radius,
            m.shape_source_ptr,m.shape_margin,m.shape_gap,m.shape_collision_aabb_lower,
            m.shape_collision_aabb_upper,self.counters,self.generation,self.pair_counter,1],
            outputs=[self.lower,self.upper,self.geometry,self.transforms],device=m.device,record_tape=False)
        self.overlaps.zero_();self.invalid.zero_()
        wp.launch(_count_expanded_aabb_pairs,dim=m.shape_count,inputs=[self.lower,self.upper,self.overlaps,self.invalid],
                  device=m.device,record_tape=False)
        if int(self.invalid.numpy()[0]):raise ValueError('Finite ordered native AABBs required')
        pairs=int(self.overlaps.numpy()[0])
        return {'overlapping_aabb_pairs_before_filters':pairs,'primitive_witness_bound':2*pairs,
                'shape_count':m.shape_count,
                'policy':'exact_pinned_expanded_aabbs_inclusive_overlap_all_pairs_without_filtering_times_two'}
