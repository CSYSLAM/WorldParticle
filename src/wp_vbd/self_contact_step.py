"""Analytic particle step caps inside the frozen separating-plane cone."""
import math

import torch

from .self_contact_geometry import validate


@torch.no_grad()
def separating_plane_cap(geometry, current, displacement, *, fraction=.9):
    """Bound every affine vertex-pair support gap, without collision repair.

    Each primitive is convex and all particle paths are linear. Keeping all
    pairwise projections positive along the query's separating axis proves
    separation of the complete features. The existing whole-path predicate
    remains authoritative after this proposal scaling.
    """
    validate(current)
    if displacement.shape!=current.shape or displacement.dtype!=current.dtype or displacement.device!=current.device:
        raise ValueError('Matched F64 particle displacement required')
    if current.shape!=geometry.reference.shape or current.ndim!=3:
        raise ValueError('One [rows,4,3] primitive batch required')
    if not math.isfinite(fraction) or not 0<fraction<1:
        raise ValueError('Strict separating-plane fraction required')
    finite=torch.isfinite(displacement).all((-1,-2))
    axis=geometry.separating_axis
    point=torch.einsum('rvi,ri->rv',current,axis)
    rate=torch.einsum('rvi,ri->rv',displacement,axis)
    split=3 if geometry.kind==9 else 2
    gap=point[:,None,split:]-point[:,:split,None]
    derivative=rate[:,None,split:]-rate[:,:split,None]
    # Bound the magnitude of every point on every proposed prefix. Extra
    # outward slack covers the dot products at the eventual accepted point.
    magnitude=(current.abs()+displacement.abs()).amax((-1,-2)).clamp_min(1.)
    slack=128*torch.finfo(current.dtype).eps*magnitude
    lower=gap-slack[:,None,None]
    valid=finite&torch.isfinite(lower).all((1,2))&torch.isfinite(derivative).all((1,2))
    valid&=(geometry.separation_lower_bound>slack)&(lower>0).all((1,2))
    closing=derivative<0
    denominator=torch.where(closing,-derivative,torch.ones_like(derivative))
    bound=torch.where(closing,fraction*lower/denominator,torch.ones_like(lower))
    cap=bound.amin((1,2)).clamp(0.,1.)
    valid&=torch.isfinite(cap)&(cap>0)
    return torch.where(valid,cap,0.),valid


@torch.no_grad()
def proposal_scale(snapshot, state, update, *, fraction=.9):
    """One conservative scene-wide scale; no added line-search iterations."""
    if len(state.x)!=1 or update.shape!=(*state.x.shape[:2],6):
        raise ValueError('Matched one-scene generalized proposal required')
    scale=state.x.new_ones(1);valid=torch.ones(1,device=state.x.device,dtype=torch.bool)
    limited=0;recentered=0
    for ids,geometry in zip(snapshot.owners,snapshot.geometry):
        if not len(ids):continue
        cap,good=separating_plane_cap(geometry,state.x[0,ids],update[0,ids,:3],fraction=fraction)
        if getattr(snapshot,'path_subdivision_depth',0):
            # A refined physical path can be valid after the original query
            # axis ceases to separate. Recenter only the proposal cone; the
            # full accepted path is still certified against the old state.
            bad=torch.where(~good)[0]
            if len(bad):
                recentered+=len(bad)
                from .self_contact_geometry import vertex_face,edge_edge
                current=state.x[0,ids[bad]]
                refreshed=(vertex_face if geometry.kind==9 else edge_edge)(current)
                local,valid_local=separating_plane_cap(refreshed,current,update[0,ids[bad],:3],fraction=fraction)
                cap=cap.index_copy(0,bad,local);good=good.index_copy(0,bad,valid_local)
        scale=torch.minimum(scale,cap.amin().reshape(1));valid&=good.all()
        limited+=int((cap<1).sum())
    return scale[:,None].expand(-1,state.x.shape[1]),valid,{
        'scale':float(scale[0]),'limited_feature_pairs':limited,'fraction':fraction,
        'recentered_start_planes':recentered,
        'all_start_planes_valid':bool(valid.all()),'scope':'One common proposal scale; existing candidate path checks still run'}
