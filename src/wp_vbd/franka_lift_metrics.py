"""Saved-state lift evidence that separates COM motion from whole-body lift."""
import numpy as np


def table_height(model,newton):
    kinds=model.shape_type.numpy();owners=model.shape_body.numpy()
    ids=np.flatnonzero((kinds==int(newton.GeoType.BOX))&(owners<0))
    if len(ids)!=1:raise ValueError('One authored static table box required for Franka lift audit')
    pose=model.shape_transform.numpy()[ids[0]];scale=model.shape_scale.numpy()[ids[0]]
    if not np.allclose(pose[3:],(0.,0.,0.,1.),rtol=0,atol=1e-7):
        raise ValueError('This saved-state table audit requires the authored axis-aligned table')
    return float(pose[2]+scale[2])


def whole_object_lift(positions,times,height,*,minimum_clearance=.005):
    if positions.ndim!=3 or positions.shape[0]!=len(times) or positions.shape[-1]!=3:
        raise ValueError('Complete saved material positions and timestamps required')
    clearance=positions[...,2].min(1)-height
    above=clearance>minimum_clearance
    longest=0.;first=None
    for i,active in enumerate(above):
        if active:
            if first is None:first=i
            longest=max(longest,float(times[i]-times[first]))
        else:first=None
    return {'table_top_height_m':height,'whole_material_peak_minimum_table_clearance_m':float(clearance.max()),
        'whole_object_lift_above_5mm_observed':bool(above.any()),
        'whole_object_lift_sustained_250ms':bool(longest>=.25-1e-9),
        'whole_object_longest_clearance_interval_s':longest,
        'whole_object_clearance_threshold_m':minimum_clearance,
        'scope':'All saved material vertices above the authored table top; COM elevation alone is not whole-object lift. '
                'Discrete recording samples, not a continuous collision or grasp certificate.'}
