"""Audit whether teacher macro endpoints fit the runtime collision envelope."""

from collections import defaultdict
import math

import numpy as np


def surface_travel(x0, q0, x1, q1, radii):
    """Runtime's translation + radius * shortest-angle travel, in meters."""
    q0, q1 = np.asarray(q0, dtype=np.float64), np.asarray(q1, dtype=np.float64)
    n0, n1 = np.linalg.norm(q0, axis=-1, keepdims=True), np.linalg.norm(q1, axis=-1, keepdims=True)
    if np.any(n0 < 1e-12) or np.any(n1 < 1e-12):
        raise ValueError('Nondegenerate recorded quaternions required for motion audit')
    q0, q1 = q0/n0, q1/n1
    vector = q0[..., 3:] * q1[..., :3] - q1[..., 3:] * q0[..., :3] - np.cross(q0[..., :3], q1[..., :3])
    scalar = np.abs((q0*q1).sum(-1))
    angle = 2*np.arctan2(np.linalg.norm(vector, axis=-1), scalar)
    return np.linalg.norm(np.asarray(x1)-np.asarray(x0), axis=-1) + np.asarray(radii)*angle


def audit_endpoint_motion(path, windows, budget):
    """Check every macro hop against the exact endpoint domain used at runtime.

    Endpoint feasibility is necessary, not sufficient for complete collision
    coverage between endpoints. Raw/native archives need their own adapter;
    never certify them from COM displacement alone.
    """
    if not math.isfinite(budget) or budget <= 0:
        raise ValueError('Positive finite query motion budget required')
    with np.load(path, allow_pickle=False) as archive:
        required = {'x', 'q', 'body_radius', 'movable'}
        if not required.issubset(archive.files):
            return {'supported': False, 'missing_fields': sorted(required-set(archive.files))}
        x, q = archive['x'], archive['q']
        radii, movable = archive['body_radius'], archive['movable'].astype(bool)
    if x.shape[:-1] != q.shape[:-1] or x.shape[-1] != 3 or q.shape[-1] != 4:
        raise ValueError('Recorded positions and quaternions have incompatible layouts')
    if radii.shape != (x.shape[1],) or movable.shape != radii.shape:
        raise ValueError('One physical radius and mobility mask per recorded owner required')
    if not np.isfinite(x).all() or not np.isfinite(q).all() or not np.isfinite(radii).all() or (radii < 0).any():
        raise ValueError('Finite states and nonnegative physical surface radii required')
    grouped = defaultdict(list)
    for window in windows:
        cursor = int(window['start'])
        for hop, (stride, duration) in enumerate(zip(window['strides'], window['durations'])):
            endpoint = cursor+int(stride)
            travel = surface_travel(x[cursor], q[cursor], x[endpoint], q[endpoint], radii)
            maximum = float(travel[movable].max()) if movable.any() else 0.
            grouped[round(float(duration), 12)].append({
                'start': cursor, 'endpoint': endpoint, 'window_start': int(window['start']),
                'hop': hop, 'maximum_surface_travel_m': maximum,
                'within_envelope': maximum <= budget+1e-10})
            cursor = endpoint
    all_rows = [row for rows in grouped.values() for row in rows]
    return {'supported': True, 'query_motion_budget_m': float(budget),
            'macro_hops': len(all_rows),
            'unreachable_macro_hops': sum(not row['within_envelope'] for row in all_rows),
            'maximum_teacher_surface_travel_m': max((row['maximum_surface_travel_m'] for row in all_rows), default=0.),
            'by_dt': [{'dt_s': dt, 'macro_hops': len(rows),
                       'unreachable_macro_hops': sum(not row['within_envelope'] for row in rows),
                       'maximum_teacher_surface_travel_m': max(row['maximum_surface_travel_m'] for row in rows)}
                      for dt, rows in sorted(grouped.items())],
            'violations': [row for row in all_rows if not row['within_envelope']],
            'scope': 'Exact endpoint travel condition; no continuous collision guarantee or solver convergence claim.'}
