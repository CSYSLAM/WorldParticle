"""Restore ideal mimic coordinates after a Float32 native-state round trip.

The correction is limited by source precision and scene size. Material
particles, velocities, forces and physical acceptance rules are unchanged.
"""
import torch

from . import math3d as so3
from .mimic_constraint import rows, retract


@torch.no_grad()
def project(scene, state):
    spec=getattr(scene,'mimic_spec',None)
    if spec is None or not spec.pairs.numel():return state, None
    before=rows(scene,state)
    if not bool(before.valid.all()):raise ValueError('Invalid native mimic chart')
    corrected,valid=retract(scene,state)
    translation=(corrected.x-state.x).norm(dim=-1).amax()
    angles=so3.log(so3.multiply(corrected.q,so3.conjugate(state.q))).norm(dim=-1)
    rotation=angles.amax()
    surface_travel=((corrected.x-state.x).norm(dim=-1)+scene.body_radius*angles).amax()
    epsilon=torch.finfo(torch.float32).eps
    position_scale=max(1.,float(state.x.abs().amax()))
    position_bound=min(8*epsilon*position_scale,1e-5*float(scene.length))
    # Coordinate rounding can amplify into angle corrections through a
    # short joint lever. Bound their physical surface travel as well.
    angular_bound=1e-4
    if not bool(valid.all()) or not bool(torch.isfinite(translation)&torch.isfinite(rotation)):
        raise ValueError('Native mimic roundoff projection failed')
    if float(surface_travel)>position_bound or float(rotation)>angular_bound:
        raise ValueError('Native mimic violation exceeds the Float32 roundoff bound')
    return corrected,dict(maximum_position_correction=float(translation),maximum_angular_correction=float(rotation),maximum_surface_travel=float(surface_travel),
                          position_bound=position_bound,angular_bound=angular_bound,
                          maximum_constraint_error_before=float(before.value.abs().amax()),
                          maximum_constraint_error_after=float(rows(scene,corrected).value.abs().amax()))
