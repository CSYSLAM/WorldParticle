"""Normal-energy majorants for a currently unloaded contact's frozen path.

This module supplies a scalar bound, not a contact solver or acceptance
rule. Callers must bound their own kinematic depth path and preserve all
motion/material checks. Active contact/friction/material energy is outside
this additional term. No model enables it merely by importing this module.
"""
import torch


@torch.no_grad()
def normal_activation_curvature(raw_depth, linear_change, angular_remainder, stabilization_shift, stiffness, *, compiled=False):
    """Return c with E_new(t) <= c*t**2/2 on the unloaded branch, 0<=t<=1.

    Required path: raw(t) <= raw(0)+t*linear_change+t**2*angular_remainder.
    The remainder must be nonnegative. Normal energy uses the existing
    piecewise stabilized depth: raw if raw<0, otherwise raw-shift. Include
    the activation jump when a negative stabilization shift is present.
    Rows already carrying normal load receive zero additional curvature.
    """
    raw,linear,remainder,shift,material=torch.broadcast_tensors(
        raw_depth,linear_change,angular_remainder,stabilization_shift,stiffness)
    if compiled and raw.is_cuda:
        from .compiled_contact import evaluate
        curvature,flags=evaluate(_curvature_tensor,raw,linear,remainder,shift,material)
        finite,nonnegative,result_finite=flags.cpu().tolist()
        if not finite:raise ValueError('Finite normal activation bound inputs required')
        if not nonnegative:raise ValueError('Nonnegative angular remainder and contact stiffness required')
        if not result_finite:raise ValueError('Normal activation bound overflowed')
        return curvature
    if any(not bool(torch.isfinite(v).all()) for v in (raw,linear,remainder,shift,material)):
        raise ValueError('Finite normal activation bound inputs required')
    if bool((remainder<0).any()) or bool((material<0).any()):
        raise ValueError('Nonnegative angular remainder and contact stiffness required')
    curvature=_curvature_value(raw,linear,remainder,shift,material)
    if not bool(torch.isfinite(curvature).all()):
        raise ValueError('Normal activation bound overflowed')
    return curvature


def _curvature_value(raw,linear,remainder,shift,material):
    initial=torch.where(raw<0,raw,raw-shift)
    eligible=(initial<=0)&(material>0)
    closing=linear+remainder
    reachable=raw+closing>=0
    endpoint=(raw+closing-shift).clamp_min(0)
    # If raw-shift>0 while raw<0, the constitutive value jumps at raw=0.
    # The earliest possible activation is t_a=-raw/closing. The energy/t²
    # maximum is then bounded by (-shift/t_a)², not just its value at t=1.
    jump=eligible&(raw<0)&(raw-shift>0)&reachable
    numerator=torch.where(jump,(-shift)*closing,0.)
    gap=torch.where(jump,-raw,1.)
    activation=numerator/gap
    amplitude=torch.where(eligible&reachable,torch.maximum(endpoint,activation),0.)
    curvature=material*amplitude.square()
    return curvature


def _curvature_tensor(raw,linear,remainder,shift,material):
    finite=(torch.isfinite(raw).all()&torch.isfinite(linear).all()&torch.isfinite(remainder).all()
            &torch.isfinite(shift).all()&torch.isfinite(material).all())
    nonnegative=(remainder>=0).all()&(material>=0).all()
    curvature=_curvature_value(raw,linear,remainder,shift,material)
    return curvature,torch.stack((finite,nonnegative,torch.isfinite(curvature).all()))
