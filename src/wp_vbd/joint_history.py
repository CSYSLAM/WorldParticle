"""Functional joint auxiliary laws for the pinned Newton compliant-ALM path.

These are local constitutive/state updates, not a dynamics solver. Numerical
duals, physical Dahl memory, and angle winding have distinct lifetimes. Every
function returns new tensors so discarding a trial leaves accepted state intact.
"""

from dataclasses import dataclass

import torch


def compliant_coefficients(material, rho):
    """Stable s, k_eff, a without forming the potentially overflowing K+rho."""
    active = (material > 0) & (rho > 0)
    large = material >= rho
    ratio = torch.where(large, rho / material.clamp_min(torch.finfo(material.dtype).tiny), material / rho.clamp_min(torch.finfo(rho.dtype).tiny))
    inverse = 1 / (1 + ratio)
    s = torch.where(large, inverse, ratio * inverse)
    a = torch.where(large, ratio * inverse, inverse)
    effective = torch.where(large, rho, material) * inverse
    return torch.where(active, s, 0.), torch.where(active, effective, 0.), torch.where(active, a, 1.)


def compliant_dual_step(dual, residual, material, rho):
    s, effective, a = compliant_coefficients(material, rho)
    return torch.where(material >= rho, dual + effective * residual - a * dual, s * dual + effective * residual)


def compliant_force_terms(material, rho, dual):
    s, effective, _ = compliant_coefficients(material, rho)
    return effective, s * dual


def unwrap_hinge(angle, reference):
    turns = (reference - angle) / (2 * torch.pi)
    # Warp uses C round(): exact half turns round away from zero. Torch's
    # default round() uses ties-to-even and selects a different branch there.
    nearest = torch.sign(turns) * torch.floor(turns.abs() + .5)
    return angle + 2 * torch.pi * nearest


def rod_strain_increment(current, previous):
    delta = current - previous
    twist=delta[...,2:]
    # The inputs are principal angles. Match the pinned bounded subtraction
    # instead of adding trigonometric roundoff that damping amplifies by 1/h.
    twist=torch.where(twist>torch.pi,twist-2*torch.pi,torch.where(twist<-torch.pi,twist+2*torch.pi,twist))
    return torch.cat((delta[..., :2],twist),-1)


@dataclass(frozen=True)
class DahlMemory:
    kappa: torch.Tensor
    increment: torch.Tensor
    stress: torch.Tensor

    def snapshot(self):
        return DahlMemory(*(value.clone() for value in (self.kappa, self.increment, self.stress)))


def dahl_axis_step(delta, previous_delta, previous_stress, envelope, tau):
    direction = torch.where(delta > 1e-6, 1., torch.where(delta < -1e-6, -1., torch.where(previous_delta >= 0, 1., -1.)))
    decay = torch.exp(-direction * delta / tau.clamp_min(torch.finfo(tau.dtype).tiny))
    return direction * envelope * (1 - decay) + previous_stress * decay, direction


def begin_dahl(memory, current, stiffness, eps_max, tau, enabled, rebaseline):
    """Freeze one step's stress/tangent; reset applies only to enabled rods."""
    reset = (enabled & rebaseline)[..., None]
    frozen = DahlMemory(torch.where(reset, current, memory.kappa), torch.where(reset, 0., memory.increment), torch.where(reset, 0., memory.stress))
    delta = rod_strain_increment(current, frozen.kappa)
    envelope = stiffness * eps_max[..., None]
    active = enabled[..., None] & (eps_max[..., None] > 0) & (tau[..., None] > 0) & (envelope > 0)
    safe_tau = torch.where(tau > 0, tau, 1.)[..., None]
    stress, direction = dahl_axis_step(torch.where(active, delta, 0.), torch.where(active, frozen.increment, 0.), torch.where(active, frozen.stress, 0.), torch.where(active, envelope, 0.), safe_tau)
    stress = torch.maximum(-envelope, torch.minimum(envelope, stress))
    tangent = ((envelope - direction * stress) / (safe_tau + delta.abs())).clamp_min(0)
    return frozen, torch.where(active, stress, 0.), torch.where(active, tangent, 0.)


def commit_dahl(frozen, final, stiffness, eps_max, tau, enabled):
    """Evaluate physical memory exactly once after an accepted physical step.

    The pinned finalizer deliberately does not clamp stress here. Its small
    direction deadband can produce a tiny envelope overshoot; begin_dahl clamps
    the frozen force on the following step. Preserve that distinction.
    """
    delta = rod_strain_increment(final, frozen.kappa)
    envelope = stiffness * eps_max[..., None]
    exists = enabled & (stiffness > 0).any(-1)
    friction = exists & (eps_max > 0) & (tau > 0)
    active = friction[..., None] & (envelope > 0)
    safe_tau = torch.where(tau > 0, tau, 1.)[..., None]
    stress, _ = dahl_axis_step(torch.where(active, delta, 0.), torch.where(active, frozen.increment, 0.), torch.where(active, frozen.stress, 0.), torch.where(active, envelope, 0.), safe_tau)
    increment = torch.where(friction[..., None], torch.where(active, delta, 0.), delta)
    return DahlMemory(final.clone(), torch.where(exists[..., None], increment, 0.), torch.where(active, stress, 0.))
