"""Auditable rod rows with compliant ALM, damping and frozen Dahl forces.

No complete solver is imported. A caller owns the physical step's previous
poses, numerical trial duals and exactly-once material-memory commit.
"""

from dataclasses import dataclass, replace

import torch

from . import math3d as so3
from .joint_history import compliant_coefficients, compliant_dual_step, rod_strain_increment
from .mixed_physics import rod_measure


def rotate_pinned(q, vector):
    """Match Warp's polynomial evaluation on FP32 normalized quaternions.

    Equivalent unit-quaternion formulae differ at the last bits. A stiff row
    amplifies that difference, so retain the pinned geometry convention here.
    """
    v, vector = torch.broadcast_tensors(q[..., :3], vector)
    c = 2 * q[..., 3:] * q[..., 3:] - 1
    d = 2 * (v * vector).sum(-1, keepdim=True)
    return vector * c + v * d + torch.linalg.cross(v, vector) * q[..., 3:] * 2


def measure_pinned(parent, child):
    """Finite curvature and Bishop spin in the pinned material convention.

    The explicit material axes and inverse-rotation polynomial matter for
    curved rods: a last-bit strain difference is amplified by damping / h.
    The caller admits only the nonsingular tangent domain (cosine > -.9).
    """
    def axes(q):
        x, y, z, w = q.unbind(-1)
        tangent = torch.stack((2*(x*z+y*w), 2*(y*z-x*w), 1-2*(x*x+y*y)), -1)
        normal = torch.stack((1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)), -1)
        return tangent, normal
    t0, m0 = axes(parent)
    t1, m1 = axes(child)
    denominator = (1+(t0*t1).sum(-1).clamp(-1, 1)).clamp_min(1e-8)
    curvature = (2/denominator)[..., None]*torch.linalg.cross(t0, t1)
    curvature = curvature*(20/curvature.square().sum(-1, keepdim=True).clamp_min(1e-20).sqrt()).clamp_max(1)
    transported = so3.normalize(m0-((m0*t1).sum(-1)/denominator)[..., None]*(t0+t1))
    twist = torch.atan2((t1*torch.linalg.cross(transported, m1)).sum(-1), (transported*m1).sum(-1))
    bend = rotate_pinned(so3.conjugate(parent), curvature)
    return torch.cat((bend[..., :2], twist[..., None]), -1)


def coordinates(position, rotation, anchor_p, anchor_c, anchor_qp, anchor_qc, rest_bend, rest_twist, *, pinned=False):
    qa = so3.multiply(rotation[..., 0, :], anchor_qp)
    qb = so3.multiply(rotation[..., 1, :], anchor_qc)
    pa = position[..., 0, :] + rotate_pinned(rotation[..., 0, :], anchor_p)
    pb = position[..., 1, :] + rotate_pinned(rotation[..., 1, :], anchor_c)
    gap = rotate_pinned(so3.conjugate(qa), pb - pa)
    measure = measure_pinned(qa, qb) if pinned else rod_measure(qa, qb)
    twist = measure[..., 2:] - rest_twist[..., None]
    angular = torch.cat((measure[..., :2] - rest_bend[..., :2], torch.atan2(torch.sin(twist), torch.cos(twist))), -1)
    return torch.cat((gap, angular), -1)


def automatic_rho(stiffness, inverse_mass, inverse_inertia, dt):
    linear_mobility = inverse_mass.sum(-1)
    angular_mobility = (inverse_inertia.diagonal(dim1=-2, dim2=-1).sum(-1) / 3 * (inverse_mass > 0)).sum(-1)
    linear = torch.where(linear_mobility > 0, 1 / (linear_mobility.clamp_min(1e-30) * dt**2), 0.)
    angular = torch.where(angular_mobility > 0, 1 / (angular_mobility.clamp_min(1e-30) * dt**2), 0.)
    support = torch.cat((linear[..., None].expand(*linear.shape, 3), angular[..., None].expand(*angular.shape, 3)), -1)
    return torch.where(stiffness > 0, torch.maximum(support, 9 * stiffness), 0.)


@dataclass(frozen=True)
class RodContext:
    stiffness: torch.Tensor
    rho: torch.Tensor
    dual: torch.Tensor
    initial: torch.Tensor
    previous: torch.Tensor
    damping: torch.Tensor
    sigma: torch.Tensor
    friction_metric: torch.Tensor
    dt: float
    alpha: float = 0.
    geometry_fp64: bool = False

    def numeric_trial(self, current):
        return replace(self, dual=compliant_dual_step(self.dual, current - self.alpha * self.initial, self.stiffness, self.rho))

    def terms(self, current, *, per_channel_energy=False):
        s, effective, _ = compliant_coefficients(self.stiffness, self.rho)
        linear = s * self.dual - self.alpha * effective * self.initial
        linear = linear + torch.cat((torch.zeros_like(self.sigma), self.sigma), -1)
        delta = torch.cat((current[..., :3] - self.previous[..., :3], rod_strain_increment(current[..., 3:], self.previous[..., 3:])), -1)
        stress = effective * current + linear + self.damping / self.dt * delta
        metric = effective + self.damping / self.dt + torch.cat((torch.zeros_like(self.friction_metric), self.friction_metric), -1)
        energy = .5 * effective * current.square() + linear * current + .5 * self.damping / self.dt * delta.square()
        if not per_channel_energy:energy = energy.sum(-1)
        return stress, metric, energy


def evaluate(position, rotation, body_com, anchor_p, anchor_c, anchor_qp, anchor_qc, rest_bend, rest_twist, context):
    """Return +energy-gradient residuals, PSD GN self blocks and row Jacobians.

    Position stores the body origin; the tangent translation is at the COM.
    Keep the authored origin rather than rounding COM and subtracting it back:
    tens of nanometres can become hundredths of a newton at stiffness 1e6.
    Supports independent rows with arbitrary leading dimensions. The admitted
    geometric domain excludes near-antiparallel material tangents, where the
    pinned reference changes to a capped/fallback branch.
    """
    enabled = torch.is_grad_enabled()
    with torch.enable_grad():
        delta = position.new_zeros(*position.shape[:-1], 6, requires_grad=True)
        q = so3.multiply(so3.exp(delta[..., 3:]), rotation)
        x = position + delta[..., :3] + (rotate_pinned(rotation, body_com) - rotate_pinned(q, body_com))
        strain = coordinates(x, q, anchor_p, anchor_c, anchor_qp, anchor_qc, rest_bend, rest_twist, pinned=context.geometry_fp64)
        if enabled:
            jacobian = torch.stack([torch.autograd.grad(strain[..., column].sum(), delta, retain_graph=True, create_graph=True)[0] for column in range(6)], -3)
        else:
            # Independent strain channels share one reverse traversal during
            # inference. Higher-order training keeps its original graph.
            basis=torch.eye(6,device=strain.device,dtype=strain.dtype).reshape(6,*([1]*(strain.ndim-1)),6).expand(6,*strain.shape)
            jacobian=torch.autograd.grad(strain,delta,basis,is_grads_batched=True)[0].movedim(0,-3)
    if not enabled:
        strain, jacobian = strain.detach(), jacobian.detach()
    stress, weights, energy = context.terms(strain)
    residual = torch.einsum('...csi,...c->...si', jacobian, stress)
    blocks = torch.einsum('...csi,...c,...csj->...sij', jacobian, weights, jacobian)
    qa = so3.multiply(rotation[..., 0, :], anchor_qp)
    qb = so3.multiply(rotation[..., 1, :], anchor_qc)
    cosine = (so3.matrix(qa)[..., 2] * so3.matrix(qb)[..., 2]).sum(-1)
    return {'residual': residual, 'blocks': blocks, 'coordinates': strain, 'jacobian': jacobian, 'energy': energy, 'in_domain': cosine > -.9}
