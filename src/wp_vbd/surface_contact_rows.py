"""Prototype finite contact rows pulled back through weighted point ports.

No solver route, collision lifetime, path guard or mixed-joint admission is
provided here. Native mesh packets require an owned geometric force filter.
"""
from dataclasses import dataclass
import math
from types import SimpleNamespace

import torch

from . import math3d as so3
from .rigid_contact_rows import ContactContext, evaluate
from .native_array import native_tensor


@dataclass
class SurfaceMaterial:
    stiffness: torch.Tensor
    damping: torch.Tensor
    friction: torch.Tensor


def export_material(model, snapshot, wp, newton):
    allowed = [int(getattr(newton.GeoType, name)) for name in
               ('PLANE', 'SPHERE', 'BOX', 'CAPSULE', 'CYLINDER', 'CONE', 'ELLIPSOID')]
    kind = native_tensor(wp, model.shape_type)[snapshot.shape].long()
    mesh = torch.isin(kind, kind.new_tensor([int(newton.GeoType.MESH), int(newton.GeoType.CONVEX_MESH)]))
    analytic = torch.isin(kind, kind.new_tensor(allowed))
    needs_filter = bool(mesh.any() | (snapshot.rigid_primitive >= 0).any())
    if not bool((analytic | mesh).all()) or (needs_filter and snapshot.mesh_force_filter is None):
        raise ValueError('Mesh primitive force eligibility requires its separate filter')
    def shape(name): return native_tensor(wp, getattr(model, name))[snapshot.shape].double().clone()
    ke = .5*(shape('shape_material_ke')+model.soft_contact_ke)
    kd = .5*(shape('shape_material_kd')+model.soft_contact_kd)
    mu_shape = shape('shape_material_mu'); mu_soft = ke.new_tensor(model.soft_contact_mu)
    if any(not bool(torch.isfinite(value).all()) or bool((value < 0).any()) for value in (ke, kd, mu_shape, mu_soft)):
        raise ValueError('Invalid finite surface contact material')
    return SurfaceMaterial(ke, kd, (mu_shape*mu_soft).sqrt())


def begin(snapshot, start, body_com, inverse_mass_blocks, dt, material, friction_epsilon, *, material_start=None, tangent_support_scale=1.):
    if not math.isfinite(dt) or dt <= 0 or not math.isfinite(friction_epsilon) or friction_epsilon < 0:
        raise ValueError('Positive dt and finite friction regularization required')
    if not math.isfinite(tangent_support_scale) or tangent_support_scale<=0:
        raise ValueError('Finite positive numerical tangential support scale required')
    if inverse_mass_blocks.shape != (len(start.x), snapshot.ports.node_count, 6, 6):
        raise ValueError('Physical inverse mass/inertia blocks required')
    for value in (material.stiffness, material.damping, material.friction):
        if value.shape != (len(snapshot.shape),) or not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError('Finite nonnegative per-row contact material required')
    inverse = inverse_mass_blocks
    if not bool(torch.isfinite(inverse).all()): raise ValueError('Nonfinite inverse mass')
    scale = inverse.abs().amax((-1, -2)).clamp_min(1.)
    if bool(((inverse-inverse.transpose(-1, -2)).abs().amax((-1, -2)) > 1e-12*scale).any()):
        raise ValueError('Inverse mass must be symmetric')
    eigen, _ = so3.eigh_zero_safe(inverse)
    if bool((eigen < -1e-12*scale[..., None]).any()): raise ValueError('Inverse mass must be PSD')
    point, jacobian, _ = snapshot.geometry(start, body_com)
    relative_j = snapshot.ports.relative_jacobian(jacobian)
    delassus = torch.einsum('brsci,brsij,brsdj->brcd', relative_j, inverse[:, snapshot.ports.ids], relative_j)
    normal = snapshot.normal
    normal_support = torch.einsum('rc,brcd,rd->br', normal, delassus, normal)
    outer = normal[:, :, None]*normal[:, None, :]
    projector = torch.eye(3, device=normal.device, dtype=normal.dtype)-outer
    tangent = projector[None]@delassus@projector[None]
    tangent_eigen, _ = so3.eigh_zero_safe(tangent)
    maximum = tangent_eigen[..., -1]
    rho_n = torch.where((normal_support > 0) & (material.stiffness[None] > 0), 1/(dt**2*normal_support.clamp_min(1e-30)), 0.)
    rho_t = torch.where(maximum > 0, 1/(dt**2*maximum.clamp_min(1e-30)), 0.)
    rho_t = rho_t*tangent_support_scale
    delta = point[:, :, 1]-point[:, :, 0]
    c0 = -(delta-(delta*normal).sum(-1, keepdim=True)*normal)
    owner = snapshot.ports.ids[:, 0]
    velocity = so3.rotate(start.q[:, owner], snapshot.body_local_velocity[None])
    previous_point = point if material_start is None else snapshot.geometry(material_start, body_com,jacobian=False)[0]
    return SimpleNamespace(start_point=previous_point, rho_n=rho_n, rho_t=rho_t, c0_t=c0,
                           dual_t=torch.zeros_like(delta), material=material, dt=dt,
                           friction_epsilon=friction_epsilon, surface_velocity=velocity)


def rows(snapshot, state, body_com, context):
    point, jacobian, depth = snapshot.geometry(state, body_com)
    if not len(snapshot.shape):
        empty_metric = point.new_zeros(*depth.shape, 3, 3)
        return {'force': torch.zeros_like(point), 'torque': torch.zeros_like(point),
                'residual': point.new_zeros(len(point), snapshot.ports.node_count, 6),
                'metric_term': snapshot.ports.metric_term(jacobian.detach(), empty_metric),
                'normal_solve_metric': empty_metric, 'tangent_solve_metric': empty_metric,
                'tangent_force': point[:, :, 0], 'valid': torch.ones_like(depth, dtype=torch.bool),
                'point': point, 'jacobian': jacobian, 'raw_depth': depth,
                'normal_load': depth, 'damping_active': depth, 'active': depth.bool()}
    material = context.material
    eligible = torch.ones_like(depth, dtype=torch.bool) if snapshot.mesh_force_filter is None else snapshot.mesh_force_filter(state, body_com)
    stiffness = material.stiffness[None]*eligible
    damping_material = material.damping[None]*eligible
    load = stiffness*depth.clamp_min(0)
    normal = snapshot.normal
    dual = (context.dual_t+load[..., None]*normal)*eligible[..., None]
    local = point.new_zeros(*point.shape[:-1], 4); local[..., 3] = 1.
    zero = torch.zeros_like(point)
    epsilon = depth.new_full(depth.shape, context.friction_epsilon)
    constitutive = ContactContext(stiffness, context.rho_n*eligible, context.rho_t*eligible,
                                  damping_material, material.friction[None], dual,
                                  context.c0_t, epsilon, context.dt,
                                  compiled_rows=bool(getattr(context,'compiled_rows',False)))
    result = evaluate(point, local, context.start_point, local, zero, zero, zero,
                      normal, depth, context.surface_velocity, constitutive)
    residual = -snapshot.ports.pullback(jacobian, result['force'])
    displacement = point[:, :, 1]-context.start_point[:, :, 1]-point[:, :, 0]+context.start_point[:, :, 0]
    normal_displacement = (displacement*normal).sum(-1)
    active = (load > 0) & (stiffness > 0) & (context.rho_n > 0)
    damping = damping_material*(normal_displacement < 0)*active
    normal_metric = (stiffness+damping/context.dt)*active
    metric = normal_metric[..., None, None]*(normal[:, :, None]*normal[:, None, :])+result['tangent_solve_metric']
    # This is a frozen positive solve metric, as in the existing elastic and
    # rigid paths. Differentiating its eigenvectors is undefined at repeated
    # tangent eigenvalues. The physical residual above remains differentiable.
    term = snapshot.ports.metric_term(jacobian.detach(), metric.detach())
    result.update(residual=residual, metric_term=term, point=point, jacobian=jacobian,
                  raw_depth=depth, normal_load=load, damping_active=damping, active=active, force_eligible=eligible)
    return result


def merit(snapshot, state, body_com, context, frozen):
    point, _, depth = snapshot.geometry(state, body_com,jacobian=False)
    return _merit_from_geometry(point,depth,snapshot.normal,context,frozen)


def _merit_tensor(snapshot, state, body_com, context, frozen):
    # The enclosing candidate evaluation validates these exact state/COM
    # tensors before invoking the compiler; do not replace those checks.
    point,_=snapshot.ports._geometry_tensor(state,body_com,jacobian=False)
    distance=((point[:,:,1]-point[:,:,0])*snapshot.normal).sum(-1)
    depth=snapshot.radius+snapshot.shape_margin-distance
    return _merit_from_geometry(point,depth,snapshot.normal,context,frozen)


def _merit_from_geometry(point, depth, normal, context, frozen):
    dt = context.dt
    displacement = point[:, :, 1]-context.start_point[:, :, 1]-point[:, :, 0]+context.start_point[:, :, 0]
    normal_displacement = (displacement*normal).sum(-1)
    energy = .5*context.material.stiffness[None]*depth.clamp_min(0).square()
    energy = energy+.5*frozen['damping_active']/dt*normal_displacement.clamp_max(0).square()
    displacement = displacement+context.surface_velocity*dt
    tangent = -(displacement-(displacement*normal).sum(-1, keepdim=True)*normal)+context.c0_t
    limit = context.material.friction[None]*frozen['normal_load']
    def potential(value):
        norm = value.norm(dim=-1)
        return torch.where(norm <= limit, .5*norm.square(), limit*norm-.5*limit.square())
    energy = energy+(potential(context.dual_t+context.rho_t[..., None]*tangent)-potential(context.dual_t))/context.rho_t.clamp_min(1e-30)
    return energy*frozen.get('force_eligible', torch.ones_like(depth, dtype=torch.bool))
