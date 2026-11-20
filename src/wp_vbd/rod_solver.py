"""Shared-network rod dynamics with explicit, transactional physical history.

Only the audited no-contact FREE/ROD contract enters here. Network parameters
belong to MixedSolver and are shared with particle, body and attachment paths.
There is no VBD import, convergence loop or reference fallback.
"""

from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from . import math3d as so3
from .data import State
from .joint_history import DahlMemory, begin_dahl, commit_dahl
from .mixed_physics import mass_whitening, predictor
from .rod_rows import RodContext, automatic_rho, coordinates, evaluate, rotate_pinned


@dataclass
class Plan:
    ids: torch.Tensor
    root: int
    basis: torch.Tensor
    columns: torch.Tensor
    kind: str = 'static'
    frame_ids: torch.Tensor | None = None
    frame_rest: torch.Tensor | None = None
    cycle: dict | None = None
    boundary_basis: torch.Tensor | None = None
    chain: dict | None = None


def prepare(scene):
    scene.rod_enabled_dahl = bool(((scene.rod_dahl_eps > 0) & (scene.rod_dahl_tau > 0)).any())
    scene.rod_plans = []
    scene.rod_material_plans = []
    scene.rod_curvature_plans = []
    scene.rod_flexure_plans = []
    scene.rod_cycle_plans = []
    cycle_candidates=[]
    positions = (scene.rest_x + rotate_pinned(scene.rest_q, scene.body_com)).cpu().numpy()
    for component in scene.object_entities:
        curvature_start=len(scene.rod_curvature_plans)
        ids = component.cpu().numpy()
        mobile = scene.movable[component].cpu().numpy()
        fixed = ids[~mobile]
        root = int(fixed[0] if len(fixed) else ids[0])
        neighbors = {int(i): [] for i in ids}
        for a, b in scene.joints.cpu().tolist():
            if a in neighbors and b in neighbors:
                neighbors[a].append(b); neighbors[b].append(a)
        chain = all(len(v) <= 2 for v in neighbors.values())
        order = []
        if chain:
            endpoint = next((int(i) for i in fixed if len(neighbors[int(i)]) <= 1), next((int(i) for i in ids if len(neighbors[int(i)]) <= 1), root))
            previous, current = -1, endpoint
            while current >= 0 and current not in order:
                order.append(current)
                following = [i for i in neighbors[current] if i != previous]
                previous, current = current, following[0] if following else -1
            chain = len(order) == len(ids)
        basis = np.zeros((len(ids), 6, 6), np.float32)
        boundary_basis = None
        if chain and len(fixed) and len(ids) >= 3:
            points = positions[order]
            arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=-1))]
            length = max(float(arc[-1]), .01); s = arc / length
            tangent = points[-1] - points[0]; tangent /= max(np.linalg.norm(tangent), 1e-12)
            normal = np.eye(3)[int(np.abs(tangent).argmin())]
            normal -= np.dot(normal, tangent) * tangent; normal /= np.linalg.norm(normal)
            other = np.cross(tangent, normal)
            both = len(fixed) > 1
            for power in (2, 3):
                f = s**power * ((1 - s)**2 if both else 1)
                derivative = power * s**(power - 1) * ((1 - s)**2 if both else 1)
                if both: derivative -= 2 * s**power * (1 - s)
                for axis, direction in enumerate((normal, other)):
                    column = (power - 2) * 2 + axis
                    for row, entity in enumerate(order):
                        local = int(np.where(ids == entity)[0][0])
                        basis[local, :3, column] = length * f[row] * direction
                        basis[local, 3:, column] = derivative[row] * np.cross(tangent, direction)
            shape = s * (1 - s) if both else s
            for row, entity in enumerate(order):
                local = int(np.where(ids == entity)[0][0])
                basis[local, :3, 4] = length * shape[row] * tangent
                basis[local, 3:, 5] = shape[row] * tangent
            if both:
                # A changed prescribed endpoint requires an affine interior
                # lift. Restrict linear fields to free DOFs, rather than
                # forcing their continuous extension to vanish at both ends.
                boundary_basis = basis.copy()
                for row, entity in enumerate(order):
                    local = int(np.where(ids == entity)[0][0])
                    boundary_basis[local, :3, 4] = length * s[row] * tangent
                    boundary_basis[local, 3:, 5] = s[row] * tangent
        else:
            center = positions[ids].mean(0)
            for column in range(3):
                axis = np.eye(3)[column]
                basis[:, :3, column] = axis
                basis[:, :3, 3 + column] = np.cross(axis, positions[ids] - center)
                basis[:, 3:, 3 + column] = axis
        basis *= scene.dof[component].cpu().numpy()[..., None]
        if boundary_basis is not None:boundary_basis *= scene.dof[component].cpu().numpy()[..., None]
        flat = basis.reshape(-1, 6).astype(np.float64)
        retained = []
        for column in range(6):
            if np.linalg.matrix_rank(flat[:, retained + [column]], tol=1e-9) > len(retained):retained.append(column)
        if retained:
            kind='rigid' if not (chain and len(fixed) and len(ids)>=3) else 'static'
            plan=Plan(component, root, torch.as_tensor(basis, device=scene.device), torch.tensor(retained, device=scene.device),kind,boundary_basis=None if boundary_basis is None else torch.as_tensor(boundary_basis,device=scene.device))
            if kind=='static':
                from .rod_chain import description
                plan.chain=description(component,order,scene.joints.cpu().tolist(),fixed,s,scene.device)
            scene.rod_plans.append(plan)
            if kind=='static' and len(ids)>=8:
                # Three scalar nodal fields follow each body's MATERIAL
                # tangent. A middle port resolves a curvature barrier; two
                # endpoint ports propagate either boundary's prescribed spin.
                local_spin=np.zeros((len(ids),6,3),np.float32)
                middle=.5
                hat=np.minimum(s/middle,(1-s)/(1-middle))
                for row,entity in enumerate(order):
                    local=int(np.where(ids==entity)[0][0])
                    local_spin[local,5]=[1-s[row],s[row],hat[row]]
                local_spin*=scene.dof[component].cpu().numpy()[...,None]
                scene.rod_material_plans.append(Plan(component,root,torch.as_tensor(local_spin,device=scene.device),torch.arange(3,device=scene.device),'twist',chain=plan.chain))
                scene.rod_material_plans.append(replace(plan,columns=torch.tensor([i for i in retained if i!=5],device=scene.device)))
                from .rod_chain import curvature_fields
                scalar=curvature_fields(s,scene.rest_q[order].cpu().numpy(),scene.movable[order].cpu().numpy())
                enriched=np.zeros((len(ids),6,scalar.shape[-1]),np.float32)
                for row,entity in enumerate(order):enriched[int(np.where(ids==entity)[0][0]),5]=scalar[row]
                scene.rod_curvature_plans.append(Plan(component,root,torch.as_tensor(enriched,device=scene.device),torch.arange(scalar.shape[-1],device=scene.device),'twist',chain=plan.chain))
                scene.rod_curvature_plans.append(scene.rod_material_plans[-1])
            else:
                scene.rod_material_plans.append(plan);scene.rod_curvature_plans.append(plan)
        scene.rod_flexure_plans.extend(scene.rod_curvature_plans[curvature_start:])
        if retained and kind=='static' and len(ids)>=8:
            from .rod_chain import curvature_flexure
            defect=curvature_flexure(s,scene.rest_q[order].cpu().numpy(),scene.movable[order].cpu().numpy())
            if defect is not None:
                flexure=np.zeros((len(ids),6,2),np.float32)
                for row,entity in enumerate(order):
                    local=int(np.where(ids==entity)[0][0])
                    flexure[local,3,0]=defect[row];flexure[local,4,1]=defect[row]
                scene.rod_flexure_plans.append(Plan(component,root,torch.as_tensor(flexure,device=scene.device),torch.arange(2,device=scene.device),'flexure',chain=plan.chain))
        if chain and not len(fixed) and len(ids)>=8 and all(len(v)==2 for v in neighbors.values()):cycle_candidates.append((component,order))
    if cycle_candidates:
        from .rod_cycle import cyclic_modes
        geometry=scene.objective_scene
        with torch.enable_grad():
            q0=geometry.rest_q[scene.joints][None];x0=geometry.rest_x[scene.joints][None];com=geometry.body_com[scene.joints]
            delta=torch.zeros(*x0.shape[:-1],6,dtype=torch.float64,device=scene.device,requires_grad=True)
            q=so3.multiply(so3.exp(delta[...,3:]),q0)
            x=x0+delta[...,:3]+rotate_pinned(q0,com)-rotate_pinned(q,com)
            c=coordinates(x,q,geometry.anchor_p,geometry.anchor_c,geometry.anchor_qp,geometry.anchor_qc,geometry.rod_rest_bend,geometry.rod_rest_twist)
            jacobian=torch.stack([torch.autograd.grad(c[...,axis].sum(),delta,retain_graph=True)[0] for axis in range(3)],-3)[0].detach().cpu().numpy()
        pairs=scene.joints.cpu().numpy();points=(geometry.rest_x+rotate_pinned(geometry.rest_q,geometry.body_com)).cpu().numpy()
        scene.rod_cycle_shear_errors=[]
        for component,order in cycle_candidates:
            basis,markers,frame,error,corrections=cyclic_modes(points,order,pairs,jacobian)
            source={int(entity):i for i,entity in enumerate(order)}
            basis=basis[[source[int(entity)] for entity in component.cpu().tolist()]]
            row_lookup={tuple(sorted(pair)):i for i,pair in enumerate(pairs)}
            rows=[row_lookup[tuple(sorted((order[i],order[(i+1)%len(order)])))] for i in range(len(order))]
            cycle={'rows':torch.tensor(rows,device=scene.device),'forward':torch.tensor([pairs[row,0]==order[i] for i,row in enumerate(rows)],device=scene.device),'in_order':torch.tensor([int(torch.where(component==entity)[0][0]) for entity in order],device=scene.device),'from_order':torch.tensor([source[int(entity)] for entity in component.cpu().tolist()],device=scene.device),'corrections':torch.as_tensor(corrections,device=scene.device)}
            scene.rod_cycle_plans.append(Plan(component,int(order[0]),torch.as_tensor(basis,dtype=torch.float32,device=scene.device),torch.arange(6,device=scene.device),'cyclic',torch.as_tensor(markers,device=scene.device),torch.as_tensor(frame,dtype=torch.float32,device=scene.device),cycle))
            scene.rod_cycle_shear_errors.append(error)


def strain(scene, state, *, precise=False):
    geometry=scene.objective_scene if precise else scene
    x,q=(state.x.double(),state.q.double()) if precise else (state.x,state.q)
    return coordinates(x[:, scene.joints], q[:, scene.joints], geometry.anchor_p, geometry.anchor_c, geometry.anchor_qp, geometry.anchor_qc, geometry.rod_rest_bend, geometry.rod_rest_twist,pinned=precise)


def begin(scene, state, dt, *, precise=False):
    if state.history is None:
        raise ValueError('Rod dynamics requires explicit history, including after restart')
    h = state.history
    reset = h['reset'].expand(-1, len(scene.joints))
    previous = State(torch.where(h['reset'][..., None], state.x, h['previous_x']), torch.where(h['reset'][..., None], state.q, h['previous_q']), state.v, state.omega)
    previous_c = strain(scene, previous,precise=precise)
    memory = DahlMemory(h['kappa'], h['increment'], h['stress'])
    enabled = torch.ones_like(reset)
    if scene.rod_enabled_dahl:
        frozen, sigma, friction = begin_dahl(memory, previous_c[..., 3:], scene.joint_stiffness[:, 3:], scene.rod_dahl_eps, scene.rod_dahl_tau, enabled, reset)
    else:
        frozen = memory; sigma = torch.zeros_like(memory.stress); friction = torch.zeros_like(sigma)
    rho = automatic_rho(scene.joint_stiffness, scene.rod_inverse_mass[scene.joints], scene.rod_inverse_inertia[scene.joints], dt)
    dual = h['dual'] * scene.metadata['joint_gamma']
    context = RodContext(scene.joint_stiffness, rho, dual, previous_c, previous_c, scene.rod_damping, sigma, friction, dt, scene.metadata['joint_alpha'],precise)
    if precise:
        context=RodContext(**{field.name:(getattr(context,field.name).double() if torch.is_tensor(getattr(context,field.name)) else getattr(context,field.name)) for field in fields(context)})
    return context, frozen, previous


def physical_rows(scene,state,context):
    geometry=scene.objective_scene if context.geometry_fp64 else scene
    x,q=(state.x.double(),state.q.double()) if context.geometry_fp64 else (state.x,state.q)
    return evaluate(x[:,scene.joints],q[:,scene.joints],geometry.body_com[scene.joints],geometry.anchor_p,geometry.anchor_c,geometry.anchor_qp,geometry.anchor_qc,geometry.rod_rest_bend,geometry.rod_rest_twist,context)


class Metric:
    def __init__(self, scene, local, jacobian, weights):
        self.scene, self.local = scene, local.detach()
        self.jacobian, self.weights = jacobian.detach(), weights.detach()

    def __call__(self, value):
        dtype=value.dtype
        value=value.to(torch.promote_types(value.dtype,self.jacobian.dtype))
        jacobian=self.jacobian.to(value.dtype)
        result = (self.local.to(value.dtype) @ value[..., None])[..., 0]
        restricted = torch.einsum('brcsi,brsi->brc', jacobian, value[:, self.scene.joints])
        reaction = torch.einsum('brcsi,brc->brsi', jacobian, self.weights.to(value.dtype) * restricted)
        return (result.index_add(1, self.scene.joints.flatten(), reaction.flatten(1, 2)) * self.scene.dof).to(dtype)


def rigid_modes(scene,state,ids):
    from .response_operator import skew
    com=state.x[:,ids]+rotate_pinned(state.q[:,ids],scene.body_com[ids])
    center=(com*scene.mass[ids][None,:,None]).sum(1,keepdim=True)/scene.mass[ids].sum()
    eye=torch.eye(3,device=scene.device).expand(len(state.x),len(ids),3,3)
    zero=torch.zeros_like(eye)
    return torch.cat((torch.cat((eye,-skew(com-center)),-1),torch.cat((zero,eye),-1)),-2)


class Coarse:
    def __init__(self, scene, state, metric, gains, *, intrinsic=False, plan_list=None, boundary_lift=False, material_twist=False, chain_shear=False, curvature_ports=False, flexure_ports=False):
        self.scene = scene; self.blocks = []
        self.valid = torch.ones(len(state.x), device=scene.device, dtype=torch.bool)
        material_plans=scene.rod_flexure_plans if flexure_ports else scene.rod_curvature_plans if curvature_ports else scene.rod_material_plans
        plans=(material_plans if material_twist else scene.rod_plans)+(scene.rod_cycle_plans if intrinsic else [])
        if plan_list is not None:plans=plan_list
        for index, plan in enumerate(plans):
            rotation = so3.multiply(state.q[:, plan.root], so3.conjugate(scene.rest_q[plan.root]))
            reference_basis=plan.boundary_basis if boundary_lift and plan.boundary_basis is not None else plan.basis
            basis = reference_basis[None].expand(len(state.x), -1, -1, -1)
            translated = so3.rotate(rotation[:, None, None, :], basis[:, :, :3].transpose(-1, -2)).transpose(-1, -2)
            angular = so3.rotate(rotation[:, None, None, :], basis[:, :, 3:].transpose(-1, -2)).transpose(-1, -2)
            modes = torch.cat((translated, angular), -2)[..., plan.columns] * scene.dof[plan.ids][None, :, :, None]
            if plan.kind in ('twist','flexure'):
                angular=so3.rotate(state.q[:,plan.ids,None,:],basis[:,:,3:].transpose(-1,-2)).transpose(-1,-2)[...,plan.columns]
                lever=rotate_pinned(state.q[:,plan.ids],scene.body_com[plan.ids])
                linear=torch.linalg.cross(angular.transpose(-1,-2),lever[:,:,None,:]).transpose(-1,-2)
                modes=torch.cat((linear,angular),-2)*scene.dof[plan.ids][None,:,:,None]
            if intrinsic and plan.kind=='rigid':modes=rigid_modes(scene,state,plan.ids)[...,plan.columns]*scene.dof[plan.ids][None,:,:,None]
            if intrinsic and plan.kind=='cyclic':
                with torch.no_grad():
                    p=state.x[:,plan.frame_ids]+rotate_pinned(state.q[:,plan.frame_ids],scene.body_com[plan.frame_ids])
                    u=p[:,0]-p[:,2];un=u.norm(dim=-1,keepdim=True);u=u/un.clamp_min(1e-20)
                    v=p[:,1]-p[:,3];v=v-(v*u).sum(-1,keepdim=True)*u;vn=v.norm(dim=-1,keepdim=True);v=v/vn.clamp_min(1e-20)
                    usable=(un[:,0]>scene.length*1e-6)&(vn[:,0]>scene.length*1e-6)
                    frame=torch.stack((u,v,torch.linalg.cross(u,v)),-1)@plan.frame_rest.T
                    # This frame only orients proposals. A degenerate marker
                    # chord need not reject an otherwise valid physical state.
                    frame=torch.where(usable[:,None,None],frame,so3.matrix(rotation).detach())
                modes=torch.cat((torch.einsum('bij,bnjk->bnik',frame,basis[:,:,:3]),torch.einsum('bij,bnjk->bnik',frame,basis[:,:,3:])),-2)
                # Reconstruct translations in the CURRENT linear shear
                # nullspace. A transported rest mode would become artificially
                # stiff as a ring bends. All solves here are 3x3 per edge or
                # 3x3 for closure, followed by one parallel prefix sum.
                cycle=plan.cycle;j=metric.jacobian[:,cycle['rows'],:3].double()
                left=torch.where(cycle['forward'][None,:,None,None],j[...,0,:],j[...,1,:])
                right=torch.where(cycle['forward'][None,:,None,None],j[...,1,:],j[...,0,:])
                local_solve=torch.linalg.solve_ex(right[...,:3],-torch.cat((left[...,3:],right[...,3:]),-1),check_errors=False)
                self.valid&=(local_solve.info==0).all(-1)&torch.isfinite(local_solve.result).flatten(1).all(-1)
                a,b=local_solve.result[...,:3],local_solve.result[...,3:]
                angular=modes[:,cycle['in_order'],3:].double()
                fields=cycle['corrections'][None].expand(len(state.x),-1,-1,-1)
                closure_map=(a@fields+b@fields.roll(-1,1)).sum(1)
                mismatch=(a@angular+b@angular.roll(-1,1)).sum(1)
                closure=torch.linalg.solve_ex(closure_map@closure_map.transpose(-1,-2),-mismatch,check_errors=False)
                self.valid&=(closure.info==0)&torch.isfinite(closure.result).flatten(1).all(-1)
                angular=angular+fields@(closure_map.transpose(-1,-2)@closure.result)[:,None]
                increments=a@angular+b@angular.roll(-1,1)
                translations=torch.cat((torch.zeros_like(increments[:,:1]),increments[:,:-1].cumsum(1)),1)
                translations=translations-translations.mean(1,keepdim=True)
                modes=torch.cat((translations,angular),-2)[:,cycle['from_order']].to(modes.dtype)
                # Remove mass-metric rigid motion; the internal learned gain
                # cannot inject a net translation or angular momentum mode.
                g=rigid_modes(scene,state,plan.ids).double();m=metric.local[:,plan.ids].double();z=modes.double()
                gram_g=torch.einsum('bnip,bnij,bnjq->bpq',g,m,g)
                rhs=torch.einsum('bnip,bnij,bnjq->bpq',g,m,z)
                solve=torch.linalg.solve_ex(gram_g,rhs,check_errors=False)
                self.valid&=(solve.info==0)&torch.isfinite(solve.result).flatten(1).all(-1)
                modes=(z-g@solve.result[:,None]).to(modes.dtype)
            if chain_shear and plan.chain is not None:
                from .rod_chain import current_modes
                modes,valid=current_modes(plan,modes,metric)
                self.valid&=valid
            full = state.x.new_zeros(len(state.x), scene.n, 6, len(plan.columns),dtype=modes.dtype).index_copy(1, plan.ids, modes)
            if material_twist and plan.kind in ('static','flexure'):
                # Separate the remaining beam fields in the CURRENT SPD
                # metric, using only the preceding small twist factor. The
                # additive blocks then do not count the same response twice.
                for old_plan,old_modes,old_scale,old_factor,_ in self.blocks:
                    if old_plan.root!=plan.root:continue
                    # Beam includes an intentionally extensional axial port;
                    # subtracting it would reintroduce strain into flexure.
                    # Beam/flexure overlap remains an additive PSD response.
                    if old_plan.kind!='twist':continue
                    previous=full.new_zeros(len(state.x),scene.n,6,len(old_plan.columns)).double().index_copy(1,old_plan.ids,old_modes)
                    z=full.double();j=metric.jacobian.double()
                    left=torch.einsum('brcsi,brsip->brcp',j,previous[:,scene.joints])
                    right=torch.einsum('brcsi,brsip->brcp',j,z[:,scene.joints])
                    cross=torch.einsum('bnip,bnij,bnjq->bpq',previous,metric.local.double(),z)+torch.einsum('brcp,brc,brcq->bpq',left,metric.weights.double(),right)
                    white=torch.linalg.solve_triangular(old_factor,cross*old_scale[:,:,None],upper=False)
                    coefficient=torch.linalg.solve_triangular(old_factor.transpose(-1,-2),white,upper=True)*old_scale[:,:,None]
                    modes=modes.double()-old_modes@coefficient[:,None]
                    full=full.double().index_copy(1,plan.ids,modes)
            # Assemble Z^T H Z from positive sparse terms in FP64. Applying H
            # in FP32 first loses the near-null bend/shear cancellation in
            # very slender rods and makes the result depend on batch shape.
            z=full.double()
            gram = torch.einsum('bnip,bnij,bnjq->bpq',z,metric.local.double(),z)
            restricted=torch.einsum('brcsi,brsip->brcp',metric.jacobian.double(),z[:,scene.joints])
            gram+=torch.einsum('brcp,brc,brcq->bpq',restricted,metric.weights.double(),restricted)
            scale = gram.diagonal(dim1=-2, dim2=-1).clamp_min(1e-30).rsqrt()
            factor, info = torch.linalg.cholesky_ex(gram * scale[:, :, None] * scale[:, None, :], check_errors=False)
            self.valid &= (info == 0) & torch.isfinite(factor).flatten(1).all(-1)
            gain = gains[index][..., plan.columns].double() if gains is not None else None
            self.blocks.append((plan, modes.double(), scale, factor, gain))

    def __call__(self, residual):
        result = torch.zeros_like(residual)
        for plan, modes, scale, factor, gain in self.blocks:
            restricted = torch.einsum('bnip,bni->bp', modes, residual[:, plan.ids].double())
            white = torch.linalg.solve_triangular(factor, (restricted * scale)[..., None], upper=False)
            if gain is not None:white = white * gain[..., None]
            coefficient = torch.linalg.solve_triangular(factor.transpose(-1, -2), white, upper=True)[..., 0] * scale
            result = result.index_add(1, plan.ids, torch.einsum('bnip,bp->bni', modes, coefficient).to(result.dtype))
        return result


def move(scene, state, update, alpha):
    delta = update * alpha[..., None] * scene.dof
    q = so3.normalize(so3.multiply(so3.exp(delta[..., 3:]), state.q))
    q = torch.where(scene.movable[None, :, None], q, state.q)
    x = state.x + delta[..., :3] + (rotate_pinned(state.q, scene.body_com) - rotate_pinned(q, scene.body_com))
    return State(torch.where(scene.movable[None, :, None], x, state.x), q, state.v, state.omega)


def merit(scene, state, target, context, dt, *, with_roundoff=False):
    # Cast before geometric subtraction/rotation. Casting an already rounded
    # FP32 strain cannot recover tiny changes in a stiff row's potential.
    geometry = scene.objective_scene
    x, q = state.x.double(), state.q.double()
    target_x, target_q = target.x.double(), target.q.double()
    displacement = x - target_x + (rotate_pinned(q, geometry.body_com) - rotate_pinned(target_q, geometry.body_com))
    angle = so3.log(so3.multiply(so3.conjugate(q), target_q))
    energy = .5 * geometry.mass[None] / dt**2 * displacement.square().sum(-1)
    energy += .5 / dt**2 * (angle * (geometry.inertia @ angle[..., None])[..., 0]).sum(-1)
    double = RodContext(**{field.name: (getattr(context, field.name).double() if torch.is_tensor(getattr(context, field.name)) else getattr(context, field.name)) for field in fields(context)})
    c = coordinates(x[:, scene.joints], q[:, scene.joints], geometry.anchor_p, geometry.anchor_c, geometry.anchor_qp, geometry.anchor_qc, geometry.rod_rest_bend, geometry.rod_rest_twist,pinned=context.geometry_fp64)
    channels = double.terms(c, per_channel_energy=True)[2]
    potential = energy.index_add(1, scene.joints[:, 1], channels.sum(-1))
    if not with_roundoff:return potential
    # Retain the same measured acceptance band for both precision branches.
    # The legacy local residual uses FP32 geometry; the FP64-row experiment
    # deliberately keeps this policy to isolate the geometry change. Compare
    # channel by channel so cancelling energies cannot hide the estimate.
    # This band is explicit; neither branch claims strict energy descent.
    rounded_c = coordinates(state.x[:,scene.joints],state.q[:,scene.joints],scene.anchor_p,scene.anchor_c,scene.anchor_qp,scene.anchor_qc,scene.rod_rest_bend,scene.rod_rest_twist,pinned=context.geometry_fp64).double()
    rounded_channels = double.terms(rounded_c, per_channel_energy=True)[2]
    row_error = (rounded_channels - channels).abs().sum(-1)
    # A principal-angle branch change is not arithmetic uncertainty.
    same_branch = (rounded_c[..., 3:] - c[..., 3:]).abs().amax(-1) < torch.pi
    row_error = torch.where(same_branch, row_error, 0.)
    rounded_displacement = state.x - target.x + (rotate_pinned(state.q, scene.body_com) - rotate_pinned(target.q, scene.body_com))
    rounded_angle = so3.log(so3.multiply(so3.conjugate(state.q), target.q)).double()
    rounded_energy = .5 * geometry.mass[None] / dt**2 * rounded_displacement.double().square().sum(-1)
    rounded_energy += .5 / dt**2 * (rounded_angle * (geometry.inertia @ rounded_angle[..., None])[..., 0]).sum(-1)
    uncertainty = (rounded_energy - energy).abs().index_add(1, scene.joints[:, 1], row_error)
    return potential, uncertainty


def step(model, scene, start, dt, learned, force, torque):
    from .mixed_solver import relation_inputs
    precise=model.config.get('rod_geometry_fp64',False)
    protected=model.config.get('protected_learning',False)
    context, frozen, previous_pose = begin(scene, start, dt,precise=precise)
    scene.use_exact_inertia = True
    com_start = State(start.x + rotate_pinned(start.q, scene.body_com), start.q, start.v, start.omega)
    target_com, target_q = predictor(scene, com_start, dt, force, torque)
    target = State(target_com - rotate_pinned(target_q, scene.body_com), target_q, start.v, start.omega)
    current = State(start.x, start.q, start.v, start.omega)
    w = mass_whitening(scene, com_start, dt).detach()
    batch = len(start.x); eye = torch.eye(6, device=scene.device)
    accepted = torch.ones(batch, device=scene.device, dtype=torch.bool)
    rejected = torch.zeros(batch, scene.n, device=scene.device, dtype=torch.bool)
    roots = torch.zeros_like(rejected); feedback = start.x.new_ones(batch, scene.n)
    minimum = start.x.new_ones(batch); last_physical = torch.zeros(batch, scene.n, 6, device=scene.device)
    rejected_cost = torch.zeros_like(feedback, dtype=torch.float64)
    roundoff_accepted = torch.zeros_like(rejected)
    precision_budget = torch.zeros_like(rejected_cost)
    velocity_seed_used=torch.zeros_like(rejected)
    intrinsic=model.config.get('rod_cycle_modes',False)
    cycle_seed=intrinsic and model.config.get('rod_cycle_velocity',False) and bool(scene.rod_cycle_plans)
    material_plans=scene.rod_flexure_plans if model.config.get('rod_curvature_flexure',False) else scene.rod_curvature_plans if model.config.get('rod_curvature_ports',False) else scene.rod_material_plans
    material_seed=model.config.get('rod_material_velocity',False) and model.config.get('rod_material_twist_modes',False) and any(plan.kind=='twist' for plan in material_plans)
    if cycle_seed or material_seed:
        # Preserve representable coarse momentum before the bounded force
        # corrections. Restarting every small step at the old pose repeatedly
        # damps even a resolved bending velocity when K is finite.
        with torch.no_grad():
            initial_rows=physical_rows(scene,start,context)
        rotation0=so3.matrix(start.q)
        mass=local_mass=start.x.new_zeros(batch,scene.n,6,6)
        mass[...,:3,:3]=scene.mass[None,:,None,None]/dt**2*eye[:3,:3]
        mass[...,3:,3:]=rotation0@scene.inertia@rotation0.transpose(-1,-2)/dt**2
        mass_metric=Metric(scene,local_mass,initial_rows['jacobian'],start.x.new_zeros(batch,len(scene.joints),6))
        seed_components=(scene.rod_cycle_plans if cycle_seed else [])+([plan for plan in material_plans if plan.kind=='twist'] if material_seed else [])
        seed_roots={plan.root for plan in seed_components}
        base_plans=material_plans if material_seed else scene.rod_plans
        plans=[plan for plan in base_plans if plan.root in seed_roots]+(scene.rod_cycle_plans if cycle_seed else [])
        transport=Coarse(scene,start,mass_metric,None,intrinsic=intrinsic,plan_list=plans,boundary_lift=model.config.get('rod_boundary_lift',False),material_twist=model.config.get('rod_material_twist_modes',False),chain_shear=model.config.get('rod_chain_shear_modes',False),curvature_ports=model.config.get('rod_curvature_ports',False),flexure_ports=model.config.get('rod_curvature_flexure',False))
        if protected:
            from .protected_response import ProtectedCoarse
            transport=ProtectedCoarse(scene,mass_metric,transport.blocks)
        velocity=torch.cat((start.v,start.omega),-1)*dt
        seed=transport((local_mass@velocity[...,None])[...,0])
        seed_scale=start.x.new_ones(batch,scene.n)
        for plan in seed_components:
            size=(seed[:,plan.ids,:3]/scene.length).square().sum(-1)+seed[:,plan.ids,3:].square().sum(-1)
            scale=(model.config.get('trust_radius',.1)/size.amax(-1).clamp_min(1e-30).sqrt()).clamp_max(1)
            seed_scale[:,plan.ids]=scale[:,None].expand(-1,len(plan.ids))
        candidate=move(scene,start,seed,seed_scale)
        with torch.no_grad():
            e0,b0=merit(scene,start,target,context,dt,with_roundoff=True)
            e1,b1=merit(scene,candidate,target,context,dt,with_roundoff=True)
        for plan in seed_components:
            ids=plan.ids;before=e0[:,ids].sum(-1)
            good=(e1[:,ids].sum(-1)-before<=before.abs()*1e-5+1e-10+(b0[:,ids]+b1[:,ids]).sum(-1))&transport.valid
            good&=seed[:,ids].abs().amax((-1,-2))>0
            velocity_seed_used[:,ids]=good[:,None].expand(-1,len(ids))
        current=State(torch.where(velocity_seed_used[...,None],candidate.x,current.x),torch.where(velocity_seed_used[...,None],candidate.q,current.q),start.v,start.omega)
    predictor_rounds=int(model.config.get('analytic_predictor_rounds',0))
    phases=[]
    for iteration in range(predictor_rounds+model.config.get('rod_unroll', model.config.get('unroll', 4))):
        iteration_learned=learned and iteration>=predictor_rounds
        phases.append({'phase':'neural_correction' if iteration_learned else 'analytic_predictor'})
        if predictor_rounds and iteration:
            interval_v=(current.x-start.x+rotate_pinned(current.q,scene.body_com)-rotate_pinned(start.q,scene.body_com))/dt
            interval_w=so3.log(so3.multiply(current.q,so3.conjugate(start.q)))/dt
            current=State(current.x,current.q,torch.where(scene.movable[None,:,None],interval_v,start.v),
                          torch.where(scene.movable[None,:,None],interval_w,start.omega))
        evaluation = physical_rows(scene,current,context)
        displacement = current.x - target.x + (rotate_pinned(current.q, scene.body_com) - rotate_pinned(target.q, scene.body_com))
        rotation = so3.matrix(current.q)
        angular_mass = rotation @ scene.inertia @ rotation.transpose(-1, -2) / dt**2
        angle = so3.log(so3.multiply(current.q, so3.conjugate(target.q)))
        residual = torch.cat((scene.mass[None, :, None] / dt**2 * displacement, (angular_mass @ angle[..., None])[..., 0]), -1)
        residual = residual.to(evaluation['residual'].dtype).index_add(1, scene.joints.flatten(), evaluation['residual'].flatten(1, 2)).to(start.x.dtype) * scene.dof
        local = start.x.new_zeros(batch, scene.n, 6, 6)
        local[..., :3, :3] = scene.mass[None, :, None, None] / dt**2 * eye[:3, :3]
        local[..., 3:, 3:] = angular_mass
        blocks = local.to(evaluation['blocks'].dtype).index_add(1, scene.joints.flatten(), evaluation['blocks'].flatten(1, 2))
        white = w.double() @ blocks.detach().double() @ w.double() + torch.diag_embed((~scene.dof).double())[None] + 1e-6 * eye.double()
        baseline64 = torch.linalg.inv_ex(white, check_errors=False).inverse.detach()
        baseline = baseline64.to(start.x.dtype)
        if model.config.get('frozen_residual_backward') and torch.is_grad_enabled():residual = residual.detach()
        r = (w @ residual[..., None])[..., 0]
        physical = -(w @ baseline @ r[..., None])[..., 0]; last_physical = physical
        empty3 = start.x.new_empty(batch, 0, 3)
        extras = (evaluation['coordinates'].to(start.x.dtype), evaluation['jacobian'].to(start.x.dtype), start.x.new_empty(batch, 0, 3, 6), empty3, start.x.new_empty(batch, 0), torch.empty(batch, 0, device=scene.device, dtype=torch.bool), empty3, start.x.new_empty(batch, 0, 3, 2, 6), [])
        jac, relation = relation_inputs(scene, current, start, extras, dt, feedback)
        entity = torch.cat(((current.x - scene.origins) / scene.length, start.v * dt / scene.length, (target.x - current.x) / scene.length, so3.log(so3.multiply(target.q, so3.conjugate(current.q))), torch.asinh(r / .1), physical * scene.generalized_scale), -1)
        gains = None
        encoded = None
        if iteration_learned and not protected:
            encoded = model.response.encode_entities(scene.layout, entity, relation)
            if model.coarse_gain is not None:
                plans=(material_plans if model.config.get('rod_material_twist_modes',False) else scene.rod_plans)+(scene.rod_cycle_plans if intrinsic else [])
                gains=[]
                for plan in plans:
                    value=1+.5*model.coarse_gain(encoded[:,plan.ids].mean(1)).tanh()
                    if intrinsic and plan.kind=='rigid':value=torch.ones_like(value)
                    if intrinsic and plan.kind=='cyclic':value=value.mean(-1,keepdim=True).expand_as(value)
                    gains.append(value)
        metric = Metric(scene, local, evaluation['jacobian'], context.terms(evaluation['coordinates'])[1])
        coarse = Coarse(scene, current, metric, gains,intrinsic=intrinsic,boundary_lift=model.config.get('rod_boundary_lift',False),material_twist=model.config.get('rod_material_twist_modes',False),chain_shear=model.config.get('rod_chain_shear_modes',False),curvature_ports=model.config.get('rod_curvature_ports',False),flexure_ports=model.config.get('rod_curvature_flexure',False))
        accepted &= coarse.valid & evaluation['in_domain'].all(-1)
        if protected:
            from .protected_response import protected_update
            update,coarse_r,valid=protected_update(model,scene,current,metric,coarse,residual,w,baseline64,relation,physical,dt,iteration_learned)
            accepted &= valid
        else:
            coarse_r = coarse(residual)
            complement = residual - metric(coarse_r)
            if iteration_learned:local_update = model.response(scene.layout, complement, w, baseline, jac, entity, relation, encoded=encoded)
            else:local_update = -(w @ baseline @ (w @ complement[..., None]))[..., 0]
            update = -coarse_r + local_update - coarse(metric(local_update))
        size = (update[..., :3] / scene.length).square().sum(-1) + update[..., 3:].square().sum(-1)
        trust=(model.config.get('trust_radius', .1) / size.clamp_min(1e-30).sqrt()).clamp_max(1)
        if protected:
            for ids in scene.interaction_entities:trust[:,ids]=trust[:,ids].amin(-1,keepdim=True).expand(-1,len(ids))
        if intrinsic:
            for plan in scene.rod_cycle_plans:
                trust[:,plan.ids]=trust[:,plan.ids].amin(-1,keepdim=True).expand(-1,len(plan.ids))
        update = update * trust[...,None] * model.config.get('relaxation', .8)
        with torch.no_grad():before, before_error = merit(scene, current, target, context, dt, with_roundoff=True)
        selected = torch.zeros_like(rejected); chosen = current; used = torch.zeros_like(feedback)
        best_change = torch.full_like(rejected_cost, float('inf'))
        for alpha in model.config.get('line_search', [1., .5, .25, .125, .0625]):
            candidate = move(scene, current, update, torch.full_like(feedback, alpha))
            with torch.no_grad():energy, energy_error = merit(scene, candidate, target, context, dt, with_roundoff=True)
            choose = torch.zeros_like(selected)
            for ids in scene.interaction_entities:
                initial_energy = before[:, ids].sum(-1)
                change = energy[:, ids].sum(-1) - initial_energy
                # Frozen ALM duals and Dahl stress add linear terms, so this
                # incremental potential can be negative. Scale the roundoff
                # allowance by its magnitude, independent of its sign.
                nominal = initial_energy.abs() * 1e-5 + 1e-10
                allowance = (before_error[:, ids] + energy_error[:, ids]).sum(-1)
                good = change <= nominal + allowance
                roundoff_accepted[:, ids] |= (good & (change > nominal))[:, None] & ~selected[:, ids]
                precision_budget[:, ids] = torch.maximum(precision_budget[:, ids], allowance[:, None])
                best_change[:, ids] = torch.minimum(best_change[:, ids], change[:, None])
                choose[:, ids] = good[:, None].expand(-1,len(ids))
            choose &= ~selected
            chosen = State(torch.where(choose[..., None], candidate.x, chosen.x), torch.where(choose[..., None], candidate.q, chosen.q), start.v, start.omega)
            used = torch.where(choose, alpha, used); selected |= choose
        tolerance = model.config.get('root_tolerance', [5e-7, 2e-6])
        small = (physical[..., :3].abs().amax(-1) <= scene.length * tolerance[0]) & (physical[..., 3:].abs().amax(-1) <= tolerance[1])
        small &= (coarse_r[..., :3].abs().amax(-1) <= scene.length * tolerance[0]) & (coarse_r[..., 3:].abs().amax(-1) <= tolerance[1])
        for ids in scene.interaction_entities:
            root = small[:, ids].all(-1, keepdim=True).expand(-1, len(ids)) & ~selected[:, ids]
            roots[:, ids] |= root; selected[:, ids] |= root; used[:, ids] = torch.where(root, 1., used[:, ids])
        accepted &= selected.all(-1); rejected |= ~selected
        rejected_cost = torch.where(~selected, best_change, rejected_cost)
        current = chosen
        context = context.numeric_trial(strain(scene, current,precise=precise))
        feedback = used; minimum = torch.minimum(minimum, used.amin(-1))
    final_c = strain(scene, current,precise=precise)
    qa = so3.multiply(current.q[:, scene.joints[:, 0]], scene.anchor_qp)
    qb = so3.multiply(current.q[:, scene.joints[:, 1]], scene.anchor_qc)
    accepted &= ((so3.matrix(qa)[..., 2] * so3.matrix(qb)[..., 2]).sum(-1) > -.9).all(-1)
    if scene.rod_enabled_dahl:
        memory = commit_dahl(frozen, final_c[..., 3:], scene.joint_stiffness[:, 3:], scene.rod_dahl_eps, scene.rod_dahl_tau, torch.ones(batch, len(scene.joints), device=scene.device, dtype=torch.bool))
    else:memory = frozen
    v = (current.x + rotate_pinned(current.q, scene.body_com) - previous_pose.x - rotate_pinned(previous_pose.q, scene.body_com)) / dt
    omega = so3.log(so3.multiply(current.q, so3.conjugate(previous_pose.q))) / dt
    history = {'dual': context.dual, 'kappa': memory.kappa, 'increment': memory.increment, 'stress': memory.stress, 'previous_x': current.x, 'previous_q': current.q, 'reset': torch.zeros_like(start.history['reset'])}
    history={key:value.to(start.history[key].dtype) for key,value in history.items()}
    result = State(current.x, current.q, v, omega, history)
    finite = torch.stack([torch.isfinite(value).flatten(1).all(-1) for value in (result.x, result.q, result.v, result.omega, *history.values())]).all(0)
    accepted &= finite
    def commit(value, old):
        return torch.where(accepted.reshape(batch, *([1] * (value.ndim - 1))), value, old)
    result = State(*(commit(getattr(result, key), getattr(start, key)) for key in ('x', 'q', 'v', 'omega')), {key: commit(value, start.history[key]) for key, value in history.items()})
    return result, {'accepted': accepted, 'iterations':phases,'min_alpha': minimum, 'rejected_entities': rejected, 'rejected_physical_step': last_physical, 'rejected_model_cost_change': rejected_cost, 'numerical_root_entities': roots, 'roundoff_accepted_entities':roundoff_accepted,'merit_precision_budget':precision_budget,'velocity_seed_used':velocity_seed_used,'contact_safety': 'NO_ACTIVE_CONTACTS', 'architecture': model.architecture_id}
