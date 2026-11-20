"""Objective and physical-unit metrics for autonomous variable-step windows."""

import torch

from . import math3d as so3


WEIGHTS = {
    'position': 1., 'rotation': .5, 'linear_velocity': .5, 'angular_velocity': .25,
    'linear_momentum': .25, 'tet_strain': .25, 'volume': .25,
    'cloth_strain': .25, 'cloth_bend': .25, 'rod_bend_twist': .25,
    'rod_stretch_shear': .25, 'contact': .25, 'repair': .05,
    'tet_worst': .35, 'volume_barrier': .15,
    'grasp_relative_position': .75, 'grasp_relative_velocity': .4,
}


def robust(error, tolerance):
    scaled = error / tolerance
    return 2 * (scaled.square() / ((1 + scaled.square()).sqrt() + 1))


def rotation_error(a, b):
    return so3.log(so3.multiply(a, so3.conjugate(b)))


class RolloutObjective:
    def __init__(self, scene):
        self.scene = scene
        self.grasp_material_ids=None;self.grasp_hand_id=None
        if scene.metadata.get('training_schema')=='authored_franka_substep_reference_v310':
            model=scene.geometry_oracle.source_model
            labels=list(model.body_label)
            hand=[i for i,name in enumerate(labels) if name.endswith('/fr3_hand') or name.endswith('fr3_hand')]
            if not hand:hand=[i for i,name in enumerate(labels) if name.endswith('/fr3_link7')]
            if len(hand)!=1:raise ValueError('Franka grasp-relative loss requires one physical hand body')
            count=scene.metadata['particle_count']
            self.grasp_hand_id=count+hand[0]
            self.grasp_material_ids=torch.where((scene.kind<2)&scene.movable&(scene.object_id==0))[0]
        self.groups = [
            (ids, self.object_scale(ids))
            for obj in torch.unique(scene.object_id[scene.movable]).tolist()
            if len(ids := torch.where(scene.movable & (scene.object_id == obj))[0])
        ]
        self.rotating = [(ids[scene.body[ids]], length) for ids, length in self.groups
                         if bool(scene.body[ids].any())]
        self.element_groups = {}
        for name in ('tets', 'triangles'):
            ids = getattr(scene, name, None)
            if ids is not None and len(ids):
                mask=(scene.kind[ids[:,0]]==1) if name=='triangles' else None
                self.element_groups[name] = self.group_elements(ids,mask)
        edges = getattr(scene, 'bend_edges', None)
        if edges is not None and len(edges):
            edges = edges[scene.bending_indices]
            self.element_groups['bend_edges'] = self.group_elements(edges)
        self.rod_groups = []
        self.rod_strain = None
        rod = getattr(scene, 'rod', None)
        if rod is not None and len(rod.joints):
            from .rod_contact_system import strain
            self.rod_strain = strain
            rod_joints = rod.joints
        elif scene.metadata.get('history_contract') == 'rod_compliant_alm_v1':
            from .rod_solver import strain
            self.rod_strain = strain
            rod_joints = scene.joints
        if self.rod_strain is not None:
            self.rod_groups = self.group_elements(rod_joints)
            self.rod_lengths = self.element_lengths(rod_joints)

    def object_scale(self, ids):
        scene = self.scene
        extent = (scene.rest_x[ids].amax(0) - scene.rest_x[ids].amin(0)).norm()
        radii = getattr(scene, 'body_radius', None)
        if radii is not None:
            extent = extent.maximum(2 * radii[ids].amax())
        return extent.clamp_min(.025)

    def group_elements(self, ids, mask=None):
        labels = self.scene.object_id[ids[:, 0]]
        mobile = self.scene.movable[ids].any(-1)
        if mask is not None:mobile=mobile & mask
        return [torch.where(mobile & (labels == obj))[0]
                for obj in torch.unique(labels[mobile]).tolist()]

    def element_lengths(self, ids):
        x = self.scene.rest_x[ids]
        return (x[:, 1] - x[:, 0]).norm(dim=-1).clamp_min(.025)

    def __call__(self, predicted, target, diagnostic):
        scene = self.scene
        zero = predicted.x.new_zeros(())
        terms = {}
        metrics = {}
        dx = predicted.x - target.x
        dv = predicted.v - target.v
        angles = rotation_error(predicted.q, target.q)
        domega = predicted.omega - target.omega
        terms['position'] = torch.stack([robust(dx[:, ids], .01 * length).mean()
                                         for ids, length in self.groups]).mean()
        terms['linear_velocity'] = torch.stack([robust(dv[:, ids], .2 * length).mean()
                                                for ids, length in self.groups]).mean()
        momentum = [(dv[:, ids] * scene.mass[ids, None]).sum(1) for ids, _ in self.groups]
        # Momentum scale is object mass times a fixed, object-scale speed.
        terms['linear_momentum'] = torch.stack([
            robust(error, .2 * length * scene.mass[ids].sum()).mean()
            for error, (ids, length) in zip(momentum, self.groups)
        ]).mean()
        metrics['position_rms_m'] = torch.stack([dx[:, ids].square().mean().sqrt()
                                                for ids, _ in self.groups]).mean()
        metrics['position_max_m'] = dx[:, scene.movable].norm(dim=-1).amax()
        metrics['linear_velocity_rms_m_s'] = torch.stack([dv[:, ids].square().mean().sqrt()
                                                          for ids, _ in self.groups]).mean()
        metrics['linear_momentum_max_kg_m_s'] = torch.stack([p.norm(dim=-1).amax() for p in momentum]).amax()
        if self.rotating:
            terms['rotation'] = torch.stack([robust(angles[:, ids], .03).mean()
                                             for ids, _ in self.rotating]).mean()
        if self.grasp_material_ids is not None and len(self.grasp_material_ids):
            ids=self.grasp_material_ids;hand=self.grasp_hand_id
            mass=scene.mass[ids];total=mass.sum().clamp_min(1e-30)
            def relative(state):
                center=(state.x[:,ids]*mass[None,:,None]).sum(1)/total
                velocity=(state.v[:,ids]*mass[None,:,None]).sum(1)/total
                offset=center-state.x[:,hand]
                local_position=so3.rotate(so3.conjugate(state.q[:,hand]),offset)
                hand_point_velocity=state.v[:,hand]+torch.linalg.cross(state.omega[:,hand],offset)
                local_velocity=so3.rotate(so3.conjugate(state.q[:,hand]),velocity-hand_point_velocity)
                return local_position,local_velocity
            predicted_position,predicted_velocity=relative(predicted)
            target_position,target_velocity=relative(target)
            scale=self.object_scale(ids)
            terms['grasp_relative_position']=robust(predicted_position-target_position,.01*scale).mean()
            terms['grasp_relative_velocity']=robust(predicted_velocity-target_velocity,.2*scale).mean()
            metrics['grasp_relative_position_error_m']=(predicted_position-target_position).norm(dim=-1).amax()
            metrics['grasp_relative_velocity_error_m_s']=(predicted_velocity-target_velocity).norm(dim=-1).amax()
            terms['angular_velocity'] = torch.stack([robust(domega[:, ids], .5).mean()
                                                     for ids, _ in self.rotating]).mean()
            metrics['rotation_rms_rad'] = torch.stack([angles[:, ids].square().mean().sqrt()
                                                       for ids, _ in self.rotating]).mean()
            metrics['angular_velocity_rms_rad_s'] = torch.stack([domega[:, ids].square().mean().sqrt()
                                                                for ids, _ in self.rotating]).mean()
        for name, strain_key, rest in (
            ('tets', 'tet_strain', getattr(scene, 'tet_Dm', None)),
            ('triangles', 'cloth_strain', getattr(scene, 'tri_Dm', None)),
        ):
            groups = self.element_groups.get(name)
            if not groups:
                continue
            ids = getattr(scene, name)
            a, b = (self.deformation_gradient(state.x, ids, rest) for state in (predicted, target))
            strain_error = .5 * (a.transpose(-2, -1) @ a - b.transpose(-2, -1) @ b)
            terms[strain_key], metrics[strain_key + '_rms'] = self.reduce_elements(strain_error, groups, .05)
            if name == 'tets':
                det_a, det_b = torch.linalg.det(a), torch.linalg.det(b)
                terms['volume'], metrics['volume_ratio_error_rms'] = self.reduce_elements(det_a - det_b, groups, .05)
                metrics['minimum_tet_volume_ratio'] = det_a.amin()
                strain_norm=(strain_error.square().sum((-1,-2))+1e-24).sqrt()
                worst=[]
                for ids_group in groups:
                    values=robust(strain_norm[:,ids_group],.05)
                    count=max(1,(values.shape[-1]+19)//20)
                    worst.append(values.topk(count,dim=-1).values.mean())
                terms['tet_worst']=torch.stack(worst).mean()
                # The logarithm is symmetric in expansion/compression while
                # the barrier separately makes near-zero volume expensive.
                safe_a=det_a.clamp_min(1e-6);safe_b=det_b.clamp_min(1e-6)
                log_error=torch.log(safe_a)-torch.log(safe_b)
                log_loss,metrics['log_volume_error_rms']=self.reduce_elements(log_error,groups,.1)
                floor=[]
                for ids_group in groups:floor.append(robust((.15-det_a[:,ids_group]).clamp_min(0),.05).mean())
                terms['volume_barrier']=log_loss+torch.stack(floor).mean()
                metrics['minimum_positive_volume_margin']=det_a.amin()
        groups = self.element_groups.get('bend_edges')
        if groups:
            ids = scene.bend_edges[scene.bending_indices]
            angle = self.bend(predicted.x[:, ids]) - self.bend(target.x[:, ids])
            terms['cloth_bend'], metrics['cloth_bend_rms_rad'] = self.reduce_elements(
                torch.atan2(angle.sin(), angle.cos()), groups, .1)
        if self.rod_groups:
            a, b = (self.rod_strain(scene, state) for state in (predicted, target))
            terms['rod_stretch_shear'], metrics['rod_stretch_shear_rms'] = self.reduce_elements(
                (a[..., :3] - b[..., :3]) / self.rod_lengths[:, None], self.rod_groups, .05)
            terms['rod_bend_twist'], metrics['rod_bend_twist_rms'] = self.reduce_elements(
                a[..., 3:] - b[..., 3:], self.rod_groups, .1)
        gap = diagnostic.get('physical_contact_gap')
        if gap is not None and gap.numel():
            overlap = (-gap).clamp_min(0)
            metrics['maximum_contact_depth_m'] = overlap.amax()
            # Penalize the deepest overlap beyond the configured experiment tolerance.
            # Candidate count cannot dilute this term.
            threshold = scene.metadata.get('training_contact_tolerance_m', .002)
            terms['contact'] = robust((overlap.amax(-1) - threshold).clamp_min(0), threshold).mean()
        # The legacy auxiliary is already multiplied by 1000 by the solver.
        terms['repair'] = diagnostic.get('contact_repair_supervision', zero) / 1000
        objective = sum(WEIGHTS[key] * value for key, value in terms.items())
        return objective, terms, metrics

    @staticmethod
    def reduce_elements(error, groups, tolerance):
        loss = torch.stack([robust(error[:, ids], tolerance).mean() for ids in groups]).mean()
        rms = torch.stack([error[:, ids].square().mean().sqrt() for ids in groups]).mean()
        return loss, rms

    @staticmethod
    def deformation_gradient(x, ids, inverse_rest):
        geometry = x[:, ids]
        return (geometry[:, :, 1:] - geometry[:, :, :1]).transpose(-2, -1) @ inverse_rest

    @staticmethod
    def bend(points):
        a, b, c, d = points.unbind(-2)
        n1 = so3.normalize(torch.linalg.cross(c - a, d - a))
        n2 = so3.normalize(torch.linalg.cross(d - b, c - b))
        edge = so3.normalize(d - c)
        return torch.atan2((torch.linalg.cross(n1, n2) * edge).sum(-1), (n1 * n2).sum(-1))
