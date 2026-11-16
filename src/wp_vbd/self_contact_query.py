"""Bounded complete particle VF/EE candidates for one linear-motion envelope.

The broad phase is deliberately simple and chunked, with quadratic work but no
quadratic global allocation. It does not instantiate a VBD solver or read its
candidate/history buffers. Overflow and an uncertified initial pair reject the
query; neither is silently truncated.
"""
from dataclasses import dataclass
import math

import numpy as np
import torch

from .contact_ports import PointPorts
from .self_contact_geometry import edge_edge, path_feasible, refined_path_feasible, vertex_face
from .self_contact_rows import SelfPointContacts, safe_norm


def interaction_geometry(configuration):
    """Resolve the pinned source's explicit and legacy margin/gap conventions."""
    radius = configuration.get('particle_self_contact_radius')
    margin = configuration.get('particle_self_contact_margin')
    gap = configuration.get('particle_self_contact_gap')
    if radius is not None:
        if gap is not None:
            raise ValueError('Legacy self radius cannot be combined with explicit gap')
        radius, gap = float(radius), float(.2 if margin is None else margin)-float(radius)
    elif margin is not None and gap is None:
        radius, gap = .2, float(margin)-.2
    else:
        radius, gap = float(.2 if margin is None else margin), float(0. if gap is None else gap)
    if not all(math.isfinite(value) and value >= 0 for value in (radius, gap)):
        raise ValueError('Finite nonnegative self interaction margin and query gap required')
    return radius, gap


@dataclass
class SelfContactSnapshot:
    contacts: SelfPointContacts
    reference: torch.Tensor
    pairs: tuple
    owners: tuple
    geometry: tuple
    motion_budget: float
    query_radius: float
    path_subdivision_depth: int = 0

    def path_feasible(self, positions):
        if positions.shape != self.reference.shape or positions.dtype != torch.float64:
            raise ValueError('Matched F64 material positions required')
        if not bool(torch.isfinite(positions).all()):
            return positions.new_tensor(False, dtype=torch.bool)
        allowed = (safe_norm(positions-self.reference) <= self.motion_budget).all()
        for owners, geometry in zip(self.owners, self.geometry):
            allowed = allowed & refined_path_feasible(geometry, positions[owners],self.path_subdivision_depth).all()
        return allowed


class SelfContactQuery:
    def __init__(self, model, configuration, *, device='cpu', motion_budget, chunk_size=256, max_rows=200000, broad_phase='chunked', responsive_particles=None,path_subdivision_depth=0):
        if not configuration['particle_enable_self_contact']:
            raise ValueError('An explicitly enabled self-contact model is required')
        if not math.isfinite(motion_budget) or motion_budget <= 0:
            raise ValueError('Finite positive particle motion budget required')
        if type(chunk_size) is not int or chunk_size <= 0 or type(max_rows) is not int or max_rows <= 0:
            raise ValueError('Positive integer query chunk size and row capacity required')
        self.device = torch.device(device)
        if broad_phase not in ('chunked', 'kdtree'):
            raise ValueError('Known self-contact broad phase required')
        self.broad_phase = broad_phase
        self.particle_count = model.particle_count
        self.node_count = model.particle_count+model.body_count+1
        if responsive_particles is not None:
            if responsive_particles.shape!=(self.particle_count,) or responsive_particles.dtype!=torch.bool or responsive_particles.device!=self.device:
                raise ValueError('Explicit matched particle response mask required')
            responsive_particles=responsive_particles.detach().clone()
        self.responsive_particles=responsive_particles
        if type(path_subdivision_depth) is not int or not 0<=path_subdivision_depth<=8:raise ValueError('Path subdivision depth in [0,8] required')
        self.path_subdivision_depth=path_subdivision_depth
        self.motion_budget = float(motion_budget)
        self.chunk_size, self.max_rows = chunk_size, max_rows
        self.margin, self.gap = interaction_geometry(configuration)
        self.query_radius = max(self.margin+self.gap, self.margin+2*self.motion_budget)
        self.rest_radius = float(configuration['particle_rest_shape_contact_exclusion_radius'])
        if not math.isfinite(self.rest_radius) or self.rest_radius < 0:
            raise ValueError('Finite nonnegative rest exclusion distance required')
        array = lambda value, dtype: torch.as_tensor(value.numpy(), device=self.device, dtype=dtype).clone()
        self.triangles = array(model.tri_indices, torch.long)
        self.edges = array(model.edge_indices, torch.long)[:, 2:4]
        self.world = array(model.particle_world, torch.long)
        self.rest = array(model.particle_q, torch.float64)
        for primitive in (self.triangles, self.edges):
            if bool(((primitive < 0) | (primitive >= self.particle_count)).any()):
                raise ValueError('Self-contact primitive references an invalid particle')
            if len(primitive) and bool((self.world[primitive] != self.world[primitive[:, :1]]).any()):
                raise ValueError('A self-contact primitive must belong to one particle world')
        self.filter_sets = self._filters(model, configuration)
        self.filter_keys = tuple(self._keys(rows, columns) for rows, columns in
                                 zip(self.filter_sets, (len(self.triangles), len(self.edges))))

    def _keys(self, sets, columns):
        values = [i*columns+j for i, row in enumerate(sets) for j in row]
        return torch.tensor(sorted(values), device=self.device, dtype=torch.long)

    def _filters(self, model, configuration):
        from newton._src.geometry.tri_mesh_collision import (
            build_edge_n_ring_edge_collision_filter, build_vertex_n_ring_tris_collision_filter,
        )
        threshold = configuration['particle_topological_contact_filter_threshold']
        if type(threshold) is not int or threshold < 0:
            raise ValueError('Nonnegative integer topology filter ring count required')
        vf = [set() for _ in range(self.particle_count)]
        ee = [set() for _ in range(len(self.edges))]
        if threshold >= 2 and len(self.triangles):
            adjacency = model.soft_mesh_adjacency.init_vertex_adjacency(model.particle_count)
            raw = lambda value: value.numpy() if hasattr(value, 'numpy') else np.asarray(value)
            args = (raw(model.edge_indices), raw(adjacency.v_adj_edges), raw(adjacency.v_adj_edges_offsets))
            vf = build_vertex_n_ring_tris_collision_filter(
                threshold, model.particle_count, *args, raw(adjacency.v_adj_tris), raw(adjacency.v_adj_tris_offsets)) or vf
            ee = build_edge_n_ring_edge_collision_filter(threshold, *args) or ee
        self.source_topological_edge_filter_sets = [set(row) for row in ee]
        self.topology_asymmetric_pairs = [(i, int(j)) for i, row in enumerate(ee) for j in row if i not in ee[j]]
        # The pinned source's adjacency includes opposite vertices of bending
        # elements, so even its built-in EE filter can be asymmetric. Our
        # reciprocal four-owner contract excludes an unordered pair if EITHER
        # source topology direction excludes it. This explicit union preserves
        # all source topology exclusions; it is not bitexact source EE force
        # ownership, nor a claim that the directed source table is symmetric.
        for first, second in self.topology_asymmetric_pairs:
            ee[second].add(first)
        for rows, columns, name in ((vf, len(self.triangles), 'vertex'), (ee, len(self.edges), 'edge')):
            external = configuration.get(f'particle_external_{name}_contact_filtering_map')
            if external is None:
                continue
            if not isinstance(external, dict):
                raise ValueError('Explicit self-contact filter dictionaries required')
            if name == 'edge' and any(key not in external.get(value, set()) for key, values in external.items() for value in values):
                raise ValueError('Asymmetric external EE filtering cannot own a reciprocal self-contact row')
            for key, values in external.items():
                if not isinstance(key, (int, np.integer)) or not 0 <= key < len(rows):
                    raise ValueError('External self-contact filter row out of range')
                if any(not isinstance(value, (int, np.integer)) or not 0 <= value < columns for value in values):
                    raise ValueError('External self-contact filter target out of range')
                rows[key].update(values)
        # An unordered four-owner interaction cannot preserve a one-sided
        # external EE rule. Such configurations need an explicit new contract.
        for first, targets in enumerate(ee):
            if any(first not in ee[second] for second in targets):
                raise ValueError('Asymmetric EE filtering cannot own a reciprocal self-contact row')
        return vf, ee

    def _query_pairs(self, positions, kind, capacity):
        if self.broad_phase == 'kdtree':
            from .self_contact_spatial import query_pairs
            return query_pairs(self, positions, kind, capacity)
        left = torch.arange(self.particle_count, device=self.device)[:, None] if kind == 9 else self.edges
        right = self.triangles if kind == 9 else self.edges
        keys = self.filter_keys[0 if kind == 9 else 1]
        target = positions[right]
        low, high = target.amin(1), target.amax(1)
        right_world = self.world[right[:, 0]]
        found = []
        for begin in range(0, len(left), self.chunk_size):
            ids = torch.arange(begin, min(begin+self.chunk_size, len(left)), device=self.device)
            primitive = left[ids]
            source = positions[primitive]
            lower, upper = source.amin(1), source.amax(1)
            gap = torch.maximum(low[None]-upper[:, None], lower[:, None]-high[None]).clamp_min(0.)
            # AABB distance is a lower bound. Round outward before exclusion;
            # do not drop a pair on an approximate closest-distance estimate.
            magnitude = torch.maximum(source.abs().amax((1, 2))[:, None], target.abs().amax((1, 2))[None]).clamp_min(1.)
            slack = 128*torch.finfo(positions.dtype).eps*magnitude
            mask = safe_norm(gap) <= self.query_radius+slack
            a_world = self.world[primitive[:, 0]][:, None]
            mask &= (a_world == right_world[None]) | (a_world < 0) | (right_world[None] < 0)
            shared = (primitive[:, None, :, None] == right[None, :, None, :]).any((-1, -2))
            mask &= ~shared
            if self.responsive_particles is not None:
                mask &= self.responsive_particles[primitive].any(-1)[:,None] | self.responsive_particles[right].any(-1)[None]
            if kind == 10:
                mask &= ids[:, None] < torch.arange(len(right), device=self.device)[None]
            pair_ids = ids[:, None]*len(right)+torch.arange(len(right), device=self.device)[None]
            if len(keys):
                index = torch.searchsorted(keys, pair_ids).clamp_max(len(keys)-1)
                mask &= keys[index] != pair_ids
            first, second = torch.where(mask)
            pairs = torch.stack((ids[first], second), -1)
            if self.rest_radius and len(pairs):
                owners = self._owners(kind, pairs)
                geometry = (vertex_face if kind == 9 else edge_edge)(self.rest[owners])
                pairs = pairs[geometry.distance >= self.rest_radius]
            capacity -= len(pairs)
            if capacity < 0:
                raise ValueError('Self-contact candidate capacity exceeded; no partial snapshot is admissible')
            found.append(pairs)
        return torch.cat(found) if found else torch.empty(0, 2, device=self.device, dtype=torch.long)

    def _owners(self, kind, pairs):
        if kind == 9:
            return torch.cat((self.triangles[pairs[:, 1]], pairs[:, :1]), -1)
        return torch.cat((self.edges[pairs[:, 0]], self.edges[pairs[:, 1]]), -1)

    @torch.no_grad()
    def snapshot(self, positions, *, finite_overlap_length=None):
        if positions.shape != (self.particle_count, 3) or positions.dtype != torch.float64 or positions.device != self.device:
            raise ValueError('One matched F64 material position array required')
        if not bool(torch.isfinite(positions).all()):
            raise ValueError('Nonfinite self-contact query positions')
        pairs_vf = self._query_pairs(positions, 9, self.max_rows)
        pairs_ee = self._query_pairs(positions, 10, self.max_rows-len(pairs_vf))
        pairs = (pairs_vf, pairs_ee)
        owners = tuple(self._owners(kind, value) for kind, value in zip((9, 10), pairs))
        geometry = tuple(compute(positions[ids]) for compute, ids in zip((vertex_face, edge_edge), owners))
        if finite_overlap_length is None:
            if any(bool((value.separation_lower_bound <= 0).any()) for value in geometry):
                raise ValueError('Initial self-contact candidate lacks a positive separation certificate')
        else:
            if not math.isfinite(finite_overlap_length) or finite_overlap_length<=0:raise ValueError('Positive finite overlap scale required')
            from types import SimpleNamespace
            from .self_contact_repair import tolerances
            bounds=tolerances(SimpleNamespace(owners=owners,geometry=geometry),self.rest,finite_overlap_length)
            for ids,value,tol in zip(owners,geometry,bounds):
                # Touching references need a unit feature frame rather than a
                # strictly positive certificate. Do not invent a zero normal.
                if not len(ids):continue
                axis=value.separating_axis;split=3 if value.kind==9 else 2
                projection=(positions[ids]*axis[:,None]).sum(-1)
                gap=projection[:,split:].amin(-1)-projection[:,:split].amax(-1)
                if bool((((axis.square().sum(-1)-1).abs()>1e-8)|(gap< -tol)).any()):
                    raise ValueError('Initial feature exceeds the finite oriented-overlap domain')
                value.normal=torch.where((value.distance>1e-12*finite_overlap_length)[:,None],value.normal,axis)
        all_owners = torch.cat(owners)
        count = len(all_owners)
        body_mask = torch.zeros(self.node_count, device=self.device, dtype=torch.bool)
        body_mask[self.particle_count:] = True
        ports = PointPorts(all_owners, torch.cat([value.weights for value in geometry]),
                           torch.zeros(count, 2, device=self.device, dtype=torch.bool),
                           positions.new_zeros(count, 2, 3), body_mask)
        kinds = torch.cat([torch.full((len(value),), kind, device=self.device, dtype=torch.long)
                           for kind, value in zip((9, 10), pairs)])
        contacts = SelfPointContacts(ports, torch.cat([value.normal for value in geometry]),
                                     positions.new_full((count,), self.margin), kinds)
        contacts.allow_finite_overlap=finite_overlap_length is not None
        return SelfContactSnapshot(contacts, positions.detach().clone(), pairs, owners, geometry,
                                   self.motion_budget, self.query_radius,self.path_subdivision_depth)
