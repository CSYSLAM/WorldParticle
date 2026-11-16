"""Explicit static scene and Markov state for the contact-free pilot."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]


@dataclass
class State:
    x: torch.Tensor
    q: torch.Tensor
    v: torch.Tensor
    omega: torch.Tensor
    history: dict[str, torch.Tensor] | None = None

    def detach(self):
        return State(*(value.detach() for value in (self.x, self.q, self.v, self.omega)), None if self.history is None else {key: value.detach() for key, value in self.history.items()})

    def clone(self):
        return State(*(value.clone() for value in (self.x, self.q, self.v, self.omega)), None if self.history is None else {key: value.clone() for key, value in self.history.items()})

    def copy_(self, source):
        if (self.history is None) != (source.history is None):
            raise ValueError('State history contract changed')
        if self.history is not None and self.history.keys() != source.history.keys():
            raise ValueError('State history fields changed')
        for key in ('x', 'q', 'v', 'omega'):
            getattr(self, key).copy_(getattr(source, key))
        if self.history is not None:
            for key in self.history:
                self.history[key].copy_(source.history[key])
        return self


class Scene:
    def __init__(self, values, metadata):
        self.values = values
        self.metadata = metadata
        self.n = len(values["rest_x"])
        self.length = float(values["reference_length"])
        if "contacts" not in metadata:
            raise ValueError("Scene must explicitly declare its contact mode")
        self.has_plane = metadata["contacts"] == "plane"
        if metadata["contacts"] and not self.has_plane:
            raise ValueError("Only explicit analytic plane contact is implemented")
        if self.has_plane:
            required = {"plane_normal", "plane_offset", "contact_radius", "contact_material"}
            if required - values.keys() or bool((values["kind"] >= 2).any()):
                raise ValueError("Plane contact requires particle geometry and explicit materials")
            if abs(float(values["plane_normal"].norm())-1) > 1e-6:
                raise ValueError("Plane normal must be normalized")
            mat = values["contact_material"]
            if mat.shape != (4,) or not bool(torch.isfinite(mat).all()) or bool((mat < 0).any()) or float(mat[3]) <= 0:
                raise ValueError("Invalid contact ke, kd, mixed mu, friction epsilon")
            if float(values["contact_radius"]) <= 0:
                raise ValueError("Contact activation radius must be positive")
        if self.length <= 0:
            raise ValueError("Reference length must be positive")
        for key in ("mass", "kind", "movable", "object_id"):
            if len(values[key]) != self.n:
                raise ValueError(f"Invalid scene field: {key}")
        if bool((values["mass"] < 0).any()) or bool(((values["kind"] < 0) | (values["kind"] > 3)).any()):
            raise ValueError("Invalid mass or entity kind")
        for key, column in (("tet_material", 2), ("tri_material", 2), ("bend_material", 1)):
            if key in values and bool((values[key][:, column] != 0).any()):
                raise ValueError("This pilot does not implement material damping")
        self.relations, self.relation_kind, self.relation_material, self.relation_extra = self._relations()
        parents = list(range(self.n))
        def find(i):
            while parents[i] != i:
                parents[i] = parents[parents[i]]
                i = parents[i]
            return i
        for row in self.relations.tolist():
            ids = [i for i in row if i >= 0]
            for i in ids[1:]:
                parents[find(i)] = find(ids[0])
        labels = [find(i) for i in range(self.n)]
        self.component = torch.tensor(labels, device=self.device)
        self.origin = torch.zeros_like(self.rest_x)
        for label in set(labels):
            mask = self.component == label
            indices = torch.where(mask)[0]
            self.origin.index_copy_(0,indices,self.rest_x[indices].mean(0).expand(len(indices),-1))
        self.component_indices = [torch.where(self.component == label)[0] for label in sorted(set(labels))]
        self.relation_slots = []
        for slot in range(4):
            rows = torch.where(self.relations[:, slot] >= 0)[0]
            self.relation_slots.append((rows, self.relations[rows, slot]))
        self.bending_indices = None
        if "bend_edges" in values:
            self.bending_indices = torch.where((self.bend_edges >= 0).all(-1) & (self.bend_material[:, 0] > 0))[0]
        self.merge_plans = []
        objects, components = self.object_id*4+self.kind, self.component
        # Shapes and membership depend only on the static object/type layout.
        # Matching and spatial sorting still depend on each current state.
        for _ in range(8):
            groups = [torch.where(objects == group)[0] for group in torch.unique(objects, sorted=True).tolist()]
            next_objects = torch.cat([objects[ids[0]].expand(max(1, len(ids)//2)) for ids in groups])
            next_components = torch.cat([components[ids[0]].expand(max(1, len(ids)//2)) for ids in groups])
            self.merge_plans.append((groups, next_objects, next_components))
            objects, components = next_objects, next_components

    def __getattr__(self, name):
        if name in self.values:
            return self.values[name]
        raise AttributeError(name)

    @property
    def device(self):
        return self.values["rest_x"].device

    def _relations(self):
        rows, tags, materials, extras = [], [], [], []
        for key, tag, matkey in (("tets", 0, "tet_material"), ("triangles", 1, "tri_material"), ("bend_edges", 2, "bend_material"), ("joints", 3, "joint_material")):
            if key not in self.values:
                continue
            for i, row in enumerate(self.values[key].tolist()):
                if min(row) < 0:
                    continue
                mat = self.values[matkey][i]
                if not bool(mat.abs().sum() > 0):
                    continue
                rows.append(row + [-1]*(4-len(row)))
                tags.append(tag)
                material = torch.zeros(4, device=self.device)
                material[:min(4, len(mat))] = mat[:4]
                materials.append(material)
                extra = torch.zeros(16, device=self.device)
                if key == "joints":
                    extra[:3] = self.anchor_p[i]/self.length
                    extra[3:6] = self.anchor_c[i]/self.length
                    extra[6:9] = self.joint_axis[i]
                    extra[9:13] = self.joint_rest_q[i]
                    extra[13] = self.joint_tag[i]
                extras.append(extra)
        self.plane_relation_start = len(rows)
        if self.has_plane:
            for i in range(self.n):
                rows.append([i, -1, -1, -1])
                tags.append(4)
                materials.append(self.contact_material)
                extra = torch.zeros(16, device=self.device)
                extra[:3] = self.plane_normal
                extra[3] = self.plane_offset/self.length
                extra[4] = self.contact_radius/self.length
                extras.append(extra)
        return (torch.tensor(rows, device=self.device, dtype=torch.long).reshape(-1, 4), torch.tensor(tags, device=self.device), torch.stack(materials) if materials else torch.zeros(0, 4, device=self.device), torch.stack(extras) if extras else torch.zeros(0, 16, device=self.device))


def load_trajectory(path, device="cpu"):
    with np.load(path, allow_pickle=False) as archive:
        metadata = json.loads(str(archive["metadata"]))
        states = {key: torch.as_tensor(archive[key].copy(), device=device, dtype=torch.float32) for key in ("x", "q", "v", "omega")}
        static = {}
        for key in archive.files:
            if key in states or key in ("metadata", "teacher_joint_angle_history"):
                continue
            value = archive[key]
            dtype = torch.bool if value.dtype.kind == "b" else torch.long if value.dtype.kind in "iu" else torch.float32
            static[key] = torch.as_tensor(value.copy(), device=device, dtype=dtype)
    return Scene(static, metadata), states


def select_state(states, indices):
    return State(*(states[key][indices] for key in ("x", "q", "v", "omega")))
