"""Chunked, pickle-free physical-substep reference records for operator work.

This offline format does not admit a demo to training. Fields are named by
owner and evaluation time; output contacts/history are labels, never inputs.
"""

import hashlib
import json
from pathlib import Path

import numpy as np


SCHEMA = 'newton_vbd_substep_trace_v1'
TRACE_FEATURE_SET = 3
EXTERNAL_TRACE_FEATURE_SET = 4
EXTERNAL_INERTIAL_FIELDS = ('particle_mass','particle_inv_mass','body_mass','body_inv_mass','body_com','body_inertia','body_inv_inertia')

# Pinned SolverVBD's persistent numerical/material state, plus step setup
# outputs needed to check a new implementation. Scratch forces are excluded.
SOLVER_FIELDS = (
    'particle_q_prev', 'body_q_prev', 'pos_prev_collision_detection',
    'body_q_prev_collision_detection', '_coupling_particle_q_prev_snapshot',
    '_coupling_body_q_prev_snapshot', '_rigid_pose_rebaseline_mask',
    '_contact_history_reset_mask', '_contact_history_reset_pending',
    'joint_angle_prev', 'joint_lambda_lin', 'joint_lambda_ang',
    'joint_C0_lin', 'joint_C0_ang', 'joint_penalty_k', 'joint_rho',
    'joint_drive_limit_support', 'joint_drive_lambda', 'joint_limit_lambda',
    'joint_sigma_prev', 'joint_kappa_prev', 'joint_dkappa_prev',
    'joint_sigma_start', 'joint_C_fric',
    'body_body_contact_penalty_k', 'body_body_contact_normal_rho',
    'body_body_contact_tangent_rho', 'body_body_contact_lambda',
    'body_body_contact_C0', 'body_particle_contact_penalty_k',
    '_prev_contact_lambda', '_prev_contact_penalty_k', '_prev_contact_normal',
    '_prev_contact_frame',
)

MUTABLE_MODEL_FIELDS = (
    'joint_enabled', 'joint_target_q', 'joint_target_qd', 'joint_target_ke',
    'joint_target_kd', 'joint_limit_lower', 'joint_limit_upper', 'joint_limit_ke',
    'joint_limit_kd', 'joint_act', 'joint_X_p', 'joint_X_c', 'particle_flags', 'body_flags', 'gravity',
    'shape_transform', 'shape_scale', 'shape_flags', 'particle_radius', 'shape_material_ke', 'shape_material_kd',
    'shape_material_mu', 'attachment_particle_body_enabled',
    'attachment_particle_body_stiffness', 'attachment_particle_body_damping',
    'soft_contact_ke', 'soft_contact_kd', 'soft_contact_kf', 'soft_contact_mu',
)


def arrays(owner, *, names=None, prefix='', depth=1):
    """Copy numeric fields without serializing device pointers or Python objects."""
    import warp as wp
    result = {}
    if owner is None:
        return result
    if names is None and owner.__class__.__name__=='ModelView':
        # Public view properties live on the parent and in an override map;
        # vars(view) alone contains only private implementation dictionaries.
        keys=set(vars(object.__getattribute__(owner,'_parent')))|set(object.__getattribute__(owner,'_overrides'))
        fields={name:getattr(owner,name,None) for name in sorted(keys) if not name.startswith('_')}
    elif names is None and owner.__class__.__name__=='_AttributeNamespaceView':
        keys=set(vars(object.__getattribute__(owner,'_parent_namespace')))|set(vars(owner))
        fields={name:getattr(owner,name,None) for name in sorted(keys) if not name.startswith('_')}
    else:fields = vars(owner) if names is None else {name: getattr(owner, name, None) for name in names}
    for name, value in fields.items():
        if names is None and name.startswith('_'):
            continue
        if name.endswith('_ptr'):
            continue
        key = prefix + name
        if isinstance(value, wp.array):
            data = value.numpy().copy()
            if data.dtype.hasobject:
                raise ValueError(f'Object array is not a portable reference field: {key}')
            result[key] = data
        elif isinstance(value, np.ndarray) and not value.dtype.hasobject:
            result[key] = value.copy()
        elif isinstance(value, (bool, int, float, np.bool_, np.integer, np.floating)):
            result[key] = np.asarray(value)
        elif depth and hasattr(value, '__dict__') and value.__class__.__name__ in ('SimpleNamespace', 'ModelAttributeNamespace', 'AttributeNamespace', '_AttributeNamespaceView', 'Namespace'):
            result.update(arrays(value, prefix=key + '__', depth=depth - 1))
    return result


def auxiliary(solver):
    result = arrays(solver, names=SOLVER_FIELDS)
    # Explicit scalar consumption flags are part of the input contract too.
    for name in ('_update_rigid_history', '_last_contact_history_buffer_id'):
        if hasattr(solver, name):
            result[name] = np.asarray(getattr(solver, name))
    return result


def contact_arrays(contacts):
    if contacts is None:
        return {}
    result = {}
    import warp as wp
    counts = {kind: int(getattr(contacts, kind + '_contact_count').numpy()[0]) for kind in ('rigid', 'soft')}
    for kind, count in counts.items():
        if count < 0 or count > getattr(contacts, kind + '_contact_max'):
            raise ValueError(f'{kind} contact buffer overflow: {count}')
    for name, value in vars(contacts).items():
        if not isinstance(value, wp.array):
            continue
        if name in ('contact_counters', 'contact_generation', '_reduction_overflow', '_soft_contact_source_overflow'):
            result[name] = value.numpy().copy()
        elif name.startswith(('rigid_contact_', 'soft_contact_')):
            kind = name.split('_')[0]
            if name.endswith(('count', 'generation')):
                result[name] = value.numpy().copy()
            elif name.endswith(('tids', 'new_indices', 'broken_indices', 'force')):
                continue
            else:
                result[name] = value.numpy()[:counts[kind]].copy()
    if any(int(result.get(name, np.array([0]))[0]) for name in ('_reduction_overflow', '_soft_contact_source_overflow')):
        raise ValueError('Reference contact candidates overflowed')
    return result


def self_contact_arrays(solver):
    info = getattr(solver, 'trimesh_collision_info', None)
    if info is None:
        return {}
    result = {}
    for name, width in (('vertex_colliding_triangles', 2), ('triangle_colliding_vertices', 1), ('edge_colliding_edges', 2)):
        count_array = getattr(info, name + '_count')
        # The reverse triangle->vertex table is optional in the pinned API.
        # Its absence is not a disabled vertex-face/edge-edge collision law.
        if count_array is None:
            continue
        counts = count_array.numpy().copy()
        capacity = getattr(info, name + '_buffer_sizes').numpy()
        offsets = getattr(info, name + '_offsets').numpy()
        if np.any(counts > capacity) or np.any(counts < 0):
            raise ValueError(f'Self-contact row overflow: {name}')
        raw = getattr(info, name).numpy().reshape(-1, width)
        selected = np.concatenate([raw[offsets[i]:offsets[i] + count] for i, count in enumerate(counts) if count], 0) if np.any(counts) else np.empty((0, width), np.int32)
        result[name] = selected
        result[name + '_count'] = counts
        result[name + '_min_dist'] = getattr(info, name + '_min_dist').numpy().copy()
    return result


def matcher_arrays(pipelines):
    result = {}
    for index, pipeline in enumerate(pipelines):
        matcher = getattr(pipeline, '_contact_matcher', None)
        if matcher is None:
            continue
        names = [name for name in vars(matcher) if name.startswith('_prev_') or name == '_reset_world_mask']
        result.update(arrays(matcher, names=names, prefix=f'pipeline{index}__'))
        result[f'pipeline{index}___last_buffer_id'] = np.asarray(matcher._last_buffer_id)
    return result


class TapeWriter:
    def __init__(self, directory, *, chunk_steps=16, max_bytes=20 * 1024**3):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=False)
        self.chunk_steps = chunk_steps
        self.max_bytes = max_bytes
        self.chunks = []
        self.pending = []
        self.data = {}
        self.dedup = {}
        self.steps = 0
        self.bytes = 0

    def append(self, fields, metadata):
        refs = {}
        for name, value in fields.items():
            value = np.array(value, copy=True, order='C')
            if value.dtype.hasobject:
                raise ValueError('Reference tapes never use pickle')
            digest = hashlib.sha256(value.tobytes()).hexdigest()
            signature = (value.dtype.str, str(value.dtype.descr), value.shape, digest)
            if signature not in self.dedup:
                key = f'a{len(self.data):06d}'
                self.data[key] = value
                self.dedup[signature] = key
            refs[name] = self.dedup[signature]
        self.pending.append({'metadata': metadata, 'fields': refs})
        self.steps += 1
        if len(self.pending) >= self.chunk_steps:
            self.flush()

    def flush(self):
        if not self.pending:
            return
        path = self.directory / f'chunk_{len(self.chunks):06d}.npz'
        np.savez_compressed(path, **self.data, records=np.asarray(json.dumps(self.pending)))
        self.bytes += path.stat().st_size
        self.chunks.append({'path': path.name, 'steps': len(self.pending), 'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        self.pending, self.data, self.dedup = [], {}, {}
        if self.bytes > self.max_bytes:
            raise ValueError('Reference tape exceeded its explicit disk budget')

    def manifest(self, metadata):
        self.flush()
        result = {**metadata, 'schema': SCHEMA, 'physical_vbd_calls': self.steps, 'chunks': self.chunks, 'compressed_bytes': self.bytes, 'training_admission': False}
        path = self.directory / 'manifest.json'
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, indent=2) + '\n')
        temporary.replace(path)
        return result


def read_tape(directory):
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    if manifest['schema'] != SCHEMA:
        raise ValueError('Unknown physical-substep schema')
    for chunk in manifest['chunks']:
        path = directory / chunk['path']
        if hashlib.sha256(path.read_bytes()).hexdigest() != chunk['sha256']:
            raise ValueError(f'Reference chunk hash changed: {path}')
        with np.load(path, allow_pickle=False) as archive:
            # np.load's zip reader does not cache arrays. A shared static blob
            # may occur hundreds of times in a chunk's field references.
            # Decompress each blob once, retaining owned copies per field.
            data = {key: archive[key] for key in archive.files if key != 'records'}
            for record in json.loads(str(archive['records'])):
                yield {name: data[key].copy() for name, key in record['fields'].items()}, record['metadata']
