"""Explicit per-body gravity for native packed worlds, including world -1."""
import numpy as np
import torch

CONTRACT = 'body_world_gravity_v32'


def export(model):
    gravity = model.gravity.numpy().reshape(-1, 3)
    body_world = model.body_world.numpy()
    shape_world = model.shape_world.numpy()
    world_count = int(model.world_count)
    if gravity.shape != (world_count + 1, 3) or not np.isfinite(gravity).all():
        raise ValueError('Packed gravity table must contain each world and the global -1 slot')
    if body_world.shape != (model.body_count,) or np.any((body_world < -1) | (body_world >= world_count)):
        raise ValueError('Invalid body world ownership')
    if shape_world.shape != (model.shape_count,) or np.any((shape_world < -1) | (shape_world >= world_count)):
        raise ValueError('Invalid shape world ownership')
    owner = model.shape_body.numpy(); attached = owner >= 0
    if not np.array_equal(shape_world[attached], body_world[owner[attached]]):
        raise ValueError('Shape world must agree with its body ownership')
    mapped = np.concatenate((gravity[body_world], np.zeros((1, 3), gravity.dtype)), axis=0)
    return {'body_world': body_world.copy(), 'shape_world': shape_world.copy(), 'gravity_table': gravity.copy(), 'gravity': mapped}, {'world_gravity_contract': CONTRACT, 'packed_world_count': world_count}


def refresh(scene, model, wp):
    table = wp.to_torch(model.gravity).reshape(-1, 3)
    expected = scene.metadata['packed_world_count']
    if int(model.world_count) != expected or table.shape != (expected + 1, 3) or not bool(torch.isfinite(table).all()):
        raise ValueError('Packed gravity layout changed or contains nonfinite values; rebuild the scene for layout changes')
    scene.values['gravity_table'] = table.double()
    indices = scene.values['body_world'].long()
    # Negative native ownership indexes the final global slot exactly.
    scene.gravity = torch.cat((table[indices].double(), table.new_zeros(1, 3).double()), dim=0)
    scene.values['gravity'] = scene.gravity
