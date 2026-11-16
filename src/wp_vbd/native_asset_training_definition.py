"""Portable static authored inputs for qualified elastic/rod training data.

The resolved JSON is embedded in the dataset. Rebuilding never reads a future
trajectory or the possibly edited original JSON file; referenced mesh bytes
and the native builder implementation remain bound explicitly.
"""
from copy import deepcopy
from pathlib import Path

from .native_assets import SCHEMA as ASSET_SCHEMA,build,digest
from .elastic_rod_configuration import record_configuration,decode_configuration

SCHEMA='resolved_native_shared_asset_training_definition_v254'


def capture(provenance,configuration):
    if provenance['schema']!=ASSET_SCHEMA or not provenance['has_native_rods']:
        raise ValueError('This accepted-history loader requires native elastic particles and rods')
    return {'schema':SCHEMA,'asset_schema':ASSET_SCHEMA,'asset_file':provenance['asset_file'],
        'asset_sha256':provenance['asset_sha256'],'resolved_spec':deepcopy(provenance['resolved_spec']),
        'mesh_sha256':deepcopy(provenance['mesh_sha256']),
        'builder_sha256':digest(Path(__file__).with_name('native_assets.py')),
        'solver_configuration':record_configuration(configuration),
        'scope':'Embedded static asset and physical parameters, not a temporal input or training-admission certificate.'}


def rebuild(definition,newton,wp,*,device='cpu'):
    required={'schema','asset_schema','asset_file','asset_sha256','resolved_spec','mesh_sha256',
              'builder_sha256','solver_configuration','scope'}
    if not isinstance(definition,dict) or set(definition)!=required or definition['schema']!=SCHEMA or definition['asset_schema']!=ASSET_SCHEMA:
        raise ValueError('Explicit versioned native asset training definition required')
    if digest(Path(__file__).with_name('native_assets.py'))!=definition['builder_sha256']:
        raise ValueError('Native asset training builder changed')
    spec=deepcopy(definition['resolved_spec'])
    if not isinstance(spec,dict) or any(key.startswith('_') for key in spec):raise ValueError('Only public static asset fields may be embedded')
    # Reuse the public schema validator without requiring the original JSON.
    from .native_assets import fields
    fields(spec,('schema','name','gravity','simulation','contact','solver','objects','joints'),('schema','objects'))
    if spec['schema']!=ASSET_SCHEMA or not isinstance(spec['objects'],list) or not spec['objects']:
        raise ValueError('Nonempty embedded native asset required')
    if not isinstance(definition['mesh_sha256'],dict):raise ValueError('Explicit referenced-mesh hashes required')
    for path,sha in definition['mesh_sha256'].items():
        if digest(path)!=sha:raise ValueError('Native asset training mesh changed: '+path)
    path=Path(definition['asset_file'])
    spec.update(_directory=str(path.parent),_asset_file=str(path),_asset_sha256=definition['asset_sha256'])
    model,configuration,provenance=build(spec,newton,wp,device=device)
    if not model.particle_count or not provenance['has_native_rods']:
        raise ValueError('This accepted-history contract requires elastic particles and native rods')
    if provenance['resolved_spec']!=definition['resolved_spec'] or provenance['mesh_sha256']!=definition['mesh_sha256']:
        raise ValueError('Embedded native asset reconstruction changed')
    if record_configuration(configuration)!=record_configuration(decode_configuration(definition['solver_configuration'])):
        raise ValueError('Embedded native asset physical solver definition changed')
    return model,configuration,provenance
