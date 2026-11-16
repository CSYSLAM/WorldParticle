"""Portable explicit particle filters and the mixed teacher force-scope check."""
import hashlib
import json

from .official import solver_configuration_record

FILTER_KEYS=('particle_external_vertex_contact_filtering_map',
             'particle_external_edge_contact_filtering_map')


def filter_record(value):
    if value is None:return None
    if not isinstance(value,dict):raise ValueError('Explicit typed particle filter map required')
    result={}
    for key,targets in value.items():
        if isinstance(key,str) and key.isdecimal() and str(int(key))==key:key=int(key)
        if type(key) is not int or key<0 or key in result:
            raise ValueError('Unique nonnegative particle filter keys required')
        if not isinstance(targets,(list,tuple,set,frozenset)) or any(type(i) is not int or i<0 for i in targets):
            raise ValueError('Nonnegative integer particle filter targets required')
        if len(set(targets))!=len(targets):raise ValueError('Duplicate particle filter targets')
        result[key]=sorted(targets)
    return dict(sorted(result.items()))


def decode_configuration(configuration):
    if not isinstance(configuration,dict):raise ValueError('Explicit mixed configuration required')
    return {**configuration,**{key:filter_record(configuration.get(key)) for key in FILTER_KEYS}}


def record_configuration(configuration):
    result=solver_configuration_record(configuration)
    result.update({key:filter_record(configuration.get(key)) for key in FILTER_KEYS})
    return result


def topology_definition(query,configuration):
    """Compare source directed exclusions with the reciprocal row contract.

    Uses only static topology, including explicit source filters; no collision
    query, force evaluation, solver, trajectory or future label is inspected.
    """
    configuration=decode_configuration(configuration)
    native=[{int(i) for i in row} for row in query.source_topological_edge_filter_sets]
    external=configuration[FILTER_KEYS[1]] or {}
    for i,row in external.items():
        if i>=len(native) or any(j>=len(native) for j in row):raise ValueError('Native edge filter out of range')
        native[i].update(row)
    asymmetric=[(i,j) for i,row in enumerate(native) for j in sorted(row) if i not in native[j]]
    canonical=lambda rows:json.dumps([sorted(int(i) for i in row) for row in rows],separators=(',',':')).encode()
    compatible=not asymmetric and native==query.filter_sets[1]
    return {'schema':'native_source_EE_force_scope_v233','compatible':compatible,
        'source_asymmetric_EE_exclusion_count':len(asymmetric),
        'native_directional_exclusions_sha256':hashlib.sha256(canonical(native)).hexdigest(),
        'reciprocal_union_exclusions_sha256':hashlib.sha256(canonical(query.filter_sets[1])).hexdigest(),
        'policy':'source_EE_ownership_matches_reciprocal_rows' if compatible else 'source_directional_EE_differs_from_reciprocal_rows'}


def reciprocal_configuration(model,configuration):
    """Explicit source profile for new authored assets, never official demos."""
    from .self_contact_query import SelfContactQuery
    configuration=decode_configuration(configuration)
    query=SelfContactQuery(model,configuration,motion_budget=.02)
    result={**configuration,FILTER_KEYS[1]:{i:sorted(int(j) for j in row) for i,row in enumerate(query.filter_sets[1]) if row}}
    evidence=topology_definition(query,result)
    if not evidence['compatible']:raise ValueError('Reciprocal source topology profile is incomplete')
    return result,evidence
