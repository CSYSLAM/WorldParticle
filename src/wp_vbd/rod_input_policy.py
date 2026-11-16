"""Explicit collision-cadence change for owned, prescribed rod scenes."""


def refresh_every_substep(example, identifier, configurations):
    """Change the numerical detection cadence, preserving material and control."""
    if identifier not in ('cable.example_cable_twist','cable.example_cable_bundle_hysteresis'):
        raise ValueError('Fresh prescribed-rod policy requires a supported authored control adapter')
    if len(configurations) != 1 or configurations[0][1].get('rigid_contact_history'):
        raise ValueError('Persistent matched-contact memory needs its separate contract')
    original = getattr(example, 'update_step_interval', None)
    if not isinstance(original, int) or original < 1:
        raise ValueError('An explicit positive source collision interval is required')
    example.update_step_interval = 1
    return {'policy':'every_physical_substep', 'source_refresh_interval':original,
            'reference_refresh_interval':1, 'authored_collision_cadence_preserved':original==1,
            'physical_material_asset_control_preserved':True}
