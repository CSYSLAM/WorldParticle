"""Host-side keys for topology derived only from versioned static tensors.

The cached plan must retain the tensors themselves to prevent identity reuse.
State-dependent Jacobians, factors and responses are never cached here.
"""


def tensor_versions(tensors):
    try:
        return tuple((id(value), value._version) for value in tensors)
    except RuntimeError:
        # Inference tensors do not expose mutation counters. Rebuild their
        # topology instead of assuming that an unversioned buffer is constant.
        return None
