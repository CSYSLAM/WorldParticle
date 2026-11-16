"""Host scalar boundary adaptation, without changing Warp or Newton files."""
from contextlib import contextmanager
import types
import numpy as np


@contextmanager
def warp_host_scalars(module,wp):
    """Accept NumPy host angles at the original Warp float32 quaternion call.

    Only this example module's temporary namespace changes during construction.
    Kernels run after restoration; global Warp functions and physics are intact.
    The resulting float32 asset must still match the archived static reference.
    """
    original=module.wp
    namespace=types.ModuleType(wp.__name__);namespace.__dict__.update(wp.__dict__)
    def quaternion(axis,angle):
        if isinstance(angle,np.floating):angle=float(angle)
        return wp.quat_from_axis_angle(axis,angle)
    namespace.quat_from_axis_angle=quaternion
    module.wp=namespace
    try:yield
    finally:module.wp=original
