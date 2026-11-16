"""Restrict host quaternion roundoff restoration to known free coordinates."""
import numpy as np


def restore_free_quaternions(model,recorded,newton):
    current=model.joint_q.numpy();allow=np.zeros(current.shape,dtype=bool)
    if not np.isfinite(current).all() or not np.isfinite(recorded).all():raise ValueError('Nonfinite joint coordinates cannot be restored as host roundoff')
    starts=model.joint_q_start.numpy();types=model.joint_type.numpy()
    for j in np.where(types==int(newton.JointType.FREE))[0]:allow[starts[j]+3:starts[j]+7]=True
    if current.shape!=recorded.shape or np.any((current!=recorded)&~allow):raise ValueError('Joint coordinate differences extend beyond free quaternion components')
    maximum=float(np.abs(current-recorded).max()) if current.size else 0.
    if maximum>1e-6:raise ValueError('Host quaternion difference exceeds the float32 asset bound')
    model.joint_q.assign(recorded)
    return maximum
