"""Original-demo time grids and causal variable-duration supervised windows."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import torch


@dataclass(frozen=True)
class ReferenceGrid:
    times: np.ndarray
    segments: tuple[tuple[int, int], ...]

    def __post_init__(self):
        if self.times.ndim!=1 or len(self.times)<2 or not np.isfinite(self.times).all():
            raise ValueError('At least two finite teacher timestamps required')
        if not self.segments:
            raise ValueError('At least one continuous teacher segment required')
        for first,last in self.segments:
            if not 0<=first<last<len(self.times) or not (np.diff(self.times[first:last+1])>0).all():
                raise ValueError('Teacher segments need strictly increasing state timestamps')


def reference_grid(path,fallback_dt=None):
    """Read sample counts from NumPy headers without loading complete trajectories."""
    path=Path(path)
    with zipfile.ZipFile(path) as archive:
        names=archive.namelist()
        dynamic=next((name for name in ('x.npy','particle_position.npy','body_pose.npy')
                       if name in names),None)
        if dynamic is None:
            raise ValueError('Reference archive has no recognizable state array')
        stream=archive.open(dynamic)
        version=np.lib.format.read_magic(stream)
        shape,_,_=np.lib.format._read_array_header(stream,version)
        stream.close()
    count=shape[0]
    with np.load(path,allow_pickle=False) as archive:
        metadata=json.loads(str(archive['metadata']))
        explicit=next((key for key in ('physical_time','time','timestamps') if key in archive),None)
        if explicit:
            if archive[explicit].shape!=(count,):
                raise ValueError('One physical timestamp per recorded state required')
            times=np.asarray(archive[explicit],dtype=np.float64)
        else:
            recorded_dt=metadata.get('dt',fallback_dt)
            if recorded_dt is None:
                raise ValueError('Original demo timestamps or timestep required')
            dt=float(recorded_dt)
            if not math.isfinite(dt) or dt<=0:
                raise ValueError('Positive original demo timestep required')
            times=np.arange(count,dtype=np.float64)*dt
    if metadata.get('data_layout')=='teacher_queried_windows_v1':
        horizon=metadata['qualified_window_horizon']
        with np.load(path,allow_pickle=False) as archive:
            starts=archive['qualified_starts'].astype(np.int64).tolist()
        segments=tuple((i,i+horizon) for i in starts if i+horizon<count)
    else:
        segments=((0,count-1),)
    return ReferenceGrid(times,segments),metadata


def command_events(path, count, *, hard_only=False):
    """Separate discrete geometry/controller events from ordinary control samples."""
    events=set()
    with np.load(path,allow_pickle=False) as archive:
        constants=set(json.loads(str(archive['metadata'])).get('constant_time_inputs',()))
        if 'caller__steps_per_sample' in archive:
            period=int(archive['caller__steps_per_sample'])
            if period<1:
                raise ValueError('Positive original caller clock period required')
            events.update(range(period,count-1,period))
        for key in archive.files:
            if not key.startswith('input_') or key in ('input_force','input_torque'):
                continue
            if hard_only and key!='input_geometry_removal_mask':
                continue
            value=archive[key]
            if value.ndim<1 or not value.size:
                continue
            if len(value)==1 and key in constants:
                continue
            if len(value) not in (count-1,count):
                raise ValueError('Input timing does not match teacher states: '+key)
            flat=value.reshape(len(value),-1)
            changed=np.any(flat[1:]!=flat[:-1],axis=1)
            events.update((np.flatnonzero(changed)+1).tolist())
            if key=='input_geometry_removal_mask':
                active=np.flatnonzero(flat.any(-1))
                events.update(active.tolist())
                events.update((active+1).tolist())
    return sorted(i for i in events if 0<i<count-1)


def long_step_window(grid,start,target_dts,*,events=(),allow_nearest=True):
    target_dts=tuple(target_dts)
    events=sorted(set(events))
    if type(start) is not int or not target_dts or any(not math.isfinite(h) or h<=0 for h in target_dts):
        raise ValueError('An integer start and positive target time intervals required')
    segment=max(((a,b) for a,b in grid.segments if a<=start<b),key=lambda pair:pair[1],default=None)
    if segment is None:
        raise ValueError('Window start is outside a continuous teacher segment')
    last=segment[1]
    endpoints=[]
    ideal=float(grid.times[start])
    cursor=start
    for requested in target_dts:
        ideal=float(grid.times[cursor])+requested
        if ideal>grid.times[last]+1e-10:
            raise ValueError('Requested window exceeds the continuous teacher segment')
        upper=int(np.searchsorted(grid.times[start:last+1],ideal))+start
        choices=[j for j in (upper-1,upper) if cursor<j<=last]
        if not choices:
            raise ValueError('Requested interval is finer than the recorded teacher grid')
        endpoint=min(choices,key=lambda j:abs(grid.times[j]-ideal))
        if not allow_nearest and abs(grid.times[endpoint]-ideal)>1e-10:
            raise ValueError('Requested endpoint does not align with the original teacher grid')
        endpoints.extend(i for i in events if cursor<i<endpoint)
        endpoints.append(endpoint)
        cursor=endpoint
    indices=[start,*endpoints]
    return {
        'start':start,'strides':[b-a for a,b in zip(indices,indices[1:])],
        'durations':[float(grid.times[b]-grid.times[a]) for a,b in zip(indices,indices[1:])],
        'requested_durations':list(target_dts),
        'actual_duration':float(grid.times[cursor]-grid.times[start]),
        'start_time':float(grid.times[start]),
        'endpoint_times':[float(grid.times[i]) for i in endpoints],
        'teacher_durations':np.diff(grid.times[start:cursor+1]).tolist(),
        'endpoint_indices':endpoints,
        'extra_event_steps':len(endpoints)-len(target_dts),
    }


def stride_pattern(teacher_dt, substeps, frame_dt=1 / 60):
    """Split a display interval into integer teacher strides that sum exactly."""
    ratio = frame_dt / teacher_dt
    count = round(ratio)
    if abs(ratio - count) > 1e-5 or not 1 <= substeps <= count:
        raise ValueError('The teacher grid cannot represent this frame budget')
    edges = [i * count // substeps for i in range(substeps + 1)]
    return tuple(b - a for a, b in zip(edges, edges[1:]))


def inputs_for_span(states, start, stride, *, durations=None):
    """Reduce known controls over one macro step without reading target states.

    Wrenches/drive inputs preserve their time integral. Prescribed poses use
    the last command in the interval; their accepted-pose history supplies
    the net displacement over the full macro duration. Smooth command changes
    never force additional dynamics calls.
    """
    result = {}
    if start<0 or stride<1:
        raise ValueError('Nonnegative start and positive teacher stride required')
    if durations is not None:
        weights=torch.as_tensor(durations,dtype=torch.float64)
        if weights.shape!=(stride,) or not bool(torch.isfinite(weights).all() & (weights>0).all()):
            raise ValueError('One positive finite duration per teacher interval required')
        weights=weights/weights.sum()
    for key, value in states.items():
        if not key.startswith('input_'):
            continue
        rows = value[start:start + stride]
        if len(rows) != stride:
            raise ValueError('Incomplete exogenous input span: ' + key)
        if key == 'input_geometry_removal_mask' and stride > 1 and bool(rows.any()):
            raise ValueError('Geometry removal requires a separate integration interval')
        if key in ('input_force','input_torque','input_joint_target',
                   'input_joint_target_rate','input_joint_effort'):
            if durations is None:
                result[key] = rows.mean(0, keepdim=True)
            else:
                weight=weights.to(rows).reshape(stride,*([1]*(rows.ndim-1)))
                result[key]=(rows*weight).sum(0,keepdim=True)
        else:
            if key in ('input_x','input_q','input_pin_x','input_pin_q',
                       'input_prescribed_x','input_prescribed_q',
                       'input_prescribed_v','input_prescribed_omega'):
                result[key]=rows[-1:]
                continue
            if stride > 1 and not torch.equal(rows, rows[:1].expand_as(rows)):
                raise ValueError('The held input changes within this span: ' + key)
            result[key] = rows[:1]
    return result
