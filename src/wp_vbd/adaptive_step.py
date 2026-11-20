"""Bounded adaptive integration using accepted states and fresh step queries."""

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class AdaptiveStepConfig:
    maximum_dt: float = 1 / 30
    minimum_dt: float = 1 / 600
    maximum_attempts: int = 64
    maximum_substeps: int = 32
    motion_safety: float = .8
    relative_update_limit: float = .15
    maximum_contact_depth: float | None = None

    def __post_init__(self):
        if not (math.isfinite(self.maximum_dt) and math.isfinite(self.minimum_dt)
                and 0 < self.minimum_dt <= self.maximum_dt):
            raise ValueError('Positive finite ordered adaptive time bounds required')
        if self.maximum_attempts < 1 or self.maximum_substeps < 1:
            raise ValueError('Positive adaptive work budgets required')
        if not 0 < self.motion_safety < 1 or not math.isfinite(self.relative_update_limit) or self.relative_update_limit <= 0:
            raise ValueError('Positive adaptive motion and update bounds required')
        if self.maximum_contact_depth is not None and (not math.isfinite(self.maximum_contact_depth) or self.maximum_contact_depth <= 0):
            raise ValueError('Positive contact-depth bound required')


class AdaptiveStepFailure(RuntimeError):
    def __init__(self, message, report):
        super().__init__(message)
        self.report = report


def motion_time_limit(scene, state, force=None, *, safety=.8):
    """Conservative forecast from current surface speed and known acceleration."""
    oracle = getattr(scene, 'geometry_oracle', None)
    envelope = getattr(oracle, 'motion_budget', None)
    if envelope is None:
        return float('inf')
    if not bool(scene.movable.any()):
        return float('inf')
    radius = getattr(scene, 'body_radius', torch.zeros_like(scene.mass))
    speed = state.v.norm(dim=-1) + state.omega.norm(dim=-1) * radius
    gravity = torch.as_tensor(scene.gravity, device=state.x.device, dtype=state.x.dtype)
    acceleration = gravity.expand_as(state.x)
    if force is not None:
        acceleration = acceleration + force / scene.mass.clamp_min(1e-30)[None, :, None]
    speed = float(speed[:, scene.movable].amax())
    acceleration = float(acceleration[:, scene.movable].norm(dim=-1).amax())
    distance = safety * float(envelope)
    # Stable positive root of speed*h + acceleration*h*h = distance.
    return 2 * distance / max(speed + math.sqrt(speed * speed + 4 * acceleration * distance), 1e-30)


class AdaptiveStepper:
    """Retry rejected proposals without publishing their state or material history."""

    def __init__(self, config=None):
        self.config = config or AdaptiveStepConfig()

    def assess(self, current, candidate, info, h, length_scale):
        """Use identical admission rules for ordinary and repaired proposals."""
        finite = all(bool(torch.isfinite(getattr(candidate, key)).all())
                     for key in ('x', 'q', 'v', 'omega'))
        if candidate.history:
            finite = finite and all(bool(torch.isfinite(v).all()) for v in candidate.history.values())
        accepted = bool(torch.as_tensor(info.get('accepted', False)).all()) and finite
        reason = 'accepted' if accepted else 'solver_rejected_or_nonfinite'
        gap = info.get('physical_contact_gap')
        if self.config.maximum_contact_depth is not None:
            minimum = info.get('minimum_contact_normal_physical_separation')
            depth = float((-gap).clamp_min(0).amax()) if gap is not None and gap.numel() else (
                -float(minimum) if minimum is not None else 0.)
            if depth > self.config.maximum_contact_depth:
                accepted, reason = False, 'contact_depth'
        update = float((candidate.v-current.v).norm(dim=-1).amax()*h/length_scale)
        if accepted and update > self.config.relative_update_limit:
            accepted, reason = False, 'relative_velocity_change'
        return accepted, reason, update

    def advance(self, state, duration, try_step, *, start_time=0., events=(),
                step_limit=None, previous_dt=None, length_scale=1., repair_step=None):
        if not math.isfinite(duration) or duration <= 0 or not math.isfinite(start_time):
            raise ValueError('Finite positive requested duration and finite start time required')
        if not math.isfinite(length_scale) or length_scale <= 0:
            raise ValueError('Positive physical length scale required')
        end = start_time + duration
        if not math.isfinite(end) or end<=start_time:
            raise ValueError('Requested physical endpoint must be finite and representable')
        events=tuple(float(t) for t in events)
        if any(not math.isfinite(t) for t in events):
            raise ValueError('Finite input event times required')
        boundaries = sorted({t for t in events if start_time < t < end}) + [end]
        # A hard event closer than the resolution floor cannot be represented
        # without violating the caller's explicit minimum timestep.
        intervals=[b-a for a,b in zip((start_time,*boundaries[:-1]),boundaries)]
        if any(value < self.config.minimum_dt-max(1e-14,duration*1e-12) for value in intervals):
            raise ValueError('Requested interval or hard event spacing is below minimum_dt')
        tolerance = max(1e-14, duration * 1e-12)
        for value in intervals:
            if math.ceil((value-tolerance)/self.config.maximum_dt) > math.floor((value+tolerance)/self.config.minimum_dt):
                raise ValueError('Requested interval cannot satisfy both adaptive time bounds')
        if previous_dt is not None and (not math.isfinite(previous_dt) or previous_dt<=0):
            raise ValueError('Positive finite previous accepted step required')
        current = state
        time = start_time
        suggested = self.config.maximum_dt if previous_dt is None else min(
            self.config.maximum_dt, 2 * previous_dt)
        attempts = []
        committed = 0
        diagnostics = []
        first_rejected_info = None
        last_rejected_info = None
        def partition(remaining, target):
            requested_parts=max(1,math.ceil(remaining/max(target,self.config.minimum_dt)-1e-10))
            maximum_parts=max(1,math.floor((remaining+tolerance)/self.config.minimum_dt))
            return remaining/min(requested_parts,maximum_parts)

        def summarize(h, time, candidate, info, mode='proposal'):
            accepted, reason, update = self.assess(current, candidate, info, h, length_scale)
            return {'dt':h, 'time':time, 'accepted':accepted, 'reason':reason,
                    'relative_velocity_change':update, 'mode':mode,
                    'network_rounds':len(info.get('iterations', ()))}

        def failure(message):
            def compact(value):
                if isinstance(value,torch.Tensor):
                    if value.numel()<=32:return value.detach().cpu().tolist()
                    return {'shape':list(value.shape),'finite':bool(torch.isfinite(value).all()),
                            'minimum':float(value.min()),'maximum':float(value.max())}
                if isinstance(value,dict):return {k:compact(v) for k,v in value.items()}
                if isinstance(value,(list,tuple)):return [compact(v) for v in value[:12]]
                return value
            def evidence(info):
                if info is None:return None
                keys=('accepted','finite','min_alpha','predictor_motion_valid','predictor_physics_domain_valid',
                      'final_physics_domain_valid','final_motion_domain_valid','contact_motion_domain_valid',
                      'contact_metric_ray_scales','separate_subspace_ray_scales','contact_activation_ray_scales')
                result={key:compact(info[key]) for key in keys if key in info}
                result['iterations']=[{**{k:compact(v) for k,v in row.items() if k!='selected'},
                    'rejected_entities':int((~torch.as_tensor(row['selected'],dtype=torch.bool)).sum()) if 'selected' in row else None}
                    for row in info.get('iterations',())]
                return result
            return AdaptiveStepFailure(message, {
                'advanced_seconds':time-start_time, 'attempts':attempts,
                'substeps':committed, 'retries':len(attempts)-committed,
                'network_rounds':sum(row['network_rounds'] for row in attempts),
                'first_rejected_diagnostic':evidence(first_rejected_info),
                'last_rejected_diagnostic':evidence(last_rejected_info)})

        while end - time > tolerance:
            if len(attempts) >= self.config.maximum_attempts or committed >= self.config.maximum_substeps:
                raise failure('Adaptive work budget exhausted')
            boundary = next(t for t in boundaries if t > time + tolerance)
            limit = step_limit(current, time) if step_limit is not None else float('inf')
            if math.isnan(limit) or limit <= 0:
                raise ValueError('The adaptive step limit must be positive')
            h = min(suggested, limit, boundary - time)
            remaining=boundary-time
            # Split the complete remainder into equal representable intervals.
            # Clamping h alone can leave a final sliver below minimum_dt.
            h=partition(remaining,h)
            # An isolated attempt owns a clone of all persistent tensors.
            candidate, info = try_step(current.clone(), h, time)
            attempts.append(summarize(h,time,candidate,info))
            accepted=attempts[-1]['accepted']
            if not accepted:
                if first_rejected_info is None:first_rejected_info=info
                last_rejected_info=info
                # Event/end alignment can make the smallest representable step
                # slightly larger than the floor. Do not retry that same step forever.
                at_floor=partition(remaining,max(self.config.minimum_dt,h/2)) >= h-tolerance
                if at_floor and repair_step is not None:
                    allowance=self.config.maximum_attempts-len(attempts)
                    candidate,info=repair_step(current.clone(),h,time,info,allowance)
                    last_rejected_info=info
                    repairs=info.get('adaptive_recovery',{}).get('attempts',[])
                    if len(repairs)>allowance:
                        raise RuntimeError('Recovery exceeded its solver-attempt allowance')
                    attempts.extend({'dt':h,'time':time,**row} for row in repairs)
                    accepted,_,_=self.assess(current,candidate,info,h,length_scale)
                    if accepted and (not repairs or not repairs[-1]['accepted']):
                        raise RuntimeError('Recovery accepted an unaccounted proposal')
                if at_floor and not accepted:
                    raise failure('Adaptive repair failed at minimum representable step')
            if not accepted:
                suggested=max(self.config.minimum_dt,h/2)
                continue
            current = candidate
            time += h
            committed += 1
            diagnostics.append(info)
            suggested = min(self.config.maximum_dt, h * 2)
        return current, {
            'accepted': True, 'advanced_seconds': duration, 'substeps': committed,
            'retries': len(attempts) - committed, 'attempts': attempts,
            'last_dt': attempts[-1]['dt'], 'diagnostics': diagnostics,
            'network_rounds': sum(row['network_rounds'] for row in attempts),
        }
