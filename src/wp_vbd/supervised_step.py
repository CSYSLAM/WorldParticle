"""Bounded, causal repairs on the integer grid of an offline reference.

No target states enter the dynamics callback. Accepted student states and
histories propagate across repaired subintervals; all attempts are counted.
"""
import math

from .data import State
from .multirate_window import inputs_for_span


def advance_reference_span(current,states,start,stride,reference_dt,advance,*,
                           minimum_dt=1/600,maximum_attempts=15):
    if (start<0 or stride<1 or not math.isfinite(reference_dt) or reference_dt<=0 or
        not math.isfinite(minimum_dt) or minimum_dt<=0 or maximum_attempts<1):
        raise ValueError('Positive, bounded reference-span repair configuration required')
    minimum_stride=max(1,math.ceil((minimum_dt-1e-12)/reference_dt))
    report={'attempts':[],'neural_rounds':0,'physical_response_rounds':0,
            'accepted_substeps':0,'rejected_attempts':0,'minimum_accepted_dt':None}
    def snapshot(state):
        return State(*(getattr(state,key).clone() for key in ('x','q','v','omega')),
                     {k:v.clone() for k,v in state.history.items()} if state.history is not None else None)
    def trial(state,first,count):
        dt=count*reference_dt
        if dt+1e-12<minimum_dt:raise RuntimeError('Reference-span repair would cross the explicit 1/600s model floor')
        if len(report['attempts'])>=maximum_attempts:raise RuntimeError('Reference-span repair exceeded its explicit attempt budget')
        actions=inputs_for_span(states,first,count,durations=[reference_dt]*count)
        proposal,info=advance(snapshot(state),actions,dt)
        accepted=bool(info['accepted'].all())
        phases=info['iterations']
        report['neural_rounds']+=sum(item.get('phase')=='neural_correction' for item in phases)
        report['physical_response_rounds']+=len(phases)
        record={'start':first,'stride':count,'dt':dt,'accepted':accepted}
        report['attempts'].append(record)
        if accepted:
            report['accepted_substeps']+=1
            report['minimum_accepted_dt']=dt if report['minimum_accepted_dt'] is None else min(dt,report['minimum_accepted_dt'])
            return proposal,info
        report['rejected_attempts']+=1
        for key in ('finite','geometric_step_domain_valid','contact_motion_domain_valid'):
            if key in info:
                value=info[key];record[key]=bool(value.all()) if hasattr(value,'all') else bool(value)
        for key in ('rejected_entities','numerical_root_entities'):
            if key in info:record[key]=int(info[key].sum())
        # Release rejected graphs before constructing differentiable repairs.
        del proposal,info
        if count<2*minimum_stride:
            raise RuntimeError('Student proposal rejected at the explicit model time floor: '+str(record))
        first_count=max(minimum_stride,min(count//2,count-minimum_stride))
        midpoint,_=trial(state,first,first_count)
        return trial(midpoint,first+first_count,count-first_count)
    result,info=trial(current,start,stride)
    info={**info,'adaptive_reference_span':report}
    return result,info,report
