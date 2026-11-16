"""Accepted-entry history and substep impulses for restarted coupling passes."""
import math

import torch

from .physical_loads import CONTACT_FAMILIES,FAMILIES,proxy_feedback


def clone_history(history):
    if history is None:return None
    if not isinstance(history,dict) or any(not isinstance(value,torch.Tensor) or not bool(torch.isfinite(value).all()) for value in history.values()):
        raise ValueError('Finite tensor material history required')
    return {key:value.detach().clone() for key,value in history.items()}


def history_schema(history):
    return None if history is None else {key:(value.shape,value.dtype,value.device) for key,value in history.items()}


class CoupledStepTransaction:
    """Keep a pass-0 baseline while a solver advances tentative substeps.

    Restart and abort return that same baseline, including physical material
    memory and the step counter. Feedback is the time-integrated, ownership-
    filtered physical force, exposed only after the whole entry interval.
    History-only mode supports a primary solver receiving external forces.
    """
    def __init__(self,node_count,*,proxy_mask=None,owned_response=None):
        if type(node_count) is not int or node_count<1:raise ValueError('Positive physical node count required')
        if (proxy_mask is None)!=(owned_response is None):raise ValueError('Both explicit ownership masks are required')
        if proxy_mask is not None:
            for value in (proxy_mask,owned_response):
                if value.dtype!=torch.bool or value.shape!=(node_count,):raise ValueError('Complete boolean physical ownership masks required')
            if proxy_mask.device!=owned_response.device or bool((proxy_mask&owned_response).any()):
                raise ValueError('Proxy and owned response masks must be disjoint on one device')
        self.node_count=node_count
        self.proxy_mask=None if proxy_mask is None else proxy_mask.detach().clone()
        self.owned_response=None if owned_response is None else owned_response.detach().clone()
        self.phase='idle';self.restarts=0;self.entry_count=0;self.recorded_substeps_total=0

    @property
    def has_baseline(self):return self.phase!='idle'

    @property
    def complete(self):return self.phase=='complete'

    def _clear_pass(self):
        self.elapsed=0.;self.substeps=0;self.impulses={};self.batch=None;self.load_schema=None

    def begin(self,history,steps,dt):
        if not math.isfinite(dt) or dt<=0 or type(steps) is not int or steps<0:
            raise ValueError('Positive entry interval and nonnegative accepted step count required')
        if self.phase=='open':raise ValueError('An unfinished entry must restart or abort before replacement')
        saved=clone_history(history)
        self.baseline=saved;self.schema=history_schema(saved);self.baseline_steps=steps;self.dt=float(dt)
        self.entry_count+=1;self.restarts=0;self.phase='open';self._clear_pass()

    def restart(self,dt):
        if not self.has_baseline or dt!=self.dt:raise ValueError('Restart requires the original entry interval and history')
        self.restarts+=1;self.phase='open';self._clear_pass()
        return clone_history(self.baseline),self.baseline_steps

    def abort(self):
        if not self.has_baseline:raise ValueError('Cannot abort without an entry baseline')
        self.phase='aborted';self._clear_pass()
        return clone_history(self.baseline),self.baseline_steps

    def record(self,history,steps_after,dt,*,packets=None,accepted=True):
        if self.phase!='open':raise ValueError('A physical substep needs an open entry')
        if not accepted:raise ValueError('Rejected physical substeps cannot enter the transaction')
        if not math.isfinite(dt) or dt<=0:raise ValueError('Positive physical substep duration required')
        if steps_after!=self.baseline_steps+self.substeps+1:raise ValueError('Physical step counter advanced inconsistently')
        snapshot=clone_history(history)
        if history_schema(snapshot)!=self.schema:raise ValueError('Physical history schema changed within an entry')
        total=self.elapsed+dt
        tolerance=16*torch.finfo(torch.float64).eps*(self.substeps+2)*max(abs(total),self.dt)
        if total>self.dt+tolerance:raise ValueError('Substeps exceed the coupled physical interval')
        additions={};batch=self.batch;load_schema=self.load_schema
        if self.proxy_mask is not None:
            if not packets:raise ValueError('Explicit evaluated physical packets required for proxy feedback')
            current_batch=len(packets[0].wrench)
            if batch is not None and batch!=current_batch:raise ValueError('Coupled batch changed within the entry')
            batch=current_batch
            current_schema=(packets[0].wrench.dtype,packets[0].wrench.device)
            if load_schema is not None and load_schema!=current_schema:raise ValueError('Coupled load storage changed within the entry')
            if any((p.wrench.dtype,p.wrench.device)!=current_schema for p in packets):raise ValueError('Physical families use inconsistent load storage')
            load_schema=current_schema
            mapping=torch.where(self.proxy_mask,torch.arange(self.node_count,device=self.proxy_mask.device),-1)
            flags=torch.ones(batch,device=mapping.device,dtype=torch.bool)
            for packet in packets:
                if packet.family in additions:raise ValueError('Repeated physical load family')
                load=proxy_feedback((packet,),mapping,self.owned_response,self.node_count,accepted=flags,families={packet.family})
                additions[packet.family]=load*dt
        # Validation precedes mutation, so an invalid duration/schema/packet
        # cannot leave a half-recorded impulse or material counter.
        for family,value in additions.items():self.impulses[family]=self.impulses.get(family,0)+value
        self.batch=batch;self.load_schema=load_schema;self.elapsed=total;self.substeps+=1;self.recorded_substeps_total+=1
        if abs(total-self.dt)<=tolerance:self.phase='complete'

    def feedback(self,*,families=CONTACT_FAMILIES):
        if not self.complete:raise ValueError('Partial or rejected coupling entries cannot export feedback')
        if self.proxy_mask is None:raise NotImplementedError('History-only entries have no proxy ownership/readout contract')
        if not set(families)<=FAMILIES:raise ValueError('Unknown physical feedback family')
        if not self.impulses:raise ValueError('No explicit physical load observations recorded')
        reference=next(iter(self.impulses.values()))
        output=torch.zeros_like(reference)
        for family,impulse in self.impulses.items():
            if family in families:output=output+impulse/self.dt
        return output
