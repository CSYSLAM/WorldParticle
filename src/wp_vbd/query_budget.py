"""Choose a complete immutable query envelope before each physical step."""
import math

import torch

from . import math3d as so3
from .mixed_physics import predictor


class QueryBudgetBank:
    """Inference-only C222 query levels; h, K and material histories stay fixed."""
    def __init__(self,runtime,maximum,*,observe_memory=None,headroom=1.05):
        scene=runtime.scene
        if scene.metadata.get('contract')!='elastic_rod_surface_joint_self_contact_v222' or not scene.metadata.get('native_asset_provenance'):
            raise ValueError('Adaptive query envelopes require the authored C222 native runtime')
        if runtime.network.training or runtime._coupling_active or scene.metadata.get('cross_step_contact_history') or scene.metadata.get('prescribed_pose_history'):
            raise ValueError('Standalone inference without external restart or matched-contact caches required')
        base=runtime.oracle.motion_budget
        if not math.isfinite(maximum) or maximum<base or not math.isfinite(headroom) or headroom<1:
            raise ValueError('Finite maximum>=base and headroom>=1 required')
        levels=[base]
        while levels[-1]<maximum:levels.append(min(maximum,2*levels[-1]))
        self.runtime=runtime;self.levels=tuple(levels);self.oracles={base:runtime.oracle};self.headroom=headroom
        self.observe_memory=observe_memory;self.total_queries=runtime.oracle.queries;self.trace=[];self.pending=None

    def prepare(self,start,force,torque,dt):
        runtime=self.runtime;scene=runtime.scene
        if torch.is_grad_enabled() or runtime.network.training or runtime._coupling_active:
            raise RuntimeError('Adaptive query envelopes may only run in standalone inference')
        if self.pending is not None:raise RuntimeError('The previous physical-step query envelope is still frozen')
        scene.use_exact_inertia=True
        tx,tq=predictor(scene,start,dt,force,torque)
        angle=so3.log(so3.multiply(tq,so3.conjugate(start.q))).norm(dim=-1)
        travel=(tx-start.x).norm(dim=-1)+scene.body_radius*angle
        reference=runtime.oracle.material_reference(start);count=scene.metadata['particle_count']
        particle=(tx[0,:count]-reference).norm(dim=-1)
        required=float(torch.maximum(travel.max(),particle.max()))
        entry={'physical_steps_before':runtime.steps,'required_forecast_surface_travel_m':required if math.isfinite(required) else None,
            'selected_motion_budget_m':None,'native_queries_before':self.total_queries,'snapshot_ready':False}
        self.pending=entry
        if not math.isfinite(required) or required>self.levels[-1]:
            raise RuntimeError(f'Forecast surface travel {required:.9g}m exceeds configured query maximum {self.levels[-1]:.9g}m')
        request=min(self.levels[-1],max(self.levels[0],self.headroom*required))
        level=next(value for value in self.levels if value>=request)
        if level not in self.oracles:
            previous=scene.metadata['self_contact_motion_budget'];scene.metadata['self_contact_motion_budget']=level
            try:oracle=type(runtime.oracle)(runtime.source_model,scene,runtime.wp,runtime.newton)
            finally:scene.metadata['self_contact_motion_budget']=previous
            self.oracles[level]=oracle
            if self.observe_memory is not None:self.observe_memory()
        runtime.oracle=self.oracles[level];runtime.oracle.queries=self.total_queries
        scene.metadata['self_contact_motion_budget']=level
        entry['selected_motion_budget_m']=level;self.active_oracle=runtime.oracle

    def queried(self,successful):
        oracle=self.runtime.oracle;self.total_queries=oracle.queries
        self.pending.update(native_queries_after=self.total_queries,snapshot_ready=successful)
        if successful:
            self.check_frozen()
            self.pending.update(self_query_radius_m=oracle.self_query.query_radius,
                actual_self_candidate_rows=len(oracle.self_snapshot.contacts.kind))

    def check_frozen(self):
        level=self.pending['selected_motion_budget_m'];oracle=self.runtime.oracle
        if oracle is not self.active_oracle or oracle.motion_budget!=level or oracle.self_query.motion_budget!=level:
            raise RuntimeError('Query envelope changed inside a physical step')
        snapshot=oracle.self_snapshot
        if snapshot.motion_budget!=level or snapshot.query_radius!=oracle.self_query.query_radius:
            raise RuntimeError('The complete self snapshot has a different motion envelope')
        if self.runtime.scene.metadata['self_contact_motion_budget']!=level:
            raise RuntimeError('Physical-step metadata changed its query envelope')

    def finish_attempt(self):
        if self.pending is None:return
        self.pending.update(physical_steps_after=self.runtime.steps,
            physical_step_accepted=self.runtime.steps==self.pending['physical_steps_before']+1)
        self.trace.append(self.pending);self.pending=None

    def report(self):
        return {'enabled':True,'base_motion_budget_m':self.levels[0],'maximum_motion_budget_m':self.levels[-1],
            'available_levels_m':list(self.levels),'constructed_levels_m':sorted(self.oracles),'forecast_headroom':self.headroom,
            'actual_native_geometry_queries':self.total_queries,'physical_step_attempts':self.trace,
            'scope':'One fresh complete query at a selected immutable envelope per physical step. Each level owns expanded rigid/soft gaps and a full VF/EE query. No h/K/material or acceptance-threshold changes, internal budget resets, solver retries or VBD fallback. Unsupported external/matched-history coupling and training reject.'}
