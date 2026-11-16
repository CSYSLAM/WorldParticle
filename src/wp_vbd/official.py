"""Load original Newton examples without changing their authored physics."""

import importlib
import inspect
from contextlib import nullcontext
from pathlib import Path
import sys

NEWTON_ROOT=Path('/home/oem/code/engine/newton')


def solver_configuration_record(configuration):
    """JSON description used by source tapes, never solver constructor kwargs.

    Legacy tapes store enum-keyed collision schedules as their exact repr.
    Compare that same description when loading; keep the live typed mapping
    intact for the original solver and for physical asset export.
    """
    return {key:value if isinstance(value,(str,float,int,bool,type(None))) else str(value)
            for key,value in configuration.items()}


def make_example(identifier, *, arguments=(), reference=True, iterations=None,viewer=None,allow_capture=False,preserve_defaults=False,host_scalar_compat=False,contact_capacity=None):
    capacity=dict(contact_capacity or {})
    allowed={'particle_vertex_contact_buffer_size','particle_edge_contact_buffer_size'}
    if set(capacity)-allowed or any(type(value) is not int or value<=0 for value in capacity.values()):
        raise ValueError('Only explicit positive particle contact buffer capacities may be overridden')
    sys.path.insert(0,str(NEWTON_ROOT))
    import newton
    import newton.examples
    import warp as wp
    from newton._src.solvers.coupled.interface import CouplingInterface
    original=newton.solvers.SolverVBD
    configurations=[]

    class InferenceOnly(newton.solvers.SolverBase,CouplingInterface):
        register_custom_attributes=original.register_custom_attributes
        SUPPORTED_OBSERVABLES=frozenset({newton.solvers.SolverBase.ObservableKind.CONTACT_F})
        def __init__(self,model,*args,**kwargs):
            super().__init__(model);self.iterations=kwargs.get('iterations',10)
            self.contacts=None
        def step(self,*args,**kwargs):raise RuntimeError('Full VBD is forbidden in inference construction')

    class TrackedSolver(original if reference else InferenceOnly):
        _worldparticle_vbd_original=original
        def __init__(self,*args,**kwargs):
            kwargs.update(capacity)
            bound=inspect.signature(original).bind(*args,**kwargs);bound.apply_defaults()
            values={key:value for key,value in bound.arguments.items() if key not in ('model','collision_pipeline')}
            if iterations is not None:kwargs['iterations']=iterations
            if reference:kwargs['deterministic']=wp.DeterministicMode.RUN_TO_RUN
            super().__init__(*args,**kwargs)
            configurations.append((self,values))
    newton.solvers.SolverVBD=TrackedSolver
    replaced=[]
    solver_bindings=[]
    try:
        module=importlib.import_module('newton.examples.'+identifier)
        # Some examples import the class directly. Cached example modules
        # must bind to this invocation's tracker, never a previous closure.
        for name,value in list(vars(module).items()):
            if value is original or (isinstance(value,type) and vars(value).get('_worldparticle_vbd_original') is original):
                solver_bindings.append(name);setattr(module,name,TrackedSolver)
        cls=module.Example
        parser=cls.create_parser() if hasattr(cls,'create_parser') else newton.examples.create_parser()
        args=parser.parse_args(list(arguments))
        for action in parser._actions:
            if action.dest in ('solver','solver_type') and action.choices and 'vbd' in action.choices:setattr(args,action.dest,'vbd')
            if action.dest=='payload_kind' and action.choices and 'vbd-cable' in action.choices:args.payload_kind='vbd-cable'
        if not hasattr(args,'solver'):args.solver='vbd'
        if not preserve_defaults:args.seed=20261030;args.num_frames=120
        args.viewer='null';args.test=False
        if viewer is None:viewer=newton.viewer.ViewerNull(num_frames=args.num_frames or 120)
        # Capturing would run physics during the construction audit. The
        # original step/simulate/control methods remain available afterwards.
        def no_capture(self,*unused,**kwargs):self.graph=None
        for name in dir(cls):
            if not allow_capture and 'capture' in name and callable(getattr(cls,name)):
                replaced.append((name,getattr(cls,name)));setattr(cls,name,no_capture)
        context=nullcontext()
        if host_scalar_compat:
            from .host_scalar_compat import warp_host_scalars
            context=warp_host_scalars(module,wp)
        with context:example=cls(viewer,args)
        return example,configurations,vars(args)
    finally:
        newton.solvers.SolverVBD=original
        for name in solver_bindings:setattr(module,name,original)
        for name,value in replaced:setattr(cls,name,value)
