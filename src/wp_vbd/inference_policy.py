"""Execution choices that preserve the checkpoint and physical iteration budget."""
from contextlib import contextmanager

POLICIES = ('reference', 'assembly', 'fused', 'fused_search', 'fused_fast', 'fused_device', 'fused_graph')
CLI_POLICIES = ('auto', *POLICIES)
AUTO_GRAPH_SCENES = frozenset(('franka', 'cloth', 'w1'))


def resolve_policy(policy, *, scene, device, offline_reference=False):
    """Resolve CLI defaults before configuring a model or recording a run.

    Only the three CUDA scene families with complete v30 task validation
    select the graph policy automatically. Explicit choices are preserved.
    This does not change solver/training defaults or native VBD execution.
    """
    if policy != 'auto':
        if policy not in POLICIES:
            raise ValueError('Unknown inference execution policy: ' + str(policy))
        return policy
    if str(device).startswith('cuda') and scene in AUTO_GRAPH_SCENES and not offline_reference:
        return 'fused_graph'
    return 'reference'


def configure(config, policy):
    if policy not in POLICIES:
        raise ValueError('Unknown inference execution policy: ' + str(policy))
    result = dict(config)
    result.update(fast_inference_assembly=policy != 'reference',
                  compiled_contact_rows=policy in ('fused', 'fused_search', 'fused_fast', 'fused_device', 'fused_graph'),
                  compiled_kinematics=policy in ('fused', 'fused_search', 'fused_fast', 'fused_device', 'fused_graph'),
                  compiled_candidate_kernels=policy in ('fused_search', 'fused_fast', 'fused_device', 'fused_graph'),
                  compiled_response_kernels=policy in ('fused_device', 'fused_graph'),
                  response_topology_cache=policy in ('fused_device', 'fused_graph'),
                  compile_small_spatial_products=policy in ('fused_device', 'fused_graph'),
                  capture_cuda_graphs=policy == 'fused_graph',
                  fused_physical_checks=policy == 'fused_graph',
                  inference_kinematics_cache=policy == 'fused_graph',
                  inference_fill_uninitialized_memory=False if policy in ('fused_fast', 'fused_device', 'fused_graph') else None,
                  inference_deterministic_algorithms=False if policy in ('fused_fast', 'fused_device', 'fused_graph') else None)
    return result


def apply_scene(scene, config):
    """Set policy before native read/preflight as well as the neural step.

    In particular, alternating benchmark variants must not inherit the last
    variant's compiled kinematics during native mimic roundoff projection.
    """
    import torch
    enabled = not torch.is_grad_enabled()
    for key in ('fast_inference_assembly', 'compiled_contact_rows',
                'compiled_kinematics', 'compiled_candidate_kernels', 'compiled_response_kernels', 'response_topology_cache',
                'fused_physical_checks'):
        setattr(scene, key, enabled and bool(config.get(key, False)))


@contextmanager
def inference_settings(config):
    """Keep compiler and arithmetic choices inside the physical call."""
    import torch
    from .compiled_contact import small_spatial_products
    enabled=bool(config.get('compile_small_spatial_products',False)) and not torch.is_grad_enabled()
    from .cuda_graph import capture
    graph_enabled=bool(config.get('capture_cuda_graphs',False)) and not torch.is_grad_enabled()
    with capture(graph_enabled),small_spatial_products(enabled),_arithmetic_settings(config):
        yield


@contextmanager
def _arithmetic_settings(config):
    """Scope optional parallel reductions/allocation diagnostics to inference.

    Precision, validity checks, initialization and gradient-enabled training
    are retained. Fast mode allows FP64 atomic sums with roundoff-order
    variation. Settings are process-global; CLI owns a dedicated process.
    """
    import torch
    requested=config.get('inference_fill_uninitialized_memory')
    deterministic=config.get('inference_deterministic_algorithms')
    if torch.is_grad_enabled() or requested is None and deterministic is None:
        yield
        return
    previous=torch.utils.deterministic.fill_uninitialized_memory
    previous_deterministic=torch.are_deterministic_algorithms_enabled()
    previous_warn_only=torch.is_deterministic_algorithms_warn_only_enabled()
    if (requested is None or previous==bool(requested)) and (deterministic is None or previous_deterministic==bool(deterministic)):
        yield
        return
    try:
        if deterministic is not None:torch.use_deterministic_algorithms(bool(deterministic))
        if requested is not None:torch.utils.deterministic.fill_uninitialized_memory=bool(requested)
        yield
    finally:
        torch.utils.deterministic.fill_uninitialized_memory=previous
        torch.use_deterministic_algorithms(previous_deterministic,warn_only=previous_warn_only)
