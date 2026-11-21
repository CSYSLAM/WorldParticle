"""Ensure fast inference restores process settings and preserves training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    import torch
    from wp_vbd.inference_policy import configure,inference_settings,apply_scene
    from types import SimpleNamespace
    original=(torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),
              torch.utils.deterministic.fill_uninitialized_memory)
    config=configure({},'fused_fast')
    precision=(torch.get_default_dtype(),torch.get_float32_matmul_precision(),
               torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
    def flags():return (torch.are_deterministic_algorithms_enabled(),torch.is_deterministic_algorithms_warn_only_enabled(),torch.utils.deterministic.fill_uninitialized_memory)
    try:
        torch.use_deterministic_algorithms(True,warn_only=True)
        torch.utils.deterministic.fill_uninitialized_memory=True
        expected=flags()
        with torch.no_grad(),inference_settings(config):
            assert flags()==(False,False,False)
            with inference_settings(config):assert flags()==(False,False,False)
            assert flags()==(False,False,False)
        assert flags()==expected
        try:
            with torch.no_grad(),inference_settings(config):raise RuntimeError('expected probe')
        except RuntimeError:pass
        assert flags()==expected
        with torch.enable_grad(),inference_settings(config):
            assert flags()==expected
            value=torch.ones(2,requires_grad=True);value.square().sum().backward()
            assert torch.equal(value.grad,torch.full_like(value,2.))
        with torch.no_grad(),inference_settings(configure({},'reference')):assert flags()==expected
        assert precision==(torch.get_default_dtype(),torch.get_float32_matmul_precision(),
                           torch.backends.cuda.matmul.allow_tf32,torch.backends.cudnn.allow_tf32)
        from wp_vbd import compiled_contact
        limit=torch._dynamo.config.cache_size_limit
        spatial=configure({},'fused_device')
        with torch.no_grad(),inference_settings(spatial):
            assert compiled_contact._small_products.get() and torch._dynamo.config.cache_size_limit>=32
            with inference_settings(spatial):assert compiled_contact._small_products.get()
            with inference_settings(config):assert not compiled_contact._small_products.get()
            assert compiled_contact._small_products.get()
        assert not compiled_contact._small_products.get() and torch._dynamo.config.cache_size_limit==limit
        try:
            with torch.no_grad(),inference_settings(spatial):raise RuntimeError('expected spatial probe')
        except RuntimeError:pass
        assert not compiled_contact._small_products.get() and torch._dynamo.config.cache_size_limit==limit
        with torch.enable_grad(),inference_settings(spatial):
            assert not compiled_contact._small_products.get() and torch._dynamo.config.cache_size_limit==limit
        from wp_vbd import cuda_graph
        graph_config=configure({},'fused_graph')
        assert graph_config['capture_cuda_graphs'] and graph_config['compile_small_spatial_products']
        assert graph_config['inference_kinematics_cache']
        assert not configure(graph_config,'reference')['inference_kinematics_cache']
        scene=SimpleNamespace()
        with torch.no_grad():
            apply_scene(scene,graph_config)
            assert scene.fused_physical_checks
            apply_scene(scene,configure({},'reference'))
            assert not scene.fused_physical_checks
            apply_scene(scene,graph_config)
        with torch.enable_grad():
            apply_scene(scene,graph_config)
            assert not scene.fused_physical_checks
        assert not spatial['capture_cuda_graphs']
        with torch.no_grad(),inference_settings(graph_config):
            assert cuda_graph.enabled() and compiled_contact._small_products.get()
            with inference_settings(config):assert not cuda_graph.enabled()
            assert cuda_graph.enabled()
        assert not cuda_graph.enabled() and flags()==expected
        try:
            with torch.no_grad(),inference_settings(graph_config):raise RuntimeError('expected graph probe')
        except RuntimeError:pass
        assert not cuda_graph.enabled()
        with torch.enable_grad(),inference_settings(graph_config):assert not cuda_graph.enabled()
        result=dict(status='PASS',nested_inference_settings_restored=True,
            exception_restores_settings=True,gradient_enabled_training_unchanged=True,
            reference_settings_unchanged=True,precision_controls_untouched=True,
            compiler_mode_and_specialization_budget_scoped_and_restored=True,
            graph_scope_nested_exception_and_training_restored=True,
            fused_checks_follow_policy_and_disable_in_training=True)
        a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
    finally:
        torch.use_deterministic_algorithms(original[0],warn_only=original[1])
        torch.utils.deterministic.fill_uninitialized_memory=original[2]


if __name__=='__main__':main()
