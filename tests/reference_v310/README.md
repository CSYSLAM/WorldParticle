# Numerical reference source

These three source files preserve the pre-cache implementation used by the
static-topology comparison test. They are not imported by the deployment path.
They were captured from research source commit `c6d12b90f7aaccf335688aee5d161c4f616bb4f1`.

With separately prepared qualified Franka data and a checkpoint:

```bash
python scripts/verify_static_topology_cache.py \
  --baseline-root tests/reference_v310 --device cpu \
  --checkpoint /path/to/checkpoint.pt --report /path/to/new_result.json
```

The source snapshot excludes training data and checkpoints. CPU success does
not exercise CUDA capture or establish a CUDA speedup.
