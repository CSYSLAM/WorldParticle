# Shared VBD research snapshot: step 156

This directory freezes the model used for the recent Franka and W1 task
recordings, including the oblique flexible-bag packing video. It is a research
checkpoint with task evidence, not an all-demo, convergence or speed certificate.

- Model: `wp_vbd.mixed_solver.MixedSolver`, architecture `shared_constitutive_v21`.
- Shared width 648, 12 attention heads, 6 encoder / 4 decoder layers,
  FFN width 1296, 46,882,811 active neural parameters.
- One full checkpoint, `model.pt`, is stored as 17 binary parts on the separate
  `models/shared-vbd-step156-20261117` branch. It includes model
  weights, the exact model configuration, AdamW state, training counters and RNG
  state. Existing inference runners can load it directly; a second inference-only
  copy is not needed.
- `model_config.json` contains the architecture options. `manifest.json` binds the
  exact checkpoint bytes, SHA-256, source tag and tested environment.
- Frozen source and weights tag: `shared-vbd-step156-20261117`.

## Download and verify

GitHub does not allow this public fork to upload new Git LFS objects. The
weights are therefore split into parts of at most 32 MiB on a separate model
branch; the ordinary research branch carries the code, configuration and
manifest. The frozen tag contains source and all weight parts together.
Use a single-branch source clone to avoid downloading the model branch until
needed:

```bash
git clone --single-branch --branch research/shared-vbd-648-20261115 \
  https://github.com/CSYSLAM/WorldParticle.git
cd WorldParticle
python scripts/fetch_model_snapshot.py
python scripts/verify_model_snapshot.py
```

The standard-library downloader retrieves parts from the frozen tag, verifies
each part and reconstructs `model.pt`, checking its final SHA-256. The resulting
`.pt` file remains ignored by Git on the source branch. The verifier additionally
checks exact configuration, strict CPU model loading, finite weights and
optimizer/RNG presence. These commands do not simulate or train the model.
If the parts are already available locally, use
`python scripts/fetch_model_snapshot.py --parts-dir /path/to/parts`.

Prepare the Torch/Warp environment described in the repository README, install
the asset dependencies, and point `WP_VBD_NEWTON_ROOT` at a separate Newton
checkout of `25298aea1691e060c86f95e8d2b773c898c006d3`. Robot URDFs/meshes and
training datasets are external and are not included.

## Run a verified W1 bag configuration

Use fresh output directories and the same pinned Newton version for inference,
verification and rendering. The W1 asset defaults to
`~/下载/assets/w1-030/robot.urdf`; supply `--asset /path/to/w1-030/robot.urdf`
on another machine.

```bash
python scripts/run_w1_bag.py \
  --checkpoint model_releases/shared_vbd_step156/model.pt \
  --bag-profile flexible --bag-grip oblique --frames 50 \
  --output outputs/w1_bag_prefix
python scripts/run_w1_bag.py \
  --checkpoint model_releases/shared_vbd_step156/model.pt \
  --bag-profile flexible --bag-grip oblique \
  --soft-tangent-support-scale 512 --elastic-shape-response \
  --resume outputs/w1_bag_prefix --resume-frame 48 \
  --output outputs/w1_bag_model
python scripts/verify_w1_bag_recording.py \
  --recording outputs/w1_bag_model --report outputs/w1_bag_task.json
python scripts/verify_w1_bag_clearance.py \
  --recording outputs/w1_bag_model --report outputs/w1_bag_clearance.json
python scripts/render_w1_bag.py \
  --recording outputs/w1_bag_model --output outputs/w1_bag.mp4
```

The first resumed frame is checked against the model's own prefix before the
explicit numerical settings change. Geometry-only previews are not inference
results. The model remains slower than realtime; rendering replays recorded
states without executing dynamics. See the [W1 description](../../docs/w1_dual_grasp.md)
and [framework assessment](../../docs/framework_assessment.md).

For further training, preserve this snapshot and write to a new output location.
The checkpoint provides optimizer/RNG state; the training data and the original
training/runtime configuration must also be available. Merely loading these
weights does not establish exact training continuation or a qualified new model.
