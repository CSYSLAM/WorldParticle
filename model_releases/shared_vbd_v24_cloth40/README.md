# Shared VBD v24 / cloth fine-tune step40

This is the exact shared checkpoint used by all six validated robot demos.
The model code lives in `src/wp_vbd/`; `model_config.json` freezes the
architecture and numerical options. The six-demo recipe is
`configs/robot_validation_cloth_ft.json`.

- Architecture: `shared_analytic_predictor_v24`, width 648, 12 heads,
  6 encoder / 4 decoder layers, FFN width 1296.
- Active shared-response and velocity-head parameters: **46,884,109**.
- Two analytic prediction rounds and one shared neural correction per physical
  step; nominally three physical steps per 60 Hz output frame. Bounded adaptive
  subdivision is recorded explicitly.
- Exact checkpoint: 188,146,967 bytes, SHA256
  `1aac9909419fbb8dda7beef4f0d533050d4ce4191849ebd54ba37dd6a85a2132`.
- The checkpoint contains model weights, configuration and training summary.
  **It does not contain optimizer/RNG state for exact training resumption.**
  Fine-tuning can initialize from it with a new optimizer and external datasets.

## Restore and verify

The checkpoint is committed as six ordinary Git binary parts, at most 32 MiB
each. After cloning this branch, reconstruct the original bytes:

```bash
python scripts/fetch_model_snapshot.py
python scripts/verify_model_snapshot.py
```

The joiner uses local committed parts first and validates every part and the
final SHA256. `model.pt` itself remains ignored, preventing a second copy in
Git. HTTP fallback uses the frozen tag `shared-vbd-v24-cloth40-20261120`.
The verification script strictly loads the model on CPU and checks finite
weights, architecture, parameter count and artifact identity.

## Run

Install the Torch/Warp and asset dependencies described in the repository
README. Use a separate Newton checkout at
`25298aea1691e060c86f95e8d2b773c898c006d3`. The recipe defaults to
`.local/newton_pinned_25298`; set its `newton_root` to your pinned checkout if
needed. External W1 and Piper assets are not redistributed.

```bash
python scripts/run_robot_validation.py --scene cloth --render
python scripts/run_robot_validation.py --scene w1 --window
python scripts/run_robot_demo_suite.py --output outputs/my_six_demos
```

The six scenes are `franka`, `w1`, `cloth`, `piper`, `w1_dual` and
`franka_rigid_cloth`. See [suite documentation](../../docs/robot_demo_suite.md)
and the committed `validation_summary.json` for actual work counts and
measured wall-clock speed. Videos and raw trajectories remain local outputs.

This is a research snapshot with six task validations. It does not establish
real-time execution, an advantage over native VBD, or success on all 42 official
VBD demos. The latest cloth fine-tune used six qualified full training drapes,
one held-out material trajectory and existing cross-type replay, for 40 updates.
