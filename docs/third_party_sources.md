# External source attribution

The repository's root license covers the original project. File-level license
identifiers are retained for adapted external material.

`src/wp_vbd/w1_bag_scene.py` is marked Apache-2.0. Its reference bag dimensions
and gusset/crease formulas follow Newton's
`assets/w1_paper_bag/build_blender.py`, copyright (c) 2026 The Newton Developers.
The research scene constructs a coarser open shell without folded handles,
uses its own motor-target schedule and shared-network runtime, and does not
import or execute the reference Blender generator or coupled robot demo.
The Apache 2.0 license text is included in
[LICENSES/Apache-2.0.txt](../LICENSES/Apache-2.0.txt).

Newton is a separately installed dependency. The external W1 robot URDF and
meshes, Newton source checkout and generated trajectories are not included.
The explicitly published step 156 research checkpoint is stored in parts on the separate
`models/shared-vbd-step156-20261117` branch; its configuration and downloader
are in `model_releases/shared_vbd_step156/`.
