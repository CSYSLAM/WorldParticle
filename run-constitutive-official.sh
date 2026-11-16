#!/usr/bin/env bash
set -euo pipefail
constitutive_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$constitutive_root/.venv/bin/python" -B "$constitutive_root/scripts/run_official_live.py" --checkpoint "$constitutive_root/checkpoints/shared_joint_batch_v31/shared_strict.pt" --demo vbd.example_vbd_rigid_rigid_contact "$@"
