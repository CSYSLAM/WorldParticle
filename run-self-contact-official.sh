#!/usr/bin/env bash
set -euo pipefail
self_contact_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
self_contact_report="$self_contact_root/.local/manual_self_contact/$(date +%Y%m%d_%H%M%S_%N).json"
exec "$self_contact_root/.venv/bin/python" -B "$self_contact_root/scripts/run_mixed_contact_live.py" \
  --checkpoint "$self_contact_root/checkpoints/shared_particle_self_v60/shared_strict.pt" \
  --demo cloth.example_cloth_bending --particle-self-contact --device cpu \
  --report "$self_contact_report" "$@"
