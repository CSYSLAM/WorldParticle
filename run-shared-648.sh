#!/usr/bin/env bash
set -euo pipefail
shared_648_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
shared_648_report="$shared_648_root/.local/manual_shared_648/$(date +%Y%m%d_%H%M%S_%N).json"
exec "$shared_648_root/.venv/bin/python" -B "$shared_648_root/scripts/run_official_live.py" \
  --checkpoint "$shared_648_root/checkpoints/shared_normal_648_v88/shared_step_000252.pt" \
  --demo basic.example_basic_joints --device cuda:0 --gpu-memory-gib 12 \
  --report "$shared_648_report" "$@"
