#!/usr/bin/env bash
set -euo pipefail
coarse_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$coarse_root/.venv/bin/python" -B "$coarse_root/scripts/run_live.py" --asset "$coarse_root/assets/live/mixed_response.json" --checkpoint "$coarse_root/checkpoints/mixed_coarse_additive_v2/shared.pt" "$@"
