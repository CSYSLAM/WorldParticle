#!/usr/bin/env bash
set -euo pipefail
mixed_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$mixed_root/.venv/bin/python" -B "$mixed_root/scripts/run_live.py" --asset "$mixed_root/assets/live/mixed_response.json" --checkpoint "$mixed_root/checkpoints/mixed_response_v1/shared.pt" "$@"
