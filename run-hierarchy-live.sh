#!/usr/bin/env bash
set -euo pipefail
hierarchy_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$hierarchy_root/.venv/bin/python" -B "$hierarchy_root/scripts/run_live.py" --asset "$hierarchy_root/assets/live/mixed_attached.json" --checkpoint "$hierarchy_root/checkpoints/shared_hierarchy_v20/shared_strict.pt" "$@"
