#!/usr/bin/env bash
set -euo pipefail
vector_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$vector_root/.venv/bin/python" -B "$vector_root/scripts/run_live.py" --asset "$vector_root/assets/live/mixed_attached.json" --checkpoint "$vector_root/checkpoints/shared_vector_v19/shared_strict.pt" "$@"
