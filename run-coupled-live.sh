#!/usr/bin/env bash
set -euo pipefail
coupled_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$coupled_root/.venv/bin/python" -B "$coupled_root/scripts/run_live.py" --asset "$coupled_root/assets/live/mixed_attached.json" --checkpoint "$coupled_root/checkpoints/mixed_conditioned_v4/shared.pt" "$@"
