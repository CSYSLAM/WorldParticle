#!/usr/bin/env bash
set -euo pipefail
rod_live_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$rod_live_root/.venv/bin/python" -B "$rod_live_root/scripts/run_rod_contact_live.py" \
  --checkpoint "$rod_live_root/checkpoints/shared_rod_contact_v41/shared_strict.pt" \
  --force-shaped-material-ports --device cpu \
  --report "$rod_live_root/.local/manual/rod-contact-$(date +%Y%m%d-%H%M%S)-$$.json" "$@"
