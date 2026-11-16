#!/usr/bin/env bash
set -euo pipefail
live_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$live_root/.venv/bin/python" -B "$live_root/scripts/run_live.py" "$@"
