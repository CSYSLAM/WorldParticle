#!/usr/bin/env bash
set -euo pipefail
official_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$official_root/.venv/bin/python" -B "$official_root/scripts/run_official_live.py" "$@"
