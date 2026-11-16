#!/usr/bin/env bash
set -euo pipefail
WP_ASSET_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec "$WP_ASSET_REPO/.venv/bin/python" -B "$WP_ASSET_REPO/scripts/run_native_asset_live.py" "$@"
