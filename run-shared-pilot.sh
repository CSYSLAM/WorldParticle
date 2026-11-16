#!/usr/bin/env bash
set -euo pipefail
pilot_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ "$#" -eq 0 ]; then
    set -- evaluate
fi
case "$1" in
    verify|evaluate) ;;
    *) printf 'Usage: %s [evaluate|verify]\n' "$0" >&2; exit 2 ;;
esac
exec "$pilot_root/.venv/bin/python" -B "$pilot_root/scripts/run_pilot.py" "$@"
