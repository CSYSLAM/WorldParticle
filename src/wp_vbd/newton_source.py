"""Select a Newton checkout without changing the user's working repository."""
import os
from pathlib import Path
import subprocess
import sys


def source_root():
    return Path(os.environ.get('WP_VBD_NEWTON_ROOT', '/home/oem/code/engine/newton')).expanduser().resolve()


def activate_newton_source():
    root = source_root()
    if not (root / 'newton' / '__init__.py').is_file():
        raise ValueError(f'Newton source checkout required: {root}')
    loaded = sys.modules.get('newton')
    if loaded is not None:
        actual = Path(loaded.__file__).resolve().parents[1]
        if actual != root:
            raise RuntimeError(f'Newton already imported from {actual}; configured checkout is {root}')
    location = str(root)
    if location in sys.path:
        sys.path.remove(location)
    sys.path.insert(0, location)
    return root


def newton_revision():
    root = activate_newton_source()
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()


def source_record():
    return {'root': str(source_root()), 'revision': newton_revision()}
