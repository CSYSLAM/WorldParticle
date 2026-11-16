"""Generate true rod material splits and retain every previous replay case."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import shutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from prepare_rod_history import DEMOS

FRAMES = {'vbd.example_cable_bend_stiffness':480, 'vbd.example_cable_dahl_hysteresis':840}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--generate', action='store_true')
    args = parser.parse_args()
    import numpy as np
    directory = ROOT / '.local/rod_history_v9'
    report_dir = ROOT / 'reports/rod_history_v9'
    report_dir.mkdir(exist_ok=True)
    old = json.loads((ROOT / 'reports/official_demos/transport/data_manifest.json').read_text())
    rows = old['scenes'].copy()
    for identifier in DEMOS:
        for name, factor in (('train', 1.), ('train_extra', 1.25), ('validation', .8), ('test', 1.1)):
            path = directory / (identifier + '_' + name + '.npz')
            frames = FRAMES.get(identifier,120)
            if path.exists():
                with np.load(path,allow_pickle=False) as data:existing_frames=json.loads(str(data['metadata']))['teacher_frames']
                if existing_frames < frames:
                    archive=directory/'prior_short';archive.mkdir(exist_ok=True)
                    shutil.move(str(path),str(archive/(path.stem+f'_{existing_frames}frames.npz')))
            if not path.exists() and args.generate:
                command = ['uv', 'run', '--no-sync', '--offline', '--project', '/home/oem/code/engine/newton', 'python', str(ROOT / 'scripts/generate_rod_history.py'), '--demo', identifier, '--factor', str(factor), '--frames', str(frames), '--output', str(path)]
                with (report_dir / (identifier + '_' + name + '.log')).open('w') as log:
                    subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
            if not path.exists():raise FileNotFoundError(path)
            with np.load(path, allow_pickle=False) as data:
                metadata = json.loads(str(data['metadata']))
                assert metadata['body_position_representation'] == 'origin'
                assert all(np.isfinite(data[key]).all() for key in ('x', 'q', 'v', 'omega', 'history__dual', 'history__stress', 'input_force', 'input_torque'))
                factor_actual = metadata.get('parameter_factor', 1.)
                assert factor_actual == factor
                count = len(data['x']) - 1
            row = {'index': len(rows), 'dataset_id': identifier + '/' + name, 'demo': identifier, 'path': str(path.relative_to(ROOT)), 'split': 'train' if name.startswith('train') else name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'parameter_factor': factor, 'substeps': count, 'dt': metadata['dt'], 'status': 'PASS', 'history_contract': 'rod_compliant_alm_v1'}
            rows.append(row); print(row['dataset_id'], 'ready', flush=True)
    manifest = {'status': 'PASS', 'newton_sha': old['newton_sha'], 'scenes': rows, 'test_isolation': 'Complete material episodes skipped before training load', 'scope': 'Six no-contact authored rod demos plus all previous official and mixed replay episodes; other 41 candidates remain operator pending'}
    (report_dir / 'data_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    config = json.loads((ROOT / 'configs/shared_official_transport_v8.json').read_text())
    config.update(name='shared_history_v9', report_dir='reports/rod_history_v9', checkpoint_dir='checkpoints/shared_history_v9', initialize_from=str(ROOT / 'checkpoints/shared_official_transport_v8/shared_strict.pt'))
    config['model'].update(joint_history=True, rod_unroll=4)
    config['training'].update(steps=320, learning_rate=3e-5, validation_interval=160,validation_steps=120, max_seconds=7200,ensure_initial_coverage=True,official_sampling_weight=4,trainable_prefixes=['response.entity_encoders.3.'])
    config['scope'].update(default_alm_semantics='Finite-material rod compliant ALM only; persistent numeric duals and Dahl history are explicit state', official_coverage='10 admitted demos, 41 still require contacts/drives/coupling operators', state_positions='COM for legacy scenes; authored origin poses for history rod scenes',training_curriculum='Calibrate the existing rod input encoder first; freeze all previously shared trunk and response weights, one checkpoint for every type')
    (ROOT / 'configs/shared_history_v9.json').write_text(json.dumps(config, indent=2) + '\n')


if __name__ == '__main__':main()
