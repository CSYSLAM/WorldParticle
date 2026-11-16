"""Validate declared standalone or complete fixed-inventory VBD scopes."""
import json
import math
from pathlib import Path

CUSTOM_PURE_VBD={'research.native_five_families':('elastic_rod','elastic_rod_surface_joint_self_contact_v222'),
                 'research.franka_vbd_block':('surface','elastic_surface_joint_contact_v38')}


def changed_weight_coverage(samples,scope,*,selected_step=None):
    """Separate actual changed-weight official and authored-data coverage."""
    if selected_step is not None and (type(selected_step) is not int or selected_step<0):
        raise ValueError('Nonnegative selected optimizer prefix required')
    allowed=set(scope['target_demos']);official=set();custom=set();datasets=set()
    for sample in samples:
        if selected_step is not None and sample['update']>selected_step:continue
        demo=sample.get('demo')
        if demo is not None and demo not in allowed and demo not in CUSTOM_PURE_VBD:
            raise ValueError('Unknown or excluded optimizer demo: '+str(demo))
        norm=sample['gradient_norm_before_clip']
        if type(norm) not in (int,float) or not math.isfinite(norm) or norm<0:
            raise ValueError('Finite nonnegative actual gradient norm required')
        if type(sample['actual_max_gradient_scalar_changed']) is not bool:
            raise ValueError('Actual changed-weight boolean required')
        if norm<=0 or not sample['actual_max_gradient_scalar_changed']:continue
        if demo in allowed:official.add(demo)
        elif demo in CUSTOM_PURE_VBD:custom.add(demo);datasets.add(sample['dataset'])
    return {'official_demos':sorted(official),'custom_pure_vbd_demos':sorted(custom),
            'custom_pure_vbd_datasets':sorted(datasets)}

def load_scope(root,*,relative_path='configs/pure_vbd_demo_scope.json'):
    # Historical runs retain their original42-demo scope. New full-scope
    # courses select all_vbd_demo_scope.json explicitly, without rewriting it.
    path=Path(root)/relative_path
    value=json.loads(path.read_text())
    if value['scope'] not in ('pure_vbd_only','all_vbd_related'):raise ValueError('An explicit fixed VBD training scope is required')
    if set(value['target_demos'])&set(value['excluded_demos']):raise ValueError('Overlapping demo scope')
    if value['scope']=='all_vbd_related':
        inventory=json.loads((Path(root)/'configs/vbd_demo_inventory.json').read_text())
        if (value['newton_sha']!=inventory['newton_sha'] or len(value['target_demos'])!=51
            or len(set(value['target_demos']))!=51 or value['excluded_demos']
            or set(value['target_demos'])!={row['id'] for row in inventory['examples']}):
            raise ValueError('The complete pinned51-demo inventory must remain intact')
    return value


def validate_rows(rows,scope):
    allowed=set(scope['target_demos']);demos=set();replay=0;custom=set()
    for row in rows:
        demo=row.get('demo')
        if demo in CUSTOM_PURE_VBD:
            loader,contract=CUSTOM_PURE_VBD[demo]
            if row.get('loader')!=loader or row.get('custom_vbd_contract')!=contract:
                raise ValueError('Custom pure-VBD data requires its explicit native loader and contract')
            custom.add(demo);continue
        if demo is None:
            if row.get('loader')!='legacy' or not row['dataset_id'].startswith(('replay_coupled/','replay_legacy_plane/')):
                raise ValueError('Unnamed data cannot bypass the pure-VBD scope')
            replay+=1;continue
        if demo not in allowed:raise ValueError('External or unknown demo excluded by current user scope: '+demo)
        demos.add(demo)
    result={'scope':scope['scope'],'target_official_demo_count':len(allowed),
            'excluded_external_demo_count':len(scope['excluded_demos']),
            'scheduled_official_demos':sorted(demos),'scheduled_official_demo_count':len(demos),
            'same_backend_legacy_replay_datasets':replay,
            'policy':('One shared learned replacement for VBD-owned dynamics, with explicit state/control/history/reaction exchange and preserved external owners.' if scope['scope']=='all_vbd_related' else
                'Only standalone VBD dynamics; optional other-backend comparison branches are not executed. Same-model rigid/soft/cloth/rod interactions remain in scope.')}
    if custom:result.update(scheduled_custom_pure_vbd_demos=sorted(custom),scheduled_custom_pure_vbd_demo_count=len(custom))
    return result
