"""Declared learning-rate changes while preserving a selected Adam state."""
import math


def learning_rate_migration(parent_rate, target_rate, reason):
    for rate in (parent_rate, target_rate):
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or rate <= 0:
            raise ValueError('Finite positive learning rates required')
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError('An explicit continuation reason is required')
    return {'parent_learning_rate': parent_rate, 'learning_rate': target_rate, 'reason': reason}


def apply_learning_rate_migration(optimizer, target_rate, declaration=None):
    """Validate all groups before changing only their lr; moments stay intact."""
    learning_rate_migration(target_rate, target_rate, 'validation')
    groups = optimizer.param_groups
    if not groups or any(group['eps'] != 1e-12 for group in groups):
        raise ValueError('The selected Adam epsilon must remain unchanged')
    if declaration is None:
        if any(group['lr'] != target_rate for group in groups):
            raise ValueError('A learning-rate change needs a manifest declaration')
        return None
    if not isinstance(declaration, dict) or set(declaration) != {'parent_learning_rate', 'learning_rate', 'reason'}:
        raise ValueError('Unsupported optimizer continuation declaration')
    checked = learning_rate_migration(declaration['parent_learning_rate'],
                                      declaration['learning_rate'], declaration['reason'])
    if checked['learning_rate'] != target_rate or any(group['lr'] != checked['parent_learning_rate'] for group in groups):
        raise ValueError('Declared source/target learning rates do not match the restored optimizer and CLI')
    for group in groups:
        group['lr'] = target_rate
    return checked
