"""Admit VF/EE latent rows without moving unrelated type embeddings."""
import torch


HEAD = 'protected_response.coarse_response.'
EMBEDDING = 'protected_response.hierarchy.relation_kind.weight'
SELF_KINDS = (9, 10)


def configure_parameters(model, *, typed_self=False):
    """Return optimizer groups and the number of independently active scalars.

    The 16x32 embedding is one PyTorch Parameter. Its hook admits only the
    two existing self-contact rows; zero decay and fresh optimizer moments
    keep all fourteen other rows bitexact, including during replay batches.
    """
    head, embeddings = [], []
    for name, parameter in model.named_parameters():
        enabled = name.startswith(HEAD) or (typed_self and name == EMBEDDING)
        parameter.requires_grad_(enabled)
        if name.startswith(HEAD):
            head.append(parameter)
        elif typed_self and name == EMBEDDING:
            if parameter.shape != (16, 32):
                raise ValueError('The registered shared relation schema changed')
            mask = torch.zeros_like(parameter)
            mask[list(SELF_KINDS)] = 1
            parameter.register_hook(lambda gradient, mask=mask: gradient*mask)
            embeddings.append(parameter)
    if sum(parameter.numel() for parameter in head) != 33 or len(embeddings) != int(typed_self):
        raise ValueError('Expected the shared head and optional shared relation embedding')
    groups = [{'params': head, 'weight_decay': 1e-4}]
    if embeddings:
        # Typed latent gradients have a much smaller scale than the coarse
        # scalar response. Keep their Adam denominator above 1e-12 rather
        # than allowing the head's 1e-8 epsilon to erase FP32 updates.
        groups.append({'params': embeddings, 'weight_decay': 0., 'eps': 1e-12})
    return groups, 33+64*int(typed_self)


def assert_frozen(model, before, *, typed_self=False):
    current = model.state_dict()
    for name, value in before.items():
        if name.startswith(HEAD):
            continue
        if typed_self and name == EMBEDDING:
            keep = torch.ones(value.shape[0], device=value.device, dtype=torch.bool)
            keep[list(SELF_KINDS)] = False
            if not torch.equal(value[keep], current[name][keep]):
                raise AssertionError('An unrelated relation embedding row moved')
        elif not torch.equal(value, current[name]):
            raise AssertionError('A frozen model tensor moved: '+name)
