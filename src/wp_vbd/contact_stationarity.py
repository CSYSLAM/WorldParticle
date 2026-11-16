"""Exact incident-row Boolean reduction without per-contact Python loops."""
import torch


def incident_stationary(owners, change, internal, threshold):
    """A node is stationary only if every incident row passes the threshold.

    Integer counts preserve repeated owners, missing ports and NaN comparison
    semantics. This changes no floating-point sum, solver budget or response.
    """
    failed=~(change<=threshold)
    values=(failed[...,None]&(owners>=0)[None]).long()
    counts=torch.zeros_like(internal,dtype=torch.long).index_add(
        1,owners.clamp_min(0).flatten(),values.flatten(1,2))
    return internal&(counts==0)
