"""Dirichlet-compatible material trial fields on a prescribed rod tree.

Reference harmonic weights use scalar tree elimination. Current 6x6 neutral
transport maps lift endpoint corrections. Neither operation solves dynamics.
"""
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class BoundaryPlan:
    nodes: torch.Tensor
    pins: torch.Tensor
    weights: torch.Tensor


def description(nodes, parents, pins, rest_x, device):
    """Harmonic measures of all prescribed bodies in one rooted tree.

    After eliminating descendants, an interior value is alpha*parent+beta.
    Pinned values are Kronecker boundary data. A forward pass reconstructs
    the weights, with no dense node-by-node matrix or iterative tolerance.
    """
    if len(pins) < 2:
        raise ValueError('A multi-boundary plan requires at least two pins')
    nodes = list(map(int, nodes))
    pins = list(map(int, pins))
    lookup = {node: i for i, node in enumerate(nodes)}
    roots = [node for node in nodes if parents[node] == node]
    if len(roots) != 1 or roots[0] not in pins:
        raise ValueError('The boundary tree must be rooted at a prescribed body')
    root = roots[0]
    children = {node: [] for node in nodes}
    for node in nodes:
        if node != root:
            children[parents[node]].append(node)
    order = [root]
    for node in order:
        order.extend(children[node])
    if len(order) != len(nodes):
        raise ValueError('Every boundary node must belong to the rooted tree')
    xyz = np.asarray(rest_x, dtype=np.float64)
    conductance = {node: 1/max(float(np.linalg.norm(xyz[node]-xyz[parents[node]])), 1e-12)
                   for node in nodes if node != root}
    boundary = {node: i for i, node in enumerate(pins)}
    alpha = {}
    beta = {}
    for node in reversed(order):
        value = np.zeros(len(pins), dtype=np.float64)
        if node in boundary:
            alpha[node] = 0.
            value[boundary[node]] = 1.
        else:
            coefficient = conductance[node]
            denominator = coefficient
            for child in children[node]:
                denominator += conductance[child]*(1-alpha[child])
                value += conductance[child]*beta[child]
            alpha[node] = coefficient/denominator
            value /= denominator
        beta[node] = value
    weights = np.zeros((len(nodes), len(pins)), dtype=np.float64)
    weights[lookup[root]] = beta[root]
    for node in order[1:]:
        weights[lookup[node]] = alpha[node]*weights[lookup[parents[node]]]+beta[node]
    if not np.isfinite(weights).all() or np.max(np.abs(weights.sum(-1)-1)) > 1e-9:
        raise ValueError('Unresolved rod boundary harmonic weights')
    return BoundaryPlan(torch.tensor(nodes, device=device), torch.tensor(pins, device=device),
                        torch.as_tensor(weights, device=device, dtype=torch.float64))


def compatible(plan, transport, fields):
    """Remove each pin's integrated defect using current neutral transport.

    For raw fields B and root transport A, solve A_pin*C_pin=B_pin locally.
    B'_i=B_i-A_i*sum_p(w_ip*C_pin). Boundary weights are Kronecker values,
    so every prescribed body has zero update. Harmonic differences distribute
    the added material strain along the reference tree. The physical residual
    and existing H projection still determine the trial direction's response.
    """
    solved = torch.linalg.solve_ex(transport[:, plan.pins], fields[:, plan.pins], check_errors=False)
    weighted = torch.einsum('np,bpij->bnij', plan.weights.to(fields.dtype), solved.result)
    result = fields[:, plan.nodes]-transport[:, plan.nodes]@weighted
    valid = ((solved.info == 0).all(-1) & torch.isfinite(solved.result).flatten(1).all(-1)
             & torch.isfinite(result).flatten(1).all(-1))
    return result, valid
