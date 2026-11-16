"""Torch-native XYZW quaternions and small-block geometry."""

import torch


def eigh_zero_safe(value):
    """Exact decomposition of zero blocks without a zero-scale GPU solve.

    Some CUDA library paths return NaN eigenvalues for an exactly zero 3x3
    block after other small solves. Decompose an identity placeholder, then
    restore the mathematically exact zero spectrum and canonical basis.
    This adds no physical ridge or nonzero metric to an inactive contact.
    """
    zero=(value==0).all(-1).all(-1)
    eye=torch.eye(value.shape[-1],device=value.device,dtype=value.dtype)
    eigen,frame=torch.linalg.eigh(torch.where(zero[...,None,None],eye,value))
    return torch.where(zero[...,None],0.,eigen),torch.where(zero[...,None,None],eye,frame)


def normalize(x):
    return x / x.square().sum(-1, keepdim=True).clamp_min(1e-20).sqrt()


def conjugate(q):
    return torch.cat((-q[..., :3], q[..., 3:]), -1)


def multiply(a, b):
    av, bv = torch.broadcast_tensors(a[..., :3], b[..., :3])
    return torch.cat((a[..., 3:]*bv + b[..., 3:]*av + torch.linalg.cross(av, bv), a[..., 3:]*b[..., 3:]-(av*bv).sum(-1, keepdim=True)), -1)


def rotate(q, x):
    v, x = torch.broadcast_tensors(q[..., :3], x)
    return x + 2*torch.linalg.cross(v, torch.linalg.cross(v, x)+q[..., 3:]*x)


def exp(rotation):
    angle = rotation.square().sum(-1, keepdim=True).clamp_min(1e-20).sqrt()
    factor = 0.5*torch.sinc(angle/(2*torch.pi))
    return normalize(torch.cat((factor*rotation, torch.cos(0.5*angle)), -1))


def log(q):
    q = normalize(q)
    q = torch.where(q[..., 3:] < 0, -q, q)
    norm = q[..., :3].square().sum(-1, keepdim=True).clamp_min(1e-20).sqrt()
    return q[..., :3] * (2*torch.atan2(norm, q[..., 3:].clamp_min(1e-12))/norm)


def matrix(q):
    basis = torch.eye(3, device=q.device, dtype=q.dtype)
    return rotate(q[..., None, :], basis).transpose(-1, -2)


def cofactor(f):
    return torch.stack((torch.linalg.cross(f[..., :, 1], f[..., :, 2]), torch.linalg.cross(f[..., :, 2], f[..., :, 0]), torch.linalg.cross(f[..., :, 0], f[..., :, 1])), -1)
