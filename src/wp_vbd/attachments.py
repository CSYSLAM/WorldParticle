"""Finite elastic particle--body attachment, expressed in COM coordinates.

No solver, persistent multipliers, or material damping is hidden here. A body
anchor is one point of its pose, never an independent particle DOF.
"""

from dataclasses import dataclass

import torch

from . import math3d as so3
from .response_operator import skew


@dataclass
class AttachmentEvaluation:
    gap: torch.Tensor
    jacobian: torch.Tensor
    residual: torch.Tensor
    diagonal: torch.Tensor
    energy: torch.Tensor


def particle_body_attachment(particle, center, orientation, local_anchor, stiffness):
    """Evaluate A attachments; inputs use [B,A,*], local_anchor is COM-local.

    Endpoint order is [particle, body], with generalized world translation and
    rotation columns. The particle's rotation columns are exactly zero. The
    returned 6x6 self blocks are GN; the full coupled action is J^T k J and
    includes off-diagonal endpoint blocks through the retained Jacobian.
    """
    arm=so3.rotate(orientation,local_anchor)
    gap=center+arm-particle
    eye=torch.eye(3,device=particle.device,dtype=particle.dtype).expand(*gap.shape[:-1],3,3)
    zero=torch.zeros_like(eye)
    left=torch.cat((-eye,zero),-1)
    right=torch.cat((eye,-skew(arm)),-1)
    jacobian=torch.stack((left,right),-2)
    traction=gap*stiffness[...,None]
    residual=torch.einsum('...dsi,...d->...si',jacobian,traction)
    diagonal=torch.einsum('...dsi,...dsj->...sij',jacobian,jacobian)*stiffness[...,None,None,None]
    energy=.5*stiffness*gap.square().sum(-1)
    return AttachmentEvaluation(gap,jacobian,residual,diagonal,energy)
