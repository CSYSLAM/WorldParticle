"""Bounded endpoint-velocity correction for variable-step dynamics."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from .data import State


class TerminalVelocity(nn.Module):
    def __init__(self,shared_width=None):
        super().__init__()
        self.shared_readout=None
        if shared_width is None:
            self.gain = nn.Sequential(nn.Linear(9, 32), nn.SiLU(), nn.Linear(32, 2))
            nn.init.zeros_(self.gain[-1].weight);nn.init.zeros_(self.gain[-1].bias)
        else:
            self.shared_readout=nn.Linear(shared_width,2)
            nn.init.zeros_(self.shared_readout.weight);nn.init.zeros_(self.shared_readout.bias)

    def forward(self, scene, start, result, dt, accepted,shared_features=None):
        dv = result.v - start.v
        dw = result.omega - start.omega
        linear = dv.norm(dim=-1) * dt / scene.length
        angular = dw.norm(dim=-1) * dt
        features = torch.stack((
            torch.full_like(linear, math.log(dt * 120)),
            torch.log1p(scene.mass).expand_as(linear),
            torch.log1p(linear),
            torch.log1p(angular),
            torch.log1p(start.omega.norm(dim=-1) * dt),
        ), dim=-1)
        kinds = F.one_hot(scene.kind.long(), num_classes=4)
        features = torch.cat((features, kinds.expand(*linear.shape, 4)), dim=-1)
        if self.shared_readout is not None:
            if shared_features is None:raise ValueError('Shared endpoint velocity requires this step\'s backbone latent')
            gain=self.shared_readout(shared_features.to(self.shared_readout.weight.dtype)).tanh().to(dv.dtype)
        else:
            gain = self.gain(features.to(self.gain[0].weight.dtype)).tanh().to(dv.dtype)
        correction = torch.cat((gain[..., :1] * dv, gain[..., 1:] * dw), dim=-1) * scene.dof
        if getattr(scene, 'mimic_spec', None) is not None:
            from .general_joint_system import tangent
            correction = tangent(scene, result).primal(correction)
        correction = torch.where(accepted[:, None, None], correction, 0.)
        return State(result.x, result.q, result.v + correction[..., :3],
                     result.omega + correction[..., 3:], result.history)
