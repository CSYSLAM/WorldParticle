"""Continuous local hinge differences across the atan2 representation cut."""
import torch


def difference(scene,angle,reference):
    delta=angle-reference
    if getattr(scene,'periodic_bending',False):
        delta=torch.atan2(delta.sin(),delta.cos())
    return delta
