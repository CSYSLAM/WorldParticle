"""Fixed-shape CUDA replay of the exact learned step, without a teacher solver."""

import torch

from .data import State


class CapturedStepper:
    def __init__(self, model, scene, state, dt, *, learned=True):
        if scene.device.type != "cuda":
            raise ValueError("Captured inference requires CUDA")
        self.input = state.clone()
        self.force = torch.zeros_like(state.x)
        self.torque = torch.zeros_like(state.omega) if bool((scene.kind >= 2).any()) else None
        options={} if learned else {'learned':False}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), torch.no_grad():
            for _ in range(3):
                model(scene, self.input, dt, force=self.force, torque=self.torque, **options)
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream), torch.no_grad():
            self.output, self.diagnostic = model(scene, self.input, dt, force=self.force, torque=self.torque, **options)
        torch.cuda.current_stream().wait_stream(stream)

    @torch.no_grad()
    def step(self, state, force=None, torque=None):
        self.input.copy_(state)
        self.force.zero_() if force is None else self.force.copy_(force)
        if self.torque is not None:
            self.torque.zero_() if torque is None else self.torque.copy_(torque)
        self.graph.replay()
        # Stable owned snapshots: a rejected replay must not overwrite the last
        # committed live state or an earlier evaluation/restart snapshot.
        return self.output.clone(), self.diagnostic
