"""Independent learned rollouts, held-out metrics, and paired reference replays."""

import hashlib
import json
import sys
import time

import numpy as np
import torch

from .data import ROOT, load_trajectory
from . import math3d as so3
from .model import SharedSolver
from .train import rollout


def coordinates(sequence, field):
    return torch.cat([getattr(state, field) for state in sequence], 0)


def metrics(scene, sequence, states, accepted):
    n = len(sequence)
    x = coordinates(sequence, "x")
    q = coordinates(sequence, "q")
    error = (x-states["x"][:n]).norm(dim=-1)[:, scene.movable]/scene.length
    angular = so3.log(so3.multiply(q, so3.conjugate(states["q"][:n]))).norm(dim=-1)
    angular = angular[:, scene.movable & (scene.kind >= 2)]
    distance = (x[-1]-x[0]).norm(dim=-1)[scene.movable].sum()
    ref_distance = (states["x"][n-1]-states["x"][0]).norm(dim=-1)[scene.movable].sum()
    return {
        "completed_steps": n-1, "accepted_full_horizon": accepted and n == len(states["x"]),
        "position_rmse_over_L": float(error.square().mean().sqrt()),
        "position_error_p95_over_L": float(torch.quantile(error.flatten(), 0.95)),
        "final_position_rmse_over_L": float(error[-1].square().mean().sqrt()),
        "angular_rmse_radians": float(angular.square().mean().sqrt()) if angular.numel() else None,
        "motion_ratio_vs_reference": float(distance/ref_distance.clamp_min(1e-12)),
    }


def project(x):
    return np.stack((x[..., 0]+0.35*x[..., 1], x[..., 2]+0.2*x[..., 1]), -1)


def render_segments(scene, x, q):
    if "triangles" in scene.values:
        triangles = scene.triangles.cpu().numpy()
        pairs = sorted({tuple(sorted((int(a), int(b)))) for row in triangles for a,b in zip(row, np.roll(row, -1))})
        return project(x[np.asarray(pairs)])
    segments = []
    for i in range(scene.n):
        if np.allclose(scene.rest_x[i].cpu().numpy(), 0) and not bool(scene.movable[i]):
            continue
        axis = np.array([0., 0., 1.]) if scene.metadata["family"] == "rod" else np.array([1., 0., 0.])
        vector = axis+2*np.cross(q[i, :3], np.cross(q[i, :3], axis)+q[i, 3]*axis)
        length = 0.1 if scene.metadata["family"] == "rod" else (0.24 if bool(scene.movable[i]) else 0.04)
        segments.append([x[i]-0.5*length*vector, x[i]+0.5*length*vector])
    return project(np.asarray(segments))


def render_replay(scene, predicted, states, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.collections import LineCollection

    px = coordinates(predicted, "x").cpu().numpy()
    pq = coordinates(predicted, "q").cpu().numpy()
    rx, rq = states["x"].cpu().numpy(), states["q"].cpu().numpy()
    horizon = min(len(px), len(rx))
    ref_segments = [render_segments(scene, rx[i], rq[i]) for i in range(horizon)]
    new_segments = [render_segments(scene, px[i], pq[i]) for i in range(horizon)]
    points = np.concatenate([item.reshape(-1, 2) for item in ref_segments+new_segments])
    lower, upper = points.min(0)-0.05, points.max(0)+0.05
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    lines = []
    for ax, title, color in zip(axes, ("Newton VBD reference", "Shared learned solver"), ("#c97935", "#168c9b")):
        line = LineCollection([], colors=color, linewidths=1.8)
        ax.add_collection(line); lines.append(line)
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), title=title, xlabel="projected x (m)", ylabel="projected z (m)")
        ax.set_aspect("equal"); ax.grid(alpha=0.2)
    title = fig.suptitle(scene.metadata["family"])
    def update(frame):
        lines[0].set_segments(ref_segments[frame]); lines[1].set_segments(new_segments[frame])
        title.set_text(f"{scene.metadata['family']} | t={frame*scene.metadata['dt']:.3f}s | no contact pilot")
        return lines
    movie = FuncAnimation(fig, update, frames=horizon, interval=70, blit=False)
    movie.save(path, writer=PillowWriter(fps=14))
    update(horizon-1)
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def evaluate(config):
    torch.set_num_threads(2)
    # An attempted teacher import in deployment now fails instead of silently
    # providing a fallback. Generation is a separate command/process.
    sys.modules["newton"] = None
    checkpoint_path = ROOT/config["checkpoint_dir"]/"shared.pt"
    checkpoint = torch.load(checkpoint_path, map_location=config["device"], weights_only=True)
    if checkpoint["config"] != config:
        raise RuntimeError("Evaluation config differs from the trained checkpoint")
    model = SharedSolver(config["model"]).to(config["device"])
    model.load_state_dict(checkpoint["model"]); model.eval()
    output = ROOT/config["report_dir"]
    manifest = json.loads((output/"data_manifest.json").read_text())
    if hashlib.sha256((output/"data_manifest.json").read_bytes()).hexdigest() != checkpoint["data_manifest_sha256"]:
        raise RuntimeError("Dataset manifest changed since training")
    test_entries = [entry for entry in manifest["scenes"] if entry["split"] == "test"]
    if len(test_entries) != len(config["families"]) or {entry["family"] for entry in test_entries} != set(config["families"]):
        raise RuntimeError("Every pilot family needs its own held-out test scene")
    report = {"status": "RUNNING", "checkpoint": str(checkpoint_path.relative_to(ROOT)), "checkpoint_sha256": hashlib.sha256(checkpoint_path.read_bytes()).hexdigest(), "checkpoint_step": checkpoint["step"], "newton_import_blocked": True, "same_checkpoint_all_families": True, "cases": [], "scope": "Held-out pilot parameter scenes; contact scope follows config. Execution status is not a dynamics quality or official-demo acceptance claim."}
    for entry in test_entries:
        path = ROOT/entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError("Test scene hash changed")
        scene, states = load_trajectory(path, config["device"])
        torch.cuda.synchronize()
        start = time.perf_counter()
        learned, accepted = rollout(model, scene, states, config["dt"], config["steps"])
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        physical, physical_accepted = rollout(model, scene, states, config["dt"], config["steps"], learned=False)
        # Poison all future teacher arrays: the rollout must depend on the
        # initial state only, with metrics computed separately afterwards.
        poisoned = {key: value.clone() for key, value in states.items()}
        for value in poisoned.values():
            value[1:] = float("nan")
        isolated, _ = rollout(model, scene, poisoned, config["dt"], min(3, config["steps"]))
        independent_error = float((coordinates(isolated, "x")-coordinates(learned[:len(isolated)], "x")).abs().max())
        if independent_error > 1e-6 or sys.modules.get("newton") is not None:
            raise RuntimeError("Inference isolation failed")
        record = {"family": entry["family"], "scene": entry["path"], "learned": metrics(scene, learned, states, accepted), "physical_only": metrics(scene, physical, states, physical_accepted), "rollout_wall_seconds": elapsed, "future_teacher_poison_max_error": independent_error}
        prediction_path = output/f"{entry['family']}_test_prediction.npz"
        np.savez_compressed(prediction_path, **{key: coordinates(learned, key).cpu().numpy() for key in ("x", "q", "v", "omega")})
        replay_path = output/f"{entry['family']}_comparison.gif"
        render_replay(scene, learned, states, replay_path)
        record["replay"] = str(replay_path.relative_to(ROOT))
        report["cases"].append(record)
        (output/"evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(record, indent=2), flush=True)
    report["status"] = "PILOT_EXECUTED" if all(case["learned"]["accepted_full_horizon"] for case in report["cases"]) else "PILOT_ROLLOUT_FAILED"
    (output/"evaluation.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
