"""Balanced shared training with scene-disjoint validation and short rollouts."""

import hashlib
import json
from pathlib import Path
import random
import time

import torch

from .data import ROOT, load_trajectory, select_state
from . import math3d as so3
from .model import SharedSolver


def state_loss(scene, predicted, target):
    mask = scene.movable[None, :, None]
    position = ((predicted.x-target.x)/scene.length).square()
    rotation = so3.log(so3.multiply(predicted.q, so3.conjugate(target.q))).square()
    rotation = rotation*(scene.kind >= 2)[None, :, None]
    return ((1e4*position+100*rotation)*mask).sum()/(mask.sum()*predicted.x.shape[0]*3).clamp_min(1)


def rollout(model, scene, states, dt, steps, *, learned=True):
    current = select_state(states, slice(0, 1))
    trajectory = [current.detach()]
    accepted = True
    for _ in range(steps):
        current, diagnostics = model(scene, current, dt, learned=learned)
        if not bool(diagnostics["accepted"].all()) or not all(bool(torch.isfinite(value).all()) for value in (current.x, current.q, current.v, current.omega)):
            accepted = False
            break
        trajectory.append(current.detach())
    return trajectory, accepted


@torch.no_grad()
def validation(model, samples, config):
    model.eval()
    scores = {}
    for family, scene, states in samples:
        indices = torch.arange(0, len(states["x"])-1, max(1, (len(states["x"])-1)//8), device=scene.device)
        predicted, _ = model(scene, select_state(states, indices), config["dt"])
        one_step = float(state_loss(scene, predicted, select_state(states, indices+1)))
        count = config["training"].get("validation_rollout_steps", 12)
        sequence, accepted = rollout(model, scene, states, config["dt"], count)
        if accepted:
            last = sequence[-1]
            free = float(state_loss(scene, last, select_state(states, slice(len(sequence)-1, len(sequence)))))
        else:
            free = 1e6
        scores[family] = {"one_step_loss": one_step, "rollout_loss": free, "rollout_steps": count, "accepted": accepted}
    model.train()
    return sum(value["one_step_loss"]+value["rollout_loss"] for value in scores.values())/len(scores), scores


def train(config):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    rng = random.Random(config["seed"])
    manifest_path = ROOT/config["report_dir"]/"data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    verification = json.loads((ROOT/config["report_dir"]/"verification.json").read_text())
    if verification["status"] != "PASS":
        raise RuntimeError("Pilot behavioral checks must pass before training")
    if manifest["status"] != "PASS" or len(manifest["scenes"]) != len(config["families"])*config["scenes_per_family"]:
        raise RuntimeError("Complete, passing pilot data manifest required")
    samples = {family: [] for family in config["families"]}
    valid = []
    for entry in manifest["scenes"]:
        if entry["split"] == "test":
            continue
        path = ROOT/entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise RuntimeError(f"Data content changed: {path}")
        scene, states = load_trajectory(path, config["device"])
        if entry["split"] == "train":
            samples[entry["family"]].append((scene, states))
        else:
            valid.append((entry["family"], scene, states))
    if len(valid) != len(config["families"]) or any(not rows for rows in samples.values()):
        raise ValueError("Every family must have independent train/validation scenes")
    model = SharedSolver(config["model"]).to(config["device"])
    cfg = config["training"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    out = ROOT/config["checkpoint_dir"]; out.mkdir(parents=True, exist_ok=True)
    report_dir = ROOT/config["report_dir"]; report_dir.mkdir(parents=True, exist_ok=True)
    history = []
    start = time.perf_counter()
    validation_score, scores = validation(model, valid, config)
    best = validation_score
    history.append({"step": 0, "validation_score": validation_score, "validation": scores, "elapsed_seconds": time.perf_counter()-start})
    initial = {"model": model.state_dict(), "config": config, "step": 0, "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
    torch.save(initial, out/"initial.pt")
    torch.save(initial, out/"shared.pt")
    report = {"status": "RUNNING", "parameters": sum(p.numel() for p in model.parameters()), "config": config, "history": history, "best_validation_score": best, "wall_seconds": time.perf_counter()-start, "checkpoint": str((out/"shared.pt").relative_to(ROOT))}
    print(f"Initial validation score: {validation_score:.6g}", flush=True)
    for step in range(1, cfg["steps"]+1):
        family = config["families"][(step-1) % len(config["families"])]
        scene, states = rng.choice(samples[family])
        horizon = cfg["rollout_length"]
        indices = torch.randint(len(states["x"])-horizon, (cfg["batch_size"],), device=scene.device)
        current = select_state(states, indices)
        optimizer.zero_grad(set_to_none=True)
        loss = 0
        for offset in range(horizon):
            current, diagnostics = model(scene, current, config["dt"])
            if not bool(diagnostics["accepted"].all()):
                raise RuntimeError("Training proposal rejected; retain failure and diagnose geometry")
            loss = loss+state_loss(scene, current, select_state(states, indices+offset+1))/horizon
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at step {step}, family {family}")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip"], error_if_nonfinite=True)
        optimizer.step()
        if step % 20 == 0:
            print(f"step {step}/{cfg['steps']} {family} loss={float(loss):.6g} grad={float(norm):.4g} elapsed={time.perf_counter()-start:.1f}s", flush=True)
        if step % cfg["validation_interval"] == 0 or step == cfg["steps"]:
            validation_score, scores = validation(model, valid, config)
            row = {"step": step, "train_family": family, "train_loss": float(loss), "validation_score": validation_score, "validation": scores, "elapsed_seconds": time.perf_counter()-start}
            history.append(row)
            checkpoint = {"model": model.state_dict(), "config": config, "step": step, "data_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
            torch.save(checkpoint, out/"last.pt")
            if validation_score < best:
                best = validation_score
                torch.save(checkpoint, out/"shared.pt")
            report = {"status": "RUNNING", "parameters": sum(p.numel() for p in model.parameters()), "config": config, "history": history, "best_validation_score": best, "wall_seconds": time.perf_counter()-start, "checkpoint": str((out/"shared.pt").relative_to(ROOT))}
            (report_dir/"training.json").write_text(json.dumps(report, indent=2)+"\n")
            print(f"validation {step}: {validation_score:.6g}; best={best:.6g}", flush=True)
        if time.perf_counter()-start > cfg["max_seconds"]:
            break
    report["status"] = "COMPLETED" if step == cfg["steps"] else "BUDGET_REACHED"
    report["wall_seconds"] = time.perf_counter()-start
    report["checkpoint_sha256"] = hashlib.sha256((out/"shared.pt").read_bytes()).hexdigest()
    (report_dir/"training.json").write_text(json.dumps(report, indent=2)+"\n")
    return report
