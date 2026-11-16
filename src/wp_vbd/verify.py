"""Behavioral checks of the pilot before its first training run."""

import io
import json
import time

import torch

from .data import ROOT, Scene, State, load_trajectory, select_state
from .model import SharedSolver
from .physics import elastic_nodes
from .feasibility import accepted_scale, tetrahedron_prefix_minimum


def energy(scene, x):
    total = x.new_zeros(())
    if len(scene.tets):
        points = x[:, scene.tets]
        f = (points[:, :, 1:]-points[:, :, :1]).transpose(-1, -2) @ scene.tet_Dm
        mu, lam = scene.tet_material[:, 0], scene.tet_material[:, :2].sum(-1)
        volume = (1/(6*torch.linalg.det(scene.tet_Dm))).abs()
        j = torch.linalg.det(f)
        total = total + (volume*(0.5*mu*f.square().sum((-1, -2))+0.5*lam*(j-1-mu/lam).square())).sum()
    if len(scene.triangles):
        points = x[:, scene.triangles]
        f = (points[:, :, 1:]-points[:, :, :1]).transpose(-1, -2) @ scene.tri_Dm
        mu, lam = scene.tri_material[:, 0], scene.tri_material[:, :2].sum(-1)
        # Zero-material collision-only surface faces do not add an energy.
        alpha = 1+mu/lam.clamp_min(1e-30)
        j = torch.linalg.cross(f[..., 0], f[..., 1]).norm(dim=-1)
        total = total+(scene.tri_area*(0.5*mu*f.square().sum((-1,-2))+0.5*lam*(j-alpha).square())).sum()
    valid = (scene.bend_edges >= 0).all(-1) & (scene.bend_material[:, 0] > 0)
    if bool(valid.any()):
        points = x[:, scene.bend_edges[valid]]
        normal = lambda v: v/v.norm(dim=-1, keepdim=True)
        n1 = normal(torch.linalg.cross(points[:, :, 2]-points[:, :, 0], points[:, :, 3]-points[:, :, 0]))
        n2 = normal(torch.linalg.cross(points[:, :, 3]-points[:, :, 1], points[:, :, 2]-points[:, :, 1]))
        e = normal(points[:, :, 3]-points[:, :, 2])
        angle = torch.atan2((torch.linalg.cross(n1,n2)*e).sum(-1), (n1*n2).sum(-1))
        total = total+(0.5*scene.bend_material[valid, 0]*scene.bend_length[valid]*(angle-scene.bend_angle[valid]).square()).sum()
    return total


def permuted_scene(scene, permutation):
    inverse = torch.argsort(permutation)
    per_node = {"rest_x", "rest_q", "mass", "movable", "inertia", "kind", "object_id"}
    indices = {"tets", "triangles", "bend_edges", "joints"}
    values = {}
    for key, value in scene.values.items():
        if key in per_node:
            values[key] = value[permutation]
        elif key in indices:
            values[key] = torch.where(value >= 0, inverse[value.clamp_min(0)], value)
        else:
            values[key] = value
    return Scene(values, scene.metadata)


def verify(config):
    torch.manual_seed(config["seed"])
    torch.set_num_threads(2)
    start = time.perf_counter()
    results = {}
    model = SharedSolver(config["model"]).to(config["device"])
    # Nonzero heads are essential: zero heads would fail to test the learned
    # tokenizer/attention/merge path for permutation or batch isolation.
    for head in model.heads:
        torch.nn.init.normal_(head[-1].weight, std=0.005)
    model.eval()
    for family in config["families"]:
        scene, states = load_trajectory(ROOT/config["data_dir"]/f"{family}_00.npz", config["device"])
        current = select_state(states, slice(4, 6))
        predicted, diagnostics = model(scene, current, config["dt"])
        objective = predicted.x.square().mean()+predicted.q.square().mean()
        model.zero_grad()
        objective.backward()
        assert all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None), family
        assert bool(diagnostics["accepted"].all()), family
        fixed_error = float((predicted.x[:, ~scene.movable]-current.x[:, ~scene.movable]).abs().max()) if bool((~scene.movable).any()) else 0.0
        assert fixed_error == 0
        with torch.no_grad():
            order = torch.randperm(scene.n, device=scene.device)
            shuffled = permuted_scene(scene, order)
            reordered, _ = model(shuffled, State(*(value[:, order] for value in (current.x, current.q, current.v, current.omega))), config["dt"])
            position_error = float((reordered.x[:, torch.argsort(order)]-predicted.x).abs().max())
            quaternion_error = float((reordered.q[:, torch.argsort(order)]-predicted.q).abs().max())
            one, _ = model(scene, State(*(value[:1] for value in (current.x, current.q, current.v, current.omega))), config["dt"])
            isolation_error = float((one.x-predicted.x[:1]).abs().max())
        assert position_error < 2e-6 and quaternion_error < 2e-6, (family, position_error, quaternion_error)
        assert isolation_error < 2e-6, (family, isolation_error)
        results[family] = {"finite_forward_backward": True, "fixed_dof_error": fixed_error, "permutation_position_error": position_error, "permutation_quaternion_error": quaternion_error, "batch_isolation_error": isolation_error}
        if family in ("softbody", "cloth"):
            values = {key: value.double() if value.is_floating_point() else value for key,value in scene.values.items()}
            double_scene = Scene(values, scene.metadata)
            point = double_scene.rest_x[None]+0.002*torch.randn(1, scene.n, 3, device=scene.device, dtype=torch.float64)
            direction = torch.randn_like(point)
            gradient, _ = elastic_nodes(double_scene, point)
            epsilon = 1e-6
            fd = (energy(double_scene, point+epsilon*direction)-energy(double_scene, point-epsilon*direction))/(2*epsilon)
            analytic = (gradient*direction).sum()
            relative_error = float((fd-analytic).abs()/analytic.abs().clamp_min(1))
            balance = float(gradient.sum(1).abs().max())
            assert relative_error < 1e-5 and balance < 1e-7, (family, relative_error, balance)
            results[family].update(elastic_directional_derivative_relative_error=relative_error, internal_force_balance_error=balance)
        if family == "softbody":
            point = scene.rest_x[None]
            end = point.clone()
            end[..., :2] = -2*end[..., :2]
            alpha, valid = accepted_scale(scene, point, end)
            minimum = tetrahedron_prefix_minimum(scene, point, end, torch.ones_like(alpha))
            assert float(minimum) < 1e-5 and 0 < float(alpha) < 1/3 and bool(valid.all())
            results[family]["returning_inversion_path"] = {"full_prefix_minimum": float(minimum), "accepted_alpha": float(alpha), "passed": True}
    checkpoint_checks = {"status": "NOT_RUN"}
    path = ROOT/config["checkpoint_dir"]/"shared.pt"
    if path.exists():
        checkpoint = torch.load(path, map_location=config["device"], weights_only=True)
        initial = torch.load(path.with_name("initial.pt"), map_location=config["device"], weights_only=True)
        trained = SharedSolver(config["model"]).to(config["device"])
        trained.load_state_dict(checkpoint["model"]); trained.eval()
        updates = {}
        for prefix in ("encoder.", "decoder.", "relation_encoder.", "heads.0.", "heads.1.", "heads.2.", "heads.3."):
            updates[prefix] = sum(float((value-initial["model"][key]).square().sum()) for key,value in checkpoint["model"].items() if key.startswith(prefix))**0.5
            assert updates[prefix] > 1e-6, (prefix, "parameters never changed")
        restart_errors = {}
        with torch.no_grad():
            for family in config["families"]:
                scene, states = load_trajectory(ROOT/config["data_dir"]/f"{family}_06.npz", config["device"])
                current = select_state(states, slice(0, 1))
                for _ in range(6):
                    current, _ = trained(scene, current, config["dt"])
                buffer = io.BytesIO()
                torch.save({key: getattr(current, key).cpu() for key in ("x", "q", "v", "omega")}, buffer)
                buffer.seek(0)
                restored = torch.load(buffer, map_location=config["device"], weights_only=True)
                recovered = State(*(restored[key] for key in ("x", "q", "v", "omega")))
                for _ in range(6):
                    current, _ = trained(scene, current, config["dt"])
                    recovered, _ = trained(scene, recovered, config["dt"])
                error = max(float((getattr(current,key)-getattr(recovered,key)).abs().max()) for key in ("x", "q", "v", "omega"))
                assert error < 1e-6, (family, "snapshot restart differs", error)
                restart_errors[family] = error
                try:
                    Scene(scene.values, {**scene.metadata, "contacts": True})
                except ValueError:
                    pass
                else:
                    raise AssertionError("Unsupported contact mode was accepted")
        checkpoint_checks = {"status": "PASS", "checkpoint_step": checkpoint["step"], "parameter_update_norms": updates, "snapshot_restart_errors": restart_errors, "unsupported_contact_rejected": True, "scope": "Pilot Markov state only; does not validate unavailable material/contact/multiturn histories"}
    report = {"status": "PASS", "config": config, "results": results, "trained_checkpoint_checks": checkpoint_checks, "wall_seconds": time.perf_counter()-start, "scope": "Pilot only. No contact/DAT proof or full Newton demo acceptance."}
    output = ROOT/config["report_dir"]/"verification.json"
    output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))
    return report
