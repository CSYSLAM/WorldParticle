"""Asset + physical parameters -> native Newton viewer, learned stepping only."""

import argparse
from copy import deepcopy
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
sys.path.insert(0, str(ROOT/"src"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", type=Path, default=ROOT/"assets/live/cloth_slide.json")
    parser.add_argument("--checkpoint", type=Path, default=ROOT/"checkpoints/live_v2/shared.pt")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=JSON", help="e.g. material.mu=100 or plane.ground_mu=0.6")
    parser.add_argument("--friction", type=float, help="Set both particle and ground mu (their geometric mean is used)")
    parser.add_argument("--frames", type=int, help="Close after this many rendered frames (default: interactive)")
    parser.add_argument("--screenshot", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--paused", action="store_true")
    parser.add_argument("--eager", action="store_true", help="Disable fixed-shape CUDA replay")
    parser.add_argument("--no-loop", action="store_true", help="Continue beyond the 2-second pilot horizon")
    parser.add_argument("--substeps", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from wp_vbd.assets import load_asset_spec, build_asset
    from wp_vbd.model import SharedSolver
    from wp_vbd.runtime import CapturedStepper
    from wp_vbd import math3d as so3

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    checkpoint_bytes=args.checkpoint.read_bytes()
    checkpoint_digest=hashlib.sha256(checkpoint_bytes).hexdigest()
    checkpoint = torch.load(io.BytesIO(checkpoint_bytes), map_location="cuda:0", weights_only=True)
    config = checkpoint["config"]
    mixed_model=checkpoint.get('architecture') in ('mixed_response_v1','mixed_coarse_v2','mixed_coupled_v3','mixed_conditioned_v4','shared_objective_v5','shared_bending_v7','shared_history_v9','shared_cycle_v10','shared_boundary_v11','shared_protected_v17','shared_inverse_v18','shared_vector_v19','shared_hierarchy_v20')
    device=config.get('device','cuda:0')
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=config["newton_root"], text=True).strip()
    if source != config["newton_sha"]:
        raise RuntimeError("Newton source differs from the checkpoint source pin")
    sys.path.insert(0, config["newton_root"])
    import newton
    import warp as wp

    def forbid_teacher(*unused, **kwargs):
        raise RuntimeError("SolverVBD is forbidden in the live inference process")
    newton.solvers.SolverVBD.__init__ = forbid_teacher
    newton.solvers.SolverVBD.step = forbid_teacher
    wp.config.kernel_cache_dir = str(ROOT/".local/p0/warp-cache")
    wp.config.deterministic = wp.DeterministicMode.NOT_GUARANTEED
    wp.init()
    wp.set_device(device)
    stream = wp.stream_from_torch(torch.cuda.current_stream())
    if mixed_model:
        from wp_vbd.mixed_solver import MixedSolver
        from wp_vbd.mixed_demo import build_mixed
        from wp_vbd.mixed_state import export_mixed,tensor_scene,initial_state as mixed_initial_state
    model = (MixedSolver(config['model']) if mixed_model else SharedSolver(config["model"])).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    spec = load_asset_spec(args.asset)
    if mixed_model != (spec['family']=='mixed'):
        raise ValueError('Asset family and checkpoint architecture do not match')
    if mixed_model and spec.get('contract')!=config['contract']:
        zero_attachment_subset=config['contract']=='finite_elastic_plane_attachment_v1' and spec.get('contract')=='finite_elastic_plane_ipc_v1' and not spec.get('attachments')
        if not zero_attachment_subset:raise ValueError('Mixed asset must declare the checkpoint physical contract or its zero-attachment subset')
    for override in args.set:
        key, value = override.split("=", 1)
        container = spec
        parts = key.split(".")
        for part in parts[:-1]:
            container = container[int(part)] if isinstance(container,list) else container[part]
        if not isinstance(container,list) and parts[-1] not in container:
            raise ValueError(f"Unknown asset key: {key}")
        container[int(parts[-1]) if isinstance(container,list) else parts[-1]] = json.loads(value)
    if args.friction is not None:
        if spec.get("contacts") != "plane":
            raise ValueError("--friction requires a plane-contact asset")
        spec["plane"].update(particle_mu=args.friction, ground_mu=args.friction)
        if mixed_model:
            for part in spec['objects']:part['friction']=args.friction
    if spec.get("contacts") and config["scope"].get("contacts") == "UNSUPPORTED":
        raise ValueError("This checkpoint was not trained for contact")
    if abs(spec.get("dt", config["dt"])-config["dt"]) > 1e-10:
        raise ValueError("This pilot checkpoint requires its trained timestep")
    if args.substeps < 1 or (args.frames is not None and args.frames < 1):
        raise ValueError("substeps and frame cap must be positive")
    dt = config["dt"]
    horizon = config["steps"]*dt
    viewer = newton.viewer.ViewerGL(width=args.width, height=args.height, vsync=False, paused=args.paused)
    viewer.gui.ui.io.set_ini_filename(str(ROOT/".local/live-viewer.ini"))
    title="WorldParticle | shared mixed inference | joint rigid soft cloth cable" if mixed_model else "WorldParticle | legacy pilot | asset + material"
    if checkpoint.get('architecture')=='mixed_coarse_v2':title="WorldParticle | v2 coarse inference | joint rigid soft cloth cable"
    if config['model'].get('coupled_attachments'):title="WorldParticle | shared inference | cloth carries rigid body"
    viewer.renderer.set_title(title)
    pending = deepcopy(spec)
    state = scene = render_model = render_state = None
    simulation_time, resets, total_steps = 0.0, 0, 0
    needs_reset, failure = False, ""
    loop = not args.no_loop
    timings = []
    parameter_history = []
    stepper, initial_state = None, None

    def sync_render():
        with wp.ScopedStream(stream):
            if mixed_model:
                p=render_model.particle_count;n=render_model.body_count
                wp.to_torch(render_state.particle_q).copy_(state.x[0,:p])
                wp.to_torch(render_state.particle_qd).copy_(state.v[0,:p])
                pose=wp.to_torch(render_state.body_q)
                pose[:,:3].copy_(state.x[0,p:p+n]-so3.rotate(state.q[0,p:p+n],scene.body_com[p:p+n]))
                pose[:,3:].copy_(state.q[0,p:p+n])
                twist=wp.to_torch(render_state.body_qd)
                twist[:,:3].copy_(state.v[0,p:p+n]);twist[:,3:].copy_(state.omega[0,p:p+n])
            elif render_model.particle_count:
                wp.to_torch(render_state.particle_q).copy_(state.x[0])
                wp.to_torch(render_state.particle_qd).copy_(state.v[0])
            else:
                n = render_model.body_count
                pose = wp.to_torch(render_state.body_q)
                pose[:, :3].copy_(state.x[0, :n]-so3.rotate(state.q[0, :n], scene.body_com[:n]))
                pose[:, 3:].copy_(state.q[0, :n])
                twist = wp.to_torch(render_state.body_qd)
                twist[:, :3].copy_(state.v[0, :n])
                twist[:, 3:].copy_(state.omega[0, :n])

    def reset(apply=False):
        nonlocal state, scene, render_model, render_state, simulation_time, failure, resets, spec, stepper, initial_state
        if render_model is not None and not apply:
            from wp_vbd.data import State
            state = State(*(getattr(initial_state,key).clone() for key in ("x","q","v","omega")))
            simulation_time, failure = 0.0, ""
            resets += 1
            sync_render()
            return
        candidate = deepcopy(pending if apply else spec)
        # Validate and build before replacing the current scene.
        if mixed_model:
            replacement,groups=build_mixed(candidate,device,newton,wp)
            static,metadata=export_mixed(replacement,groups,candidate,newton)
            geometry=tensor_scene(static,metadata,device)
            initial=mixed_initial_state(geometry,replacement)
        else:replacement, geometry, initial = build_asset(candidate, device, newton, wp)
        render_model, scene, state = replacement, geometry, initial
        render_state = render_model.state()
        initial_state = state.detach()
        stepper = None if args.eager else CapturedStepper(model,scene,state,dt)
        spec = candidate
        viewer.set_model(render_model)
        sync_render()
        simulation_time, failure = 0.0, ""
        resets += 1
        parameter_history.append({"material":deepcopy(spec.get('objects') if mixed_model else spec['material']),"plane":deepcopy(spec.get('plane')),"gravity":spec.get('gravity'),"attachments":deepcopy(spec.get('attachments',[])),"interobject_contacts":spec.get('interobject_contacts')})

    reset()
    center = scene.rest_x[scene.movable].mean(0).cpu().numpy()
    distance = max(.85, float((scene.rest_x.max(0).values-scene.rest_x.min(0).values).norm())*2)
    viewer.set_camera(wp.vec3(*(center+np.array([distance, -distance, .8*distance]))))
    viewer.camera.look_at(center)
    if mixed_model:
        viewer.set_camera(wp.vec3(2.5,-3,2));viewer.camera.look_at((0,.1,.15))

    def request_reset():
        nonlocal needs_reset
        needs_reset = "current"
    viewer.set_reset_callback(request_reset)

    def ui(imgui):
        nonlocal needs_reset, loop
        label='Shared coarse response v2' if checkpoint.get('architecture')=='mixed_coarse_v2' else 'Shared mixed response'
        if checkpoint.get('architecture')=='mixed_coupled_v3':label='Shared coupled response v3'
        if checkpoint.get('architecture')=='mixed_conditioned_v4':label='Shared conditioned response v4'
        if checkpoint.get('architecture') in ('shared_objective_v5','shared_bending_v7','shared_history_v9','shared_cycle_v10','shared_boundary_v11'):label='Shared material response'
        if checkpoint.get('architecture')=='shared_protected_v17':label='Shared protected factor response'
        if checkpoint.get('architecture')=='shared_inverse_v18':label='Shared projected inverse response'
        if checkpoint.get('architecture')=='shared_vector_v19':label='Shared directional response'
        if checkpoint.get('architecture')=='shared_hierarchy_v20':label='Shared heterogeneous hierarchy'
        imgui.text(f"{label} (K={config['model']['unroll']})" if mixed_model else "Legacy shared pilot (K=2)")
        scope_text='Finite elastic + plane contact'
        if config['model'].get('coupled_attachments'):scope_text='Cloth <-> rigid + ground'
        imgui.text(scope_text if mixed_model else "Long-rollout quality: NOT ACCEPTED")
        if config['model'].get('coupled_attachments'):
            active=sum(a.get('enabled',True) and a['stiffness']>0 for a in spec.get('attachments',[]))
            imgui.text(f'Applied attachment links: {active}')
        imgui.text(f"Asset: {args.asset.name}")
        imgui.text(f"Entities: {scene.n}, dt: {dt:.6f} s")
        imgui.text(f"Sim time: {simulation_time:.3f} / {horizon:.1f} s pilot")
        if timings:
            milliseconds = 1000*np.median(timings[-60:])
            imgui.text(f"Inference: {milliseconds:.2f} ms/step")
            imgui.text(f"Sim/wall speed: {total_steps*dt/max(time.perf_counter()-start_wall,1e-6):.2f}x")
        _, loop = imgui.checkbox("Loop at pilot horizon", loop)
        imgui.separator()
        imgui.text("Physical parameters")
        if imgui.button("Apply parameters + reset"):
            needs_reset = "apply"
        if mixed_model:
            uniform=float(np.sqrt(pending['plane']['particle_mu']*pending['plane']['ground_mu']))
            imgui.set_next_item_width(220)
            changed,uniform=imgui.input_float('Friction mu',uniform,.05,0,'%.3f')
            if changed:
                pending['plane'].update(particle_mu=uniform,ground_mu=uniform)
                for part in pending['objects']:part['friction']=uniform
        parts=pending['objects'] if mixed_model else [pending]
        if pending.get('attachments'):
            stiffness=float(pending['attachments'][0]['stiffness'])
            imgui.set_next_item_width(220)
            changed,stiffness=imgui.input_float('Attachment k (N/m)',stiffness,50.,0.,'%.5g')
            if changed:
                for attachment in pending['attachments']:attachment['stiffness']=stiffness
        for part in parts:
            if mixed_model and not imgui.collapsing_header(part['name']+' material'):continue
            for key,value in part['material'].items():
                if key=='damping':continue
                imgui.set_next_item_width(220)
                names={'mass':'Mass (kg)','density':'Density (kg/m3)','linear_ke':'Linear stiffness (N/m)','angular_ke':'Angular stiffness (Nm/rad)'}
                if part.get('family')=='softbody':names.update(mu='Shear mu (Pa)',lmbd='Lame lambda (Pa)')
                elif part.get('family')=='cloth':names.update(mu='Membrane shear',lmbd='Membrane area',bending='Bending coefficient')
                else:names.update(stretch='Stretch coefficient',shear='Shear coefficient',bending='Bending coefficient',twist='Twist coefficient')
                label=names.get(key,key)+'##'+part.get('name','')+key
                _,part['material'][key]=imgui.input_float(label,float(value),max(abs(value)*.05,.001),0,'%.5g')
        if pending.get('attachments') and imgui.collapsing_header('Cross-object attachments'):
            for index,attachment in enumerate(pending['attachments']):
                key=f'Attachment {index} stiffness (N/m)'
                _,attachment['stiffness']=imgui.input_float(key,float(attachment['stiffness']),50.,0.,'%.5g')
        if pending.get("contacts") == "plane" and (not mixed_model or imgui.collapsing_header('Contact details')):
            for key in ("particle_mu", "ground_mu", "ke", "kd"):
                value = float(pending["plane"][key])
                imgui.set_next_item_width(220)
                _, pending["plane"][key] = imgui.input_float(key.replace('_',' '), value, .05 if "mu" in key else 10., 0, "%.5g")
            mixed = np.sqrt(spec["plane"]["particle_mu"]*spec["plane"]["ground_mu"])
            imgui.text(f"Applied mixed friction mu: {mixed:.3f}")
            if mixed_model:
                if imgui.button('Use ground mu on all objects'):
                    mu=pending['plane']['ground_mu'];pending['plane']['particle_mu']=mu
                    for part in pending['objects']:part['friction']=mu
                    needs_reset='apply'
        imgui.text("Zero material damping pilot; plane contact only")
        if failure:
            imgui.text_wrapped(f"Paused: {failure}")

    viewer.register_ui_callback(ui, position="panel")
    count, start_wall = 0, time.perf_counter()
    try:
        while viewer.is_running() and (args.frames is None or count < args.frames):
            if needs_reset:
                try:
                    reset(apply=needs_reset == "apply")
                except (ValueError, KeyError) as error:
                    failure = str(error)
                needs_reset = False
            advance = viewer.should_step()
            if advance and not failure:
                if loop and simulation_time >= horizon-1e-9:
                    reset()
                for _ in range(args.substeps):
                    with wp.ScopedStream(stream):
                        render_state.clear_forces()
                        viewer.apply_forces(render_state)
                    force = torch.zeros_like(state.x)
                    torque = None
                    if mixed_model:
                        p=render_model.particle_count;n=render_model.body_count
                        force[0,:p]=wp.to_torch(render_state.particle_f)
                        force[0,p:p+n]=wp.to_torch(render_state.body_f)[:,:3]
                        torque=torch.zeros_like(state.omega);torque[0,p:p+n]=wp.to_torch(render_state.body_f)[:,3:]
                    elif render_model.particle_count:
                        force[0] = wp.to_torch(render_state.particle_f)
                    else:
                        n = render_model.body_count
                        force[0, :n] = wp.to_torch(render_state.body_f)[:, :3]
                        torque = torch.zeros_like(state.omega)
                        torque[0, :n] = wp.to_torch(render_state.body_f)[:, 3:]
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    with torch.no_grad():
                        candidate, diagnostic = stepper.step(state,force,torque) if stepper is not None else model(scene, state, dt, force=force, torque=torque)
                    finite = all(bool(torch.isfinite(getattr(candidate,key)).all()) for key in ("x", "q", "v", "omega"))
                    if not bool(diagnostic["accepted"].all()) or not finite:
                        failure = "Proposal rejected; state retained. Reset to resume."
                        break
                    torch.cuda.synchronize()
                    timings.append(time.perf_counter()-begin)
                    state = candidate.detach()
                    simulation_time += dt
                    total_steps += 1
                sync_render()
            with wp.ScopedStream(stream):
                viewer.begin_frame(simulation_time)
                viewer.log_state(render_state)
                viewer.end_frame()
            count += 1
        if args.screenshot and viewer.is_running():
            from PIL import Image
            args.screenshot.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(viewer.get_frame(render_ui=True).numpy()).save(args.screenshot)
    finally:
        wall = time.perf_counter()-start_wall
        report = {"status": "FAIL" if failure else "LIVE_INFERENCE_EXECUTED", "failure": failure, "runtime": "EAGER" if args.eager else "CUDA_GRAPH", "asset": str(args.asset.resolve()), "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_digest, "checkpoint_step": checkpoint["step"], "newton_source": source, "teacher_constructor_and_step": "FORBIDDEN", "trajectory_files_read": 0, "frames": count, "physical_steps": total_steps, "resets": resets, "wall_seconds": wall, "simulated_seconds_per_wall_second": total_steps*dt/max(wall,1e-9), "inference_p50_ms": float(1000*np.median(timings)) if timings else None, "parameter_history": parameter_history, "final_position": state.x[0].cpu().tolist(), "scope": "Small asset pilot, zero material damping; plane contact for particles only; no full official-demo acceptance"}
        if mixed_model:
            report.update(architecture=model.architecture_id,contract=config['contract'],scope=config['scope'])
        viewer.close()
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps({key:value for key,value in report.items() if key not in ("final_position", "parameter_history")}, indent=2))
    if failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
