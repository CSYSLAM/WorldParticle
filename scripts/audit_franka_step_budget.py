"""Compare timestep, response budget and query envelope at identical Franka states.

Only saved model states and controls are replayed. No dynamics teacher or
optimizer is used. A common held command spans the same 1/60 s interval in
every branch. This diagnoses one-step numerical behavior, not full-task quality.
"""

import argparse
import copy
import inspect
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--times', nargs='+', type=float, default=(1.9, 2.1, 2.5))
    parser.add_argument('--device', default='cuda:0', choices=('cpu', 'cuda:0'))
    args = parser.parse_args()
    from wp_vbd.newton_source import activate_newton_source, newton_revision
    activate_newton_source()
    import numpy as np
    import torch
    import warp as wp
    import newton
    from wp_vbd.franka_vbd_scene import build
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd.mixed_solver import MixedSolver
    from wp_vbd.gpu_ownership import claim_cuda
    from wp_vbd.gpu_budget import GpuBudget

    ownership = claim_cuda(args.device, ROOT)
    budget = GpuBudget(args.device, 12.)
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = str(ROOT/'.local/p0/warp-cache')
    wp.init(); wp.set_device(args.device)
    if args.device.startswith('cuda'):
        wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    run = json.loads((args.recording/'run.json').read_text())
    if run['newton_source']['revision'] != newton_revision():
        raise ValueError('Use the recorded Newton revision')
    if run['mode'] != 'shared_network_inference' or run['teacher_trajectory_reads']:
        raise ValueError('An independent model recording is required')
    with np.load(args.recording/'trajectory.npz') as archive:
        tape = {key: archive[key].copy() for key in archive.files}
    signature = inspect.signature(newton.solvers.SolverVBD)
    def forbidden(*unused, **options):
        raise AssertionError('No native dynamics allowed in budget audit')
    for cls in (newton.solvers.SolverVBD, newton.solvers.SolverFeatherstone):
        cls.__init__ = forbidden; cls.step = forbidden
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    network = MixedSolver(saved['model_config']).to(args.device).eval()
    network.load_state_dict(saved['model']); del saved
    calls = [0]
    def count(*unused): calls[0] += 1
    handle = network.protected_response.hierarchy.register_forward_hook(count)
    parameters = run['physical_parameters']
    rows = []
    cases = [('macro_3', 1, 3, .003), ('macro_30', 1, 30, .003),
             ('fine_10x3', 10, 3, .003), ('macro_3_envelope20mm', 1, 3, .02)]
    for requested_time in args.times:
        index = int(np.abs(tape['physical_time']-requested_time).argmin())
        actual_time = float(tape['physical_time'][index])
        for label, substeps, rounds, motion_budget in cases:
            e = build(wp, newton, device=args.device, soft_asset='block', start_time=run['initial_command_time'],
                material_mu=parameters['tet_material'][0], material_lambda=parameters['tet_material'][1],
                arm_drive_ke=parameters['arm_drive_ke'], grasp_x=parameters['grasp_x'],
                grasp_yaw=parameters['grasp_yaw'], grasp_height=parameters['grasp_height'],
                grip_open=parameters['grip_open'], grip_closed=parameters['grip_closed'],
                friction=parameters['friction_override'])
            # Preserve authored model constants, restore only the dynamic state.
            for name, key in (('particle_q','particle_position'), ('particle_qd','particle_velocity'),
                              ('body_q','body_pose'), ('body_qd','body_velocity')):
                getattr(e.state_0, name).assign(tape[key][index])
            held_command = tape['joint_target'][min(index+1, len(tape['joint_target'])-1)]
            e.control.joint_target_q.assign(held_command)
            bound = signature.bind(None, **{
                'integrate_with_external_rigid_solver': False, 'particle_enable_self_contact': False,
                'rigid_joint_linear_ke': parameters['joint_linear_ke'],
                'rigid_joint_angular_ke': parameters['joint_angular_ke'],
                'rigid_joint_linear_kd': 0., 'rigid_joint_angular_kd': 0.})
            bound.apply_defaults()
            config = {k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
            dt = 1/60/substeps
            scene = export_model(e.model, config, run['scene'], newton, dt, args.device)
            scene.metadata.update(detection_motion_budget=motion_budget, bounded_motion_forecast_target_only=True,
                joint_tree_coarse_modes='scalar_tree_twelve_columns_v307', separate_coarse_ray_step=True,
                articulation_material_motion_modes=True, soft_tangent_support_scale=8.,
                force_aware_material_anticipation=True, record_grasp_forces=True)
            solver = SurfaceJointExampleSolver(network,scene,dt,wp,newton,e.model,e.state_0)
            network.config['unroll'] = rounds
            diagnostics = []; initial_x = e.state_0.particle_q.numpy().astype(float)
            first_calls = calls[0]; begin = time.perf_counter(); failure = None
            with torch.no_grad():
                for substep in range(substeps):
                    try:
                        e.state_0.clear_forces(); e.state_1.clear_forces()
                        solver.step(e.state_0,e.state_1,e.control,None,dt)
                        e.state_0,e.state_1=e.state_1,e.state_0
                    except Exception as error:
                        failure = repr(error)
                    detail = solver.last_diagnostic or {}
                    diagnostics.append({key: copy.deepcopy(value) for key,value in detail.items()
                        if key in ('min_alpha','contact_motion_ray_scales','contact_metric_ray_scales',
                                   'separate_subspace_ray_scales','contact_activation_ray_scales','iterations',
                                   'dynamic_soft_tangent_force_world','dynamic_soft_normal_load_sum',
                                   'dynamic_soft_coulomb_capacity_sum','final_maximum_surface_travel')})
                    if failure: break
            final_x=e.state_0.particle_q.numpy().astype(float)
            row = {'time_s':actual_time,'label':label,'substeps':substeps,'rounds_per_step':rounds,
                'query_motion_budget_m':motion_budget,'network_calls':calls[0]-first_calls,
                'seconds':time.perf_counter()-begin,'failure':failure,
                'material_center_shift_m':(final_x-initial_x).mean(0).tolist(),
                'material_max_shift_m':float(np.linalg.norm(final_x-initial_x,axis=-1).max()),
                'diagnostics':diagnostics}
            rows.append(row); budget.observe(force=True)
            print(json.dumps({k:v for k,v in row.items() if k!='diagnostics'}),flush=True)
            del solver,scene,e
    handle.remove()
    def serial(value):
        if torch.is_tensor(value):return value.detach().cpu().tolist()
        raise TypeError(type(value).__name__)
    report={'checkpoint':str(args.checkpoint),'recording':str(args.recording),'rows':rows,
            'w1_used':False,'optimizer_updates':0,'native_dynamics_calls':0,'gpu':budget.report(),
            'scope':'Matched initial state and held control over one display interval; diagnostic ablations only.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report,indent=2,default=serial,allow_nan=False)+'\n')


if __name__=='__main__': main()
