"""Separate measured gripper-relative creep from lift and material deformation.

No dynamics or optimization. The final reported Coulomb ratio is a diagnostic
for saved contact rows, not a feasible six-dimensional grasp wrench bound.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings',nargs='+',type=Path,required=True)
    parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args()
    import numpy as np
    from scipy.spatial.transform import Rotation
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.franka_vbd_scene import build
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache')
    wp.init();wp.set_device('cpu')
    def forbidden(*unused,**options):raise AssertionError('Saved-state audit must not execute dynamics')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=forbidden;cls.step=forbidden
    cases=[]
    for recording in args.recordings:
        run=json.loads((recording/'run.json').read_text())
        if run['scene']!='research.franka_vbd_block' or run['mode']!='shared_network_inference':
            raise ValueError('An independently recorded Franka/soft-block model rollout is required')
        if run['newton_source']['revision']!=newton_revision():raise ValueError('Use the recorded Newton source')
        if run['teacher_trajectory_reads'] or any(run['forbidden_solver_calls'].values()):
            raise ValueError('Inference record contains teacher reads or native dynamics')
        p=run['physical_parameters'];e=build(wp,newton,device='cpu',initialize_controls=False,
            material_mu=p['tet_material'][0],material_lambda=p['tet_material'][1],
            arm_drive_ke=p['arm_drive_ke'],friction=p.get('friction_override'))
        # The pinned URDF parser merges the fixed hand into fr3_link7.
        # Use that actual rigid pose, never the commanded tool coordinates.
        names=list(e.model.body_label)
        hand_ids=[i for i,name in enumerate(names) if name.endswith('/fr3_hand') or name.endswith('fr3_hand')]
        if not hand_ids:hand_ids=[i for i,name in enumerate(names) if name.endswith('/fr3_link7')]
        if len(hand_ids)!=1:raise ValueError('Unique physical fr3_hand body required: '+repr(names))
        hand_id=hand_ids[0]
        mass=e.model.particle_mass.numpy().astype(np.float64);weight=mass.sum()*9.81
        with np.load(recording/'trajectory.npz') as archive:
            x=archive['particle_position'].astype(np.float64);poses=archive['body_pose'].astype(np.float64)
            times=archive['physical_time'].astype(np.float64);rest=archive['particle_rest'].astype(np.float64)
            tets=archive['tets'];vel=archive['particle_velocity'].astype(np.float64)
        center=(x*mass[None,:,None]).sum(1)/mass.sum()
        from wp_vbd.franka_lift_metrics import table_height,whole_object_lift
        lift=whole_object_lift(x,times,table_height(e.model,newton))
        hand=poses[:,hand_id];rotation=Rotation.from_quat(hand[:,3:])
        relative=rotation.inv().apply(center-hand[:,:3])
        rows=[]
        global_times=times+run.get('initial_command_time',0.)
        release=run.get('teacher_release')
        hold_end=min(4.,release['start_time_s']) if release else 4.
        hold_mask=(global_times>=3.3-1e-9)&(global_times<=hold_end+1e-9)
        hold_coverage=bool(global_times[0]<=3.3+1e-9 and global_times[-1]>=hold_end-1e-9 and hold_mask.sum()>=2)
        hold_lift=whole_object_lift(x[hold_mask],times[hold_mask],table_height(e.model,newton)) if hold_coverage else None
        if hold_lift is not None:
            hold_lift['minimum_table_clearance_during_hold_m']=float(x[hold_mask,:,2].min()-table_height(e.model,newton))
            hold_lift['all_saved_hold_states_clear_by_5mm']=bool((x[hold_mask,:,2].min(1)-table_height(e.model,newton)>.005).all())
        release_result=None
        if release and global_times[-1]>=release['start_time_s']-1e-9:
            opening=int(np.abs(global_times-release['start_time_s']).argmin())
            release_result={'command_time_s':release['start_time_s'],
                'COM_drop_until_end_m':float(center[opening,2]-center[-1,2]),
                'final_lowest_vertex_table_clearance_m':float(x[-1,:,2].min()-table_height(e.model,newton)),
                'final_active_robot_material_rows':run['last_diagnostic'].get('active_dynamic_particle_body_rows')}
        intervals=[('lift',2.1,3.3),('hold',3.3,hold_end)]
        if release:
            intervals.append(('opening_and_release',release['start_time_s'],
                min(float(global_times[-1]),release['start_time_s']+release['duration_s']+.35)))
        for phase,first,last in intervals:
            if last<=first:continue
            a,b=(int(np.abs(global_times-value).argmin()) for value in (first,last))
            if abs(global_times[a]-first)>run['frame_dt']*.51 or abs(global_times[b]-last)>run['frame_dt']*.51:
                continue
            gravity_in_hand=rotation[a].inv().apply([0.,0.,-1.])
            creep=float((relative[b]-relative[a])@gravity_in_hand)
            rows.append({'phase':phase,'start_s':float(global_times[a]),'end_s':float(global_times[b]),
                         'downward_com_motion_relative_to_hand_m':creep,
                         'mean_relative_creep_m_s':creep/(times[b]-times[a]),
                         'world_com_vertical_change_m':float(center[b,2]-center[a,2])})
        origin=rest-(rest*mass[:,None]).sum(0)/mass.sum();rms=[]
        for points in x:
            centered=points-(points*mass[:,None]).sum(0)/mass.sum()
            u,_,vt=np.linalg.svd((origin*mass[:,None]).T@centered)
            frame=u@np.diag([1.,1.,np.linalg.det(u@vt)])@vt
            displacement=centered-origin@frame
            rms.append(float(np.sqrt(((displacement**2).sum(-1)*mass).sum()/mass.sum())))
        shape=x[:,tets];rest_shape=rest[tets]
        volume=np.linalg.det((shape[:,:,1:]-shape[:,:,:1])@np.linalg.inv(rest_shape[:,1:]-rest_shape[:,:1]))
        diagnostic=run['last_diagnostic']
        cases.append({'recording':str(recording.resolve()),'hand_body_id':hand_id,
            'integration_completed':run['status']=='COMPLETE',
            'peak_com_lift_m':float(center[:,2].max()-center[0,2]),
            'peak_hand_body_lift_m':float(hand[:,2].max()-hand[0,2]),
            'object_lift_above_2cm_observed':bool(center[:,2].max()-center[0,2]>.02),
            'whole_object_lift':lift,
            'whole_object_lift_during_commanded_hold':hold_lift,
            'release_result':release_result,
            'physical_dt_s':run['physical_dt'],'material_mass_kg':float(mass.sum()),'weight_N':float(weight),
            'friction_coefficient':p['soft_contact_mu'],'slip_intervals':rows,
            'peak_nonrigid_mass_rms_m':max(rms),'minimum_tet_volume_ratio':float(volume.min()),
            'maximum_tet_volume_ratio':float(volume.max()),
            'final_coulomb_row_max_utilization':diagnostic.get('maximum_dynamic_soft_coulomb_ratio'),
            'final_reported_dynamic_normal_load_N':diagnostic.get('dynamic_soft_normal_load_sum'),
            'final_reported_dynamic_tangent_force_world_N':diagnostic.get('dynamic_soft_tangent_force_world'),
            'final_reported_dynamic_coulomb_capacity_N':diagnostic.get('dynamic_soft_coulomb_capacity_sum'),
            'force_diagnostic_scope':'Final saved contact rows only. Capacity sums are not a feasible grasp wrench or force-balance proof.',
            'particle_velocity_finite':bool(np.isfinite(vel).all())})
    result={'cases':cases,'native_dynamics_calls':0,'optimizer_updates':0,
        'scope':'Measured COM motion in actual hand coordinates includes sliding and deformation; local contact tracking is needed to separate them exactly. Opening/release motion is intentional and excluded from the hold interval.'}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
