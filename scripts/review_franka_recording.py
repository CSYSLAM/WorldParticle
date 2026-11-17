"""Inspect actual saved grasp states without executing dynamics or assigning a pass threshold."""
import argparse
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import activate_newton_source, source_root, newton_revision
NEWTON_ROOT=source_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    activate_newton_source()
    if args.report.exists():
        raise FileExistsError(args.report)
    import numpy as np
    import torch
    import warp as wp
    import newton
    sys.path.append(str(NEWTON_ROOT/'.venv/lib/python3.12/site-packages'))
    from wp_vbd.franka_vbd_scene import build
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd.soft_surface_ports import native_state

    torch.set_num_threads(1)
    wp.config.kernel_cache_dir = str(ROOT/'.local/p0/warp-cache')
    wp.init(); wp.set_device('cpu')
    run = json.loads((args.recording/'run.json').read_text())
    recorded_revision=run.get('newton_source',{}).get('revision')
    if recorded_revision is not None and recorded_revision!=newton_revision():
        raise ValueError('Review with the recording Newton revision')
    bound = inspect.signature(newton.solvers.SolverVBD).bind(None)
    bound.apply_defaults()
    cfg = {k:v for k,v in bound.arguments.items() if k not in ('model', 'collision_pipeline')}
    def forbidden(*unused, **kwargs):
        raise AssertionError('Recorded geometry inspection executes no dynamics')
    for cls in (newton.solvers.SolverVBD, newton.solvers.SolverFeatherstone):
        cls.__init__ = forbidden; cls.step = forbidden
    parameters = run['physical_parameters']; material = parameters['tet_material']
    rigid_cloth=run['scene']=='research.franka_vbd_rigid_cloth'
    if rigid_cloth:
        from wp_vbd.franka_rigid_cloth_scene import build
        material=[None,None]
    e = build(wp, newton, device='cpu', initialize_controls=False,
              soft_asset=run['scene'].rsplit('_',1)[-1], material_mu=material[0],
              material_lambda=material[1], friction=parameters.get('friction_override'))
    scene = export_model(e.model, cfg, run['scene'], newton, run['physical_dt'], 'cpu')
    scene.metadata['detection_motion_budget'] = run['detection_motion_budget']
    solver = SurfaceJointExampleSolver(SimpleNamespace(config={}), scene, run['physical_dt'],
                                       wp, newton, e.model, e.state_0)
    with np.load(args.recording/'trajectory.npz') as archive:
        x = archive['particle_position'].astype(np.float64)
        poses = archive['body_pose']; times = archive['physical_time']
        phases = archive['phase']; tets = archive['tets']
    if not (np.isfinite(x).all() and np.isfinite(poses).all()):
        raise ValueError('Nonfinite recorded state')
    if len(x) != len(poses) or len(x) != len(times) or len(phases) != len(x)-1:
        raise ValueError('Recording states/timestamps/phases must have consistent lengths')
    radius = e.model.particle_radius.numpy().astype(np.float64)
    rows = []
    for i, (particle, body) in enumerate(zip(x, poses)):
        e.state_0.particle_q.assign(particle.astype(np.float32)); e.state_0.body_q.assign(body)
        state, com, _ = native_state(e.model, e.state_0, wp)
        contacts = solver.oracle.query(state)
        depth = contacts.soft.geometry(state, com)[-1][0]
        filter_fn = contacts.soft.mesh_force_filter
        eligible = filter_fn(state, com)[0] if filter_fn is not None else torch.ones_like(depth, dtype=torch.bool)
        dynamic = contacts.soft.ports.owners[:,0] < scene.world
        chosen = depth[eligible & dynamic]
        rigid = contacts.rigid.geometry(scene, state)[-1][0]
        rigid_overlap=rigid-contacts.rigid.physical_margin
        soft_overlap=depth-contacts.soft.shape_margin
        chosen_overlap=soft_overlap[eligible & dynamic]
        payload_contact=eligible & (contacts.soft.ports.owners[:,0]==e.model.particle_count+e.payload_id) if rigid_cloth else torch.zeros_like(eligible)
        rows.append({'frame':i, 'time':float(times[i]),
                     'robot_soft_contact_depth_m':max(0.,float(chosen.max())) if len(chosen) else 0.,
                     'rigid_contact_depth_m':max(0.,float(rigid.max())) if len(rigid) else 0.,
                     'rigid_surface_overlap_m':max(0.,float(rigid_overlap.max())) if len(rigid) else 0.,
                     'robot_soft_surface_overlap_m':max(0.,float(chosen_overlap.max())) if len(chosen_overlap) else 0.,
                     'material_surface_min_z':float((particle[:,2]-radius).min()),
                     'payload_cloth_contact_rows':int((payload_contact & (depth>0)).sum()),
                     'payload_cloth_surface_overlap_m':max(0.,float(soft_overlap[payload_contact].max())) if bool(payload_contact.any()) else 0.})
        if i%60==0:print(f'inspect {i}/{len(times)-1}',flush=True)
    if rigid_cloth:
        from itertools import product
        payload=poses[:,e.payload_id].astype(np.float64)
        corners=np.asarray(list(product((-.025,.025),repeat=3)))
        q=payload[:,None,3:];q=q/np.linalg.norm(q,axis=-1,keepdims=True)
        world=payload[:,None,:3]+corners+2*np.cross(q[...,:3],np.cross(q[...,:3],corners)+q[...,3:]*corners)
        minimum_z=world[...,2].min(1)
        flags=e.model.particle_flags.numpy();fixed=(flags & int(newton.ParticleFlags.ACTIVE))==0
        triangles=x[:,e.model.tri_indices.numpy()]
        area=np.linalg.norm(np.cross(triangles[:,:,1]-triangles[:,:,0],triangles[:,:,2]-triangles[:,:,0]),axis=-1)
        area_ratio=area/area[:1]
        patch=np.linalg.norm(x[0,:,:2]-np.array([.14,-.5]),axis=-1)<.05
        pre=int(np.argmin(np.abs(times-2.5)))
        center_height=x[:,patch,2].mean(1)
        # Interpolate the actual cloth under the final rigid COM.
        surface=[]
        for tri in triangles[-1]:
            matrix=(tri[1:,:2]-tri[0,:2]).T
            uv=np.linalg.solve(matrix,payload[-1,:2]-tri[0,:2])
            if uv.min()>=-1e-7 and uv.sum()<=1+1e-7:
                surface.append(float(tri[0,2]+uv@(tri[1:,2]-tri[0,2])))
        with np.load(args.recording/'trajectory.npz') as z:
            opening=z['joint_position'][-1,7:9].tolist()
            final_velocity=z['body_velocity'][-1,e.payload_id].tolist()
        result={'status':'COMPLETE_RECORDED_GEOMETRY_INSPECTION','recording':str(args.recording),
            'frames_checked':len(rows),'physical_seconds':float(times[-1]),
            'payload_initial_position':payload[0,:3].tolist(),'payload_final_position':payload[-1,:3].tolist(),
            'payload_max_lift_m':float(payload[:,2].max()-payload[0,2]),
            'payload_transfer_m':float(payload[-1,0]-payload[0,0]),
            'final_payload_min_z_m':float(minimum_z[-1]),
            'final_cloth_height_under_payload_com_m':max(surface) if surface else None,
            'final_payload_min_z_minus_cloth_center_m':float(minimum_z[-1]-max(surface)) if surface else None,
            'final_payload_velocity':final_velocity,'final_finger_opening_coordinates_m':opening,
            'max_pinned_vertex_error_m':float(np.linalg.norm(x[:,fixed]-x[:1,fixed],axis=-1).max()),
            'cloth_center_height_before_load_m':float(center_height[pre]),
            'cloth_center_height_final_m':float(center_height[-1]),
            'additional_cloth_center_deflection_m':float(center_height[pre]-center_height[-1]),
            'cloth_triangle_area_ratio_min':float(area_ratio.min()),'cloth_triangle_area_ratio_max':float(area_ratio.max()),
            'max_rigid_surface_overlap_m':max(r['rigid_surface_overlap_m'] for r in rows),
            'max_dynamic_cloth_surface_overlap_m':max(r['robot_soft_surface_overlap_m'] for r in rows),
            'max_payload_cloth_surface_overlap_m':max(r['payload_cloth_surface_overlap_m'] for r in rows),
            'final_payload_cloth_contact_rows':rows[-1]['payload_cloth_contact_rows'],
            'full_vbd_calls':0,'featherstone_calls':0,
            'scope':'Native geometry at every saved state, with contact eligibility and physical margins. Discrete inspection; no continuous collision proof or automatic human visual acceptance.',
            'samples':[rows[i] for i in sorted(set(range(0,len(rows),30)) | {len(rows)-1})]}
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(result,indent=2)+'\n');print(args.report.read_text());return
    vertices = x[:,tets]
    matrix = vertices[:,:,1:]-vertices[:,:,:1]
    ratio = np.linalg.det(matrix)/np.linalg.det(matrix[:1])
    centers = x.mean(1)
    hold = np.flatnonzero(phases == 'hold')
    start_hold = int(hold[0]) if len(hold) else None
    samples = sorted(set(range(0, len(rows), 30)) | {len(rows)-1})
    result = {'status':'COMPLETE_RECORDED_GEOMETRY_INSPECTION',
              'recording':str(args.recording), 'frames_checked':len(rows),
              'max_robot_soft_contact_depth_m':max(r['robot_soft_contact_depth_m'] for r in rows),
              'max_rigid_contact_depth_m':max(r['rigid_contact_depth_m'] for r in rows),
              'max_rigid_surface_overlap_m':max(r['rigid_surface_overlap_m'] for r in rows),
              'max_robot_soft_surface_overlap_m':max(r['robot_soft_surface_overlap_m'] for r in rows),
              'min_tet_volume_ratio_over_recording':float(ratio.min()),
              'max_tet_volume_ratio_over_recording':float(ratio.max()),
              'minimum_material_surface_z':min(r['material_surface_min_z'] for r in rows),
              'table_top_z':.2,
              'hold_seconds':float(times[-1]-times[start_hold]) if start_hold is not None else 0.,
              'hold_com_vertical_change_m':float(centers[-1,2]-centers[start_hold,2]) if start_hold is not None else None,
              'final_material_surface_height_above_table_m':rows[-1]['material_surface_min_z']-.2,
              'full_vbd_calls':0, 'featherstone_calls':0,
              'scope':'Native contact candidates and mesh force eligibility at every saved state; includes contact radii/margins. Discrete geometry diagnostic, not a continuous collision proof. Numerical diagnostics do not assign human visual acceptance.',
              'samples':[rows[i] for i in samples]}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2)+'\n')
    print(args.report.read_text())


if __name__ == '__main__':
    main()
