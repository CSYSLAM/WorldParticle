"""Qualified authored Franka substep data for offline learning/evaluation.

Rebuilding a dataset never constructs a dynamics solver. The standalone
inference entry does not import this module or read reference states.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from . import math3d as so3
from .data import ROOT, State
from .franka_vbd_scene import build
from .surface_joint_scene import export_model, CONTRACT
from .surface_joint_data import restore_joint_source
from .surface_joint_runtime import SurfaceJointCollisionOracle
from .newton_source import newton_revision

SCHEMA='authored_franka_substep_reference_v310'


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def newton_version():
    return newton_revision()


def rebuild(run, source, wp, newton, device):
    p=run['physical_parameters'];material=p['tet_material']
    e=build(wp,newton,device=device,initialize_controls=False,
            soft_asset=run['scene'].rsplit('_',1)[-1],material_mu=material[0],material_lambda=material[1],
            arm_drive_ke=p['arm_drive_ke'],grasp_x=p['grasp_x'],grasp_yaw=p['grasp_yaw'],
            grasp_height=p.get('grasp_height',.21),grip_open=p['grip_open'],grip_closed=p['grip_closed'],
            friction=p.get('friction_override'))
    # Initial joint coordinates belong to the recorded authored asset.
    # Recompute FK before verifying the complete frozen native source.
    e.model.joint_q.assign(source['source__joint_q'])
    scratch=e.model.state();newton.eval_fk(e.model,e.model.joint_q,e.model.joint_qd,scratch)
    wp.copy(e.model.body_q,scratch.body_q);e.model.joint_target_q.assign(e.model.joint_q)
    roundoff=restore_joint_source(e.model,source)
    for name in ('joint_mimic_coeffs','joint_target_mode'):
        if not np.array_equal(getattr(e.model,name).numpy(),source['source__'+name]):
            raise ValueError('Franka authored constraint/drive changed: '+name)
    for name in ('soft_contact_ke','soft_contact_kd','soft_contact_mu'):
        if getattr(e.model,name)!=p[name]:raise ValueError('Franka contact material changed: '+name)
    scene=export_model(e.model,run['solver_configuration'],run['scene'],newton,run['physical_dt'],device)
    scene.current_device_static_roundoff=roundoff
    return e,scene


def model_policy(scene):
    scene.metadata.update(detection_motion_budget=.003,bounded_motion_forecast_target_only=True,
        joint_tree_coarse_modes='scalar_tree_twelve_columns_v307',separate_coarse_ray_step=True,
        articulation_material_motion_modes=True,soft_tangent_support_scale=8.,force_aware_material_anticipation=True)


def tensors(scene, archive, device):
    raw=lambda key:torch.as_tensor(archive[key].copy(),device=device,dtype=torch.float64)
    p=raw('particle_position');pose=raw('body_pose');pv=raw('particle_velocity');bv=raw('body_velocity')
    count,n=p.shape[:2];b=pose.shape[1]
    x=scene.rest_x[None].expand(count,-1,-1).clone();q=scene.rest_q[None].expand(count,-1,-1).clone()
    v=torch.zeros_like(x);omega=torch.zeros_like(x)
    rotation=so3.normalize(pose[...,3:])
    x[:,:n]=p;x[:,n:n+b]=pose[...,:3]+so3.rotate(rotation,scene.body_com[n:n+b])
    q[:,n:n+b]=rotation;v[:,:n]=pv;v[:,n:n+b]=bv[...,:3];omega[:,n:n+b]=bv[...,3:]
    coordinates=scene.joint_coordinates(State(x,q,v,omega),raw('joint_position'))
    angle=torch.where(scene.joint_spec.angular_mode==1,coordinates.axis[...,3],0.)
    initialized=torch.ones(count,device=device,dtype=torch.bool);initialized[0]=False
    states={'x':x,'q':q,'v':v,'omega':omega,'history__pin_x_prev':x[:,scene.fixed_particle_ids].clone(),
            'history__pin_initialized':initialized,'history__joint_angle_prev':angle,
            'input_pin_x':x[:,scene.fixed_particle_ids].clone(),'input_force':torch.zeros_like(x),'input_torque':torch.zeros_like(x)}
    for key,raw_name,indices in [('input_joint_target','input_joint_target_raw',scene.joint_spec.target_index),
                                ('input_joint_target_rate','input_joint_rate_raw',scene.joint_spec.dof_index),
                                ('input_joint_effort','input_joint_effort_raw',scene.joint_spec.dof_index)]:
        a=raw(raw_name)
        if len(a)!=count-1:raise ValueError('One actual input is required for every physical transition')
        value=a[:,indices.clamp_min(0)]*(indices>=0)
        # Terminal held input is storage padding, never a supervised transition.
        states[key]=torch.cat((value,value[-1:]),0)
    return states


def prepare(recording, review_path, output, wp, newton, device='cpu'):
    recording=Path(recording);review_path=Path(review_path);output=Path(output)
    qualification=output.with_suffix('.qualification.json')
    if output.exists() or qualification.exists():raise FileExistsError(output)
    run=json.loads((recording/'run.json').read_text())
    review=json.loads(review_path.read_text())['cases'][recording.name]
    if not review['automatic_task_and_mesh_screen_passed'] or run['mode']!='offline_vbd_reference':
        raise ValueError('A separately executed, task/mesh-screened native reference is required')
    if review.get('run_sha256')!=digest(recording/'run.json') or review.get('trajectory_sha256')!=digest(recording/'trajectory.npz'):
        raise ValueError('Run or trajectory changed after its reference screen')
    audit=run.get('native_substep_audit')
    if not audit or not audit['enabled'] or audit['steps']!=run['physical_substeps'] or audit['nonpositive_tet_steps']:
        raise ValueError('Every native physical step must be audited before data export')
    if run.get('recording_stride_substeps')!=1 or not run.get('source_static_sha256'):
        raise ValueError('Actual per-substep states, velocities, inputs and source assets required')
    if run['ideal_mimic_count']!=1 or audit['max_mimic_coordinate_error']>1e-5:
        raise ValueError('The original scalar finger mimic must be retained accurately')
    if digest(recording/'source_static.npz')!=run['source_static_sha256']:raise ValueError('Source archive changed')
    builder_sha=run['implementation_sha256']['src/wp_vbd/franka_vbd_scene.py']
    if digest(ROOT/'src/wp_vbd/franka_vbd_scene.py')!=builder_sha:raise ValueError('Requalify after authored builder changes')
    with np.load(recording/'source_static.npz',allow_pickle=False) as z:source={k:z[k].copy() for k in z.files}
    with np.load(recording/'trajectory.npz',allow_pickle=False) as z:archive={k:z[k].copy() for k in z.files}
    length=len(archive['physical_time'])
    if length!=run['physical_substeps']+1 or not np.allclose(np.diff(archive['physical_time']),run['physical_dt'],rtol=0,atol=1e-12):
        raise ValueError('Complete constant-dt physical transitions required')
    if not np.array_equal(archive['input_joint_target_raw'],archive['joint_target'][1:]):
        raise ValueError('Recorded target causality changed')
    e,scene=rebuild(run,source,wp,newton,device)
    with torch.no_grad():states=tensors(scene,archive,device)
    if any(not bool(torch.isfinite(value).all()) for value in states.values()):raise ValueError('Nonfinite training state/input')
    angle=so3.log(so3.multiply(states['q'][1:],so3.conjugate(states['q'][:-1]))).norm(dim=-1)
    travel=(states['x'][1:]-states['x'][:-1]).norm(dim=-1)+scene.body_radius*angle
    maximum_travel=float(travel[:,scene.movable].max())
    if maximum_travel>.003:raise ValueError('Reference transition exceeds the declared model motion envelope')
    metadata={**scene.metadata,'training_schema':SCHEMA,'reference_final_check':'PASS',
              'source_run':run,'source_builder_sha256':builder_sha,'is_official_demo':False,
              'source_recording':str(recording.resolve()),'source_trajectory_sha256':digest(recording/'trajectory.npz'),
              'source_static_sha256':run['source_static_sha256'],'source_review_sha256':digest(review_path),
              'external_force_and_torque':'zero; model gravity and recorded native drive inputs remain explicit',
              'terminal_input':'Repeated held terminal value; no target transition exists after the last state.'}
    values={**source,**{k:v.detach().cpu().numpy() for k,v in scene.values.items()},
            **{k:v.detach().cpu().numpy() for k,v in states.items()}}
    for prefix,obj in [('joint__',scene.joint_spec),('mimic__',scene.mimic_spec)]:
        for key,value in vars(obj).items():values[prefix+key]=value.detach().cpu().numpy()
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,metadata=json.dumps(metadata,allow_nan=False),**values)
    report={'status':'QUALIFIED_FRANKA_PHYSICAL_SUBSTEP_DATA','output_sha256':digest(output),
            'newton_sha':newton_version(),
            'physical_steps':length-1,'dt':run['physical_dt'],'maximum_reference_surface_travel_m':maximum_travel,
            'complete_native_substep_audit':audit,'input_target_causality_verified':True,
            'source_static_verified':True,'original_mimic_retained':True,'training_admitted':True,
            'new_optimizer_updates':0,'scope':'One authored native-VBD grasp with an explicit end-of-iteration mimic policy. Data eligibility is not evidence of a training update or model task success.'}
    qualification.write_text(json.dumps(report,indent=2)+'\n')
    return report


def load_reference(path,wp,newton,device='cpu'):
    path=Path(path);proof=json.loads(path.with_suffix('.qualification.json').read_text())
    if proof['status']!='QUALIFIED_FRANKA_PHYSICAL_SUBSTEP_DATA' or proof['output_sha256']!=digest(path):
        raise ValueError('Exact qualified authored data required')
    if proof.get('newton_sha')!=newton_version():raise ValueError('Qualified Newton version changed')
    with np.load(path,allow_pickle=False) as z:
        metadata=json.loads(str(z['metadata']));values={k:z[k].copy() for k in z.files if k!='metadata'}
    if metadata['training_schema']!=SCHEMA or metadata['contract']!=CONTRACT:raise ValueError('Franka data schema changed')
    if metadata['source_builder_sha256']!=digest(ROOT/'src/wp_vbd/franka_vbd_scene.py'):raise ValueError('Authored builder changed')
    e,scene=rebuild(metadata['source_run'],values,wp,newton,device)
    for key,value in scene.values.items():
        saved=torch.as_tensor(values[key],device=device,dtype=value.dtype)
        same=torch.allclose(value,saved,atol=1e-12,rtol=1e-12) if value.is_floating_point() else torch.equal(value,saved)
        if not same:raise ValueError('Derived authored asset changed: '+key)
    for prefix,obj in [('joint__',scene.joint_spec),('mimic__',scene.mimic_spec)]:
        for key,value in vars(obj).items():
            saved=torch.as_tensor(values[prefix+key],device=device,dtype=value.dtype)
            same=torch.allclose(value,saved,atol=1e-12,rtol=1e-12) if value.is_floating_point() else torch.equal(value,saved)
            if not same:raise ValueError('Authored constraint changed: '+prefix+key)
    scene.metadata.update(metadata);model_policy(scene)
    scene.geometry_oracle=SurfaceJointCollisionOracle(e.model,scene,wp,newton)
    keys=('x','q','v','omega','history__pin_x_prev','history__pin_initialized','history__joint_angle_prev',
          'input_pin_x','input_force','input_torque','input_joint_target','input_joint_target_rate','input_joint_effort')
    states={key:torch.as_tensor(values[key],device=device) for key in keys}
    return scene,states
