"""Check bounded tree columns on real Franka frames and a shared physical metric."""
import argparse
import inspect
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','cuda:0'),default='cpu')
    parser.add_argument('--compiled-response',action='store_true')
    parser.add_argument('--spatial-products',action='store_true');a=parser.parse_args()
    from wp_vbd.newton_source import activate_newton_source,source_root
    activate_newton_source()
    if a.report.exists():raise FileExistsError(a.report)
    from wp_vbd.gpu_ownership import claim_cuda
    ownership=claim_cuda(a.device,ROOT)
    from wp_vbd.compiled_contact import small_spatial_products
    with small_spatial_products(a.spatial_products):verify(a)


def verify(a):
    from wp_vbd.newton_source import source_root
    import torch
    import warp as wp
    import newton
    sys.path.append(str(source_root()/'.venv/lib/python3.12/site-packages'))
    from wp_vbd.franka_vbd_scene import build,controller
    from wp_vbd.surface_joint_scene import export_model
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    from wp_vbd import articulation_coarse as tree,surface_joint_system as physics,rigid_contact_system as rigid,math3d as so3
    from wp_vbd.protected_response import ProtectedCoarse
    from wp_vbd.mixed_physics import predictor
    from wp_vbd.data import State
    from wp_vbd.mimic_constraint import rows as mimic_rows
    torch.set_num_threads(1);torch.manual_seed(307)
    torch.set_grad_enabled(False)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(a.device)
    if a.device.startswith('cuda'):wp.set_stream(wp.stream_from_torch(torch.cuda.current_stream()))
    bound=inspect.signature(newton.solvers.SolverVBD).bind(None);bound.apply_defaults()
    configuration={k:v for k,v in bound.arguments.items() if k not in ('model','collision_pipeline')}
    def forbidden(*unused,**kwargs):raise AssertionError('No dynamics solver in tree-mode verification')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    e=build(wp,newton,device=a.device,start_time=1.95,material_mu=1e6,material_lambda=1e6,arm_drive_ke=20000.)
    controller(e,3.3,wp)
    scene=export_model(e.model,configuration,'research.franka_vbd_block',newton,e.sim_dt,a.device)
    scene.compiled_response_kernels=a.compiled_response
    scene.metadata['detection_motion_budget']=.003
    solver=SurfaceJointExampleSolver(SimpleNamespace(config={}),scene,e.sim_dt,wp,newton,e.model,e.state_0)
    solver.refresh_control(e.control);start=solver.read(e.state_0)
    scene.candidates=solver.oracle.query(start);scene.candidate_kind=scene.candidates.relation_kind;scene.geometry_oracle=solver.oracle
    context=physics.begin(scene,start,scene.candidates,e.sim_dt)
    tx,tq=predictor(scene,start,e.sim_dt)
    residual,metric,result=physics.system(scene,start,start,tx,tq,context)
    scene.update_graph(scene.candidates.owners,result.get('topology_active',result['active'])[0])
    modes=tree.raw_modes(scene,start)
    spec=scene.joint_spec;coordinates=scene.joint_coordinates(start,solver.history['joint_angle_prev'])
    structural=torch.einsum('bjcsi,bjsip->bjcp',coordinates.structural_jacobian,modes[:,spec.owners])
    structural_error=float(structural.abs().max())
    assert structural_error<1e-5,structural_error
    mimic=mimic_rows(scene,start)
    derivative=torch.einsum('brsi,brsip->brp',mimic.jacobian,modes[:,mimic.owners])
    mimic_error=float(derivative.abs().max());assert mimic_error<1e-10,mimic_error
    assert not bool(modes[:,:e.model.particle_count].any()) and not bool(modes[:,scene.world].any())
    q=so3.exp(start.x.new_tensor([.23,-.18,.31]));shift=start.x.new_tensor([1.1,-.7,.4])
    transformed=State(so3.rotate(q,start.x)+shift,so3.multiply(q,start.q),start.v,start.omega,start.history)
    rotated=torch.cat((so3.rotate(q,modes[...,:3,:].transpose(-1,-2)).transpose(-1,-2),
                       so3.rotate(q,modes[...,3:,:].transpose(-1,-2)).transpose(-1,-2)),-2)
    # Native URDF frame quaternions and the existing mimic covectors retain
    # FP32 source rounding. Check covariance at that interface precision.
    covariance=float((tree.raw_modes(scene,transformed)-rotated).abs().max());assert covariance<1e-6,covariance
    base=rigid.coarse(scene,start);augmented=tree.augment(scene,start,base,metric,residual)
    assert max(z.shape[-1] for _,z,*_ in augmented.blocks)<=12
    old=ProtectedCoarse(scene,metric,base.blocks);new=ProtectedCoarse(scene,metric,augmented.blocks)
    assert bool(old.valid.all() & new.valid.all())
    before=float((residual*old(residual)).sum());after=float((residual*new(residual)).sum())
    assert after>=before-1e-8*max(1.,before),(before,after)
    u=torch.randn_like(residual)*scene.dof;v=torch.randn_like(residual)*scene.dof
    symmetry=abs(float((u*new(v)).sum()-(v*new(u)).sum()))
    assert symmetry<1e-8 and float((u*new(u)).sum())>=-1e-12
    scene.metadata['articulation_material_motion_modes']=True
    material=tree.augment(scene,start,base,metric,residual)
    material_response=ProtectedCoarse(scene,metric,material.blocks)
    assert bool(material_response.valid.all())
    assert all(z.shape[-1]<=12 for _,z,*_ in material.blocks)
    material_selection=scene.articulation_coarse_diagnostic
    assert any(row['base_space']=='material_rigid' for row in material_selection['selected'])
    assert not bool(material_response(torch.zeros_like(residual)).any())
    # Negative ownership cases must not create arbitrary motion columns.
    owners=scene.joint_spec.owners.clone();scene.joint_spec.owners[0,0]=scene.joint_spec.owners[-1,1]
    cyclic_rejected=False
    try:tree.topology(scene)
    except ValueError:cyclic_rejected=True
    finally:scene.joint_spec.owners=owners
    assert cyclic_rejected
    result={'status':'PASS','device':a.device,'small_spatial_products':a.spatial_products,'compiled_response':a.compiled_response,'independent_tree_columns':modes.shape[-1],'maximum_component_columns':12,
            'structural_coordinate_directional_error':structural_error,'mimic_directional_error':mimic_error,
            'rigid_gauge_covariance_error':covariance,'coarse_response_symmetry_error':symmetry,
            'old_projected_residual_work':before,'new_projected_residual_work':after,
            'selection':scene.articulation_coarse_diagnostic,'full_vbd_calls':0,'featherstone_calls':0,
            'material_motion_selection':material_selection,'material_zero_residual_root_preserved':True,
            'cyclic_joint_tree_rejected':True,
            'new_optimizer_updates':0,'new_network_parameters':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(a.report.read_text())


if __name__=='__main__':main()
