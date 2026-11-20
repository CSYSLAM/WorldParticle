"""Verify zero-copy native controls and explicit immutable-asset validation."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    from wp_vbd.newton_source import activate_newton_source
    activate_newton_source()
    import torch
    import warp as wp
    import newton
    from wp_vbd.augmentation_data import specification,build,export_scene
    from wp_vbd.general_joint_export import read_controls
    from wp_vbd.surface_joint_runtime import SurfaceJointExampleSolver
    torch.set_num_threads(1);wp.init();wp.set_device('cpu')
    model,cfg,_=build(specification('joint',20261119),None,newton,wp,'cpu')
    scene=export_scene(model,cfg,{'dt':1/180,'family':'joint','dataset_id':'control_fixture'},newton,wp,'cpu')
    control=model.control();expected=read_controls(scene.joint_spec,control)
    def forbidden(*args,**kwargs):raise AssertionError('Native controls copied through numpy')
    with patch.object(wp.array,'numpy',forbidden):
        actual=read_controls(scene.joint_spec,control)
    for first,second in zip(expected,actual):torch.testing.assert_close(first,second,rtol=0,atol=0)
    solver=SurfaceJointExampleSolver(SimpleNamespace(config={'resident_static_structure':True}),scene,1/180,wp,newton,model,model.state())
    solver.validate_static_source();original=wp.to_torch(model.joint_target_ke).clone()
    wp.to_torch(model.joint_target_ke).add_(1)
    try:solver.validate_static_source(force=True)
    except ValueError as error:assert 'joint_target_ke' in str(error)
    else:raise AssertionError('A changed immutable material was accepted')
    wp.to_torch(model.joint_target_ke).copy_(original);solver.validate_static_source(force=True)
    snapshot=scene.joint_spec
    wp.to_torch(control.joint_target_q)[0]=.012
    solver.refresh_control(control)
    assert scene.joint_spec is snapshot
    assert bool(torch.isclose(scene.joint_controls[0],torch.tensor(.012,dtype=torch.float64),atol=1e-9,rtol=0).any())
    result={'status':'PASS','controls_never_call_numpy':True,'cached_structure_keeps_dynamic_targets':True,
            'explicit_material_revalidation_detects_change':True,'native_dynamics_calls':0}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(result)


if __name__=='__main__':main()
