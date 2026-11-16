"""Changing IK drive targets must not mutate rest geometry or actual states."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    if a.report.exists():raise FileExistsError(a.report)
    import numpy as np
    import torch
    import warp as wp
    import newton
    sys.path.append('/home/oem/code/engine/newton/.venv/lib/python3.12/site-packages')
    from wp_vbd.franka_vbd_scene import build,controller
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    def forbidden(*unused,**kwargs):raise AssertionError('A target controller must not construct or execute a dynamics solver')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    e=build(wp,newton,device='cpu')
    original={name:getattr(e.model,name).numpy().copy() for name in ('joint_q','body_q','particle_q')}
    actual={name:getattr(e.state_0,name).numpy().copy() for name in ('joint_q','body_q','particle_q')}
    targets=[]
    for t in (0.,.7,1.95,3.5):
        controller(e,t,wp)
        targets.append(e.control.joint_target_q.numpy().copy())
        assert all(np.array_equal(value,getattr(e.model,name).numpy()) for name,value in original.items())
        assert all(np.array_equal(value,getattr(e.state_0,name).numpy()) for name,value in actual.items())
    assert not np.array_equal(targets[0],targets[1]) and not np.array_equal(targets[1],targets[2])
    assert e.helper.ik_joint_q.ptr!=e.model.joint_q.ptr
    a.report.parent.mkdir(parents=True,exist_ok=True)
    a.report.write_text(json.dumps({'status':'PASS','control_times':[0.,.7,1.95,3.5],
        'changing_targets':True,'model_rest_and_actual_states_bitexact':True,'independent_ik_coordinate_storage':True,
        'full_vbd_calls':0,'featherstone_calls':0},indent=2)+'\n')
    print(a.report.read_text())


if __name__=='__main__':main()
