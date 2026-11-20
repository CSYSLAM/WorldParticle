"""Check causal window wiring with a differentiable stub, not dynamics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    if args.report.exists():raise FileExistsError(args.report)
    import torch
    from wp_vbd import math3d as so3
    from wp_vbd.data import State
    from wp_vbd.caller_pose import RigidCallerPose
    from wp_vbd.rigid_feedback import RigidTargetFeedback
    from wp_vbd.rigid_caller_program import RigidCallerProgram,CallerMemory,HISTORY_KEYS,SCHEMA,attach_memory
    from wp_vbd.rod_caller_window import step
    torch.set_num_threads(1);torch.manual_seed(215);dtype=torch.float64
    n=4;com=.01*torch.randn(n,3,dtype=dtype);mass=torch.tensor([2.,.7,1.,0.],dtype=dtype)
    editor=RigidCallerPose(com,torch.tensor([2]),torch.tensor([3]),torch.tensor([[0.,0.,-.05]],dtype=dtype),
        torch.tensor([1]),torch.tensor([0]),torch.tensor([[.1,0.,0.]],dtype=dtype),torch.tensor([[0.,0.,0.,1.]],dtype=dtype))
    feedback=RigidTargetFeedback(com,mass,0,torch.tensor([1]),torch.tensor([0,1]))
    program=RigidCallerProgram(editor,feedback,2,torch.tensor([False,True,False]))
    x=torch.randn(1,n,3,dtype=dtype,requires_grad=True);q=so3.exp(.2*torch.randn_like(x))
    v=torch.ones_like(x)*1.5;w=torch.randn_like(x)*.1
    physical=State(x,q,v,w,{'raw_x':x.clone()})
    memory=CallerMemory(torch.randn(1,3,dtype=dtype),torch.tensor([1]),torch.tensor([False]))
    current=attach_memory(physical,memory);snapshot={k:a.clone() for k,a in current.history.items()}
    class Oracle:
        queries=0
        def query(self,state):self.queries+=1;self.last=state;return SimpleNamespace()
    oracle=Oracle();scene=SimpleNamespace(metadata={'rigid_caller_training_schema':SCHEMA,'dt':.01},
        caller_program=program,gravity=torch.tensor([0.,0.,-9.81],dtype=dtype),geometry_oracle=oracle)
    class Stub:
        config={'rod_caller_pose_history':True,'rod_prescribed_fresh_inputs':True}
        accepted=True
        calls=0
        def __call__(self,scene,state,dt,*,force,torque,learned=True):
            self.calls+=1;self.last_force=force
            assert not HISTORY_KEYS&state.history.keys()
            # Explicit algebraic stub to test data flow only; it is neither
            # the shared NN nor a physical integrator acceptance test.
            updated=state.x+dt*dt*force
            out=State(updated,state.q,state.v+dt*force,state.omega,{'raw_x':updated})
            return out,{'accepted':torch.tensor([self.accepted])}
    backend=Stub();canary=torch.full((3,n,3),float('nan'),dtype=dtype,requires_grad=True)
    states={'x':canary,'q':canary,'history__caller_target':canary,
        'input_caller_stiffness':torch.tensor([3.,3.],dtype=dtype),
        'input_caller_damping':torch.tensor([2.,2.],dtype=dtype),
        'input_caller_picked_body':torch.tensor([-1,-1])}
    first,d0=step(backend,scene,current,states,0,None)
    assert oracle.queries==1 and backend.calls==1 and d0['caller_target_refreshed']
    # Follower1 shares spring acceleration but its damping must use the
    # velocity before synchronization, while the backend sees zero velocity.
    origin=x[:,0]-so3.rotate(q[:,0],com[0]);acceleration=(10+mass[0])*3/mass[0]*(memory.target-origin)
    expected=mass[1]*acceleration-(10+mass[1])*2*v[:,1]-mass[1]*scene.gravity
    torch.testing.assert_close(backend.last_force[:,1],expected,rtol=1e-12,atol=1e-12)
    assert torch.count_nonzero(oracle.last.v[:,1])==0
    boundary_start=attach_memory(physical,CallerMemory(memory.target,torch.tensor([0]),memory.align))
    long,dl=step(backend,scene,boundary_start,states,0,None,dt=.02)
    assert dl['caller_target_refreshed'] and int(long.history['caller_phase'])==0
    calls=backend.calls
    try:step(backend,scene,current,states,0,None,dt=.02)
    except ValueError as error:assert 'sample boundary' in str(error)
    else:raise AssertionError('A long step skipped a caller refresh boundary')
    assert backend.calls==calls
    second,d1=step(backend,scene,first,states,1,None)
    assert oracle.queries==3 and backend.calls==3 and not d1['caller_target_refreshed']
    assert all(torch.equal(a,current.history[k]) for k,a in snapshot.items())
    objective=second.x.square().sum()+second.q.square().sum()
    gx,gfuture=torch.autograd.grad(objective,(x,canary),allow_unused=True)
    assert gfuture is None and torch.isfinite(gx).all() and gx.norm()>0
    # Neither failure nor repeated preparation may publish edited poses or
    # advance the externally carried target/phase/history.
    backend.accepted=False;rejected,dr=step(backend,scene,current,states,0,None)
    assert rejected is current and not dr['caller_target_refreshed']
    invalid=[]
    for key in ['input_force','input_torque','input_pin_x','input_pin_q']:
        try:step(backend,scene,current,{**states,key:canary},0,None)
        except ValueError:invalid.append(key)
        else:raise AssertionError(key)
    backend.config={**backend.config,'rod_caller_pose_history':False}
    try:step(backend,scene,current,states,0,None)
    except ValueError:invalid.append('missing_velocity_history_policy')
    else:raise AssertionError('Missing mandatory velocity policy')
    result={'status':'CALLER_WINDOW_CAUSAL_WIRING_UNIT_PASS','oracle_stub_calls':oracle.queries,'backend_stub_calls':backend.calls,
        'two_hop_prediction_gradient_norm':float(gx.norm()),'future_pose_target_canary_has_no_gradient_dependency':True,
        'feedback_before_follower_velocity_reset_verified':True,'physical_backend_receives_only_physical_history':True,
        'rejected_transaction_returns_original_state_object':True,'original_state_history_unchanged':True,'invalid_channels_rejected':invalid,
        'actual_nn_calls':0,'actual_geometry_queries':0,'new_optimizer_updates':0,'new_training_admissions':0,'full_vbd_prediction_calls':0,
        'sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),ROOT/'src/wp_vbd/rod_caller_window.py',ROOT/'src/wp_vbd/rigid_caller_program.py']},
        'scope':'Differentiable algebraic backend and oracle stubs test only production wrapper wiring, phase transaction and future-label independence. No neural prediction, mesh collision, qualified dataset, shared training or native-quality claim.'}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
