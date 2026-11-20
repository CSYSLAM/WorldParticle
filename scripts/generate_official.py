"""Record physical substeps through original Example.step and its controls.

One isolated process per authored scene/reference budget avoids module-level
solver aliases and persistent solver caches leaking between reference runs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import source_root,activate_newton_source
NEWTON=source_root()
sys.path.insert(0,str(NEWTON))
activate_newton_source()


def worker(args):
    import numpy as np
    import warp as wp
    import newton
    from wp_vbd.official import make_example
    from wp_vbd.official_adapter import export_official,snapshot
    if args.output.exists():raise FileExistsError('Keep existing reference evidence immutable: '+str(args.output))
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device(args.device)
    example,solvers,arguments=make_example(args.demo,iterations=args.iterations)
    if len(solvers)!=1:raise ValueError('A single VBD instance is required')
    solver,configuration=solvers[0]
    configuration={k:v if isinstance(v,(str,float,int,bool,type(None))) else str(v) for k,v in configuration.items()}
    model=example.model
    # Parameter variants keep the exact asset and authored h. Baseline factor
    # 1.0 is bit-for-bit authored; entire variant episodes are held out.
    for name in ('tet_materials','tri_materials','edge_bending_properties'):
        array=getattr(model,name,None)
        if array is not None and array.size:
            value=array.numpy();value[:,:min(3,value.shape[1])]*=args.factor
            array.assign(value)
    if args.demo=='basic.example_basic_dzhanibekov' and args.factor!=1:
        for state in (example.state_0,example.state_1):
            value=state.body_qd.numpy();value[:,3:]*=args.factor;state.body_qd.assign(value)
    static,metadata=export_official(example,configuration,args.demo,newton)
    metadata.update(index=0,arguments=arguments,parameter_factor=args.factor,source_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=NEWTON,text=True).strip())
    trajectory=[snapshot(static,model,example.state_0)];original=solver.step
    max_contact_count=0;max_force=0.;max_control_jump=0.;commands_x=[];commands_q=[]
    def record(state_in,state_out,control,contacts,dt):
        nonlocal max_contact_count,max_force,max_control_jump
        if dt!=example.sim_dt:raise ValueError('The authored substep changed')
        before=snapshot(static,model,state_in)
        jump=max(float(np.max(np.abs(before[key]-trajectory[-1][key]))) for key in before)
        max_control_jump=max(max_control_jump,jump)
        if 'kinematic_ids' in static:
            ids=static['kinematic_ids'];mobile=static['movable']
            if max(float(np.max(np.abs(before[key][mobile]-trajectory[-1][key][mobile]))) for key in before)>1e-6:raise ValueError('A controller changed dynamic state outside the kinematic input contract')
            commands_x.append(before['x'][ids].copy());commands_q.append(before['q'][ids].copy())
        elif jump>1e-6:raise ValueError('Prescribed state control needs the kinematic input adapter before admission')
        for name in ('particle_f','body_f'):
            array=getattr(state_in,name,None)
            if array is not None and array.size:max_force=max(max_force,float(np.abs(array.numpy()).max()))
        if max_force>1e-12:raise ValueError('Nonzero external force requires the force input recording adapter')
        if contacts is not None:
            for name in ('soft_contact_count','rigid_contact_count'):
                array=getattr(contacts,name,None)
                if array is not None:max_contact_count=max(max_contact_count,int(array.numpy()[0]))
        # Collision buffers contain candidates out to the detection gap.
        # Only a compressive contact activates this missing ALM operator.
        owners=static['site_owner']
        if model.body_count and len(owners) and np.any(before['x'][owners,2]-static['plane_height']-static['site_margin']<=0):
            raise ValueError('Active ALM contacts require their persistent state operator')
        original(state_in,state_out,control,contacts,dt)
        trajectory.append(snapshot(static,model,state_out))
    capture_buffers=None
    if args.capture_substeps:
        if args.demo not in ('softbody.example_softbody_hanging','vbd.example_cloth_stiff_material_hanging') or model.body_count or configuration['particle_enable_self_contact']:
            raise ValueError('Captured recorder currently admits only the two controller-free particle material examples')
        n=model.particle_count;substeps=example.sim_substeps
        capture_buffers={key:wp.zeros(n*substeps,dtype=wp.vec3,device=args.device) for key in ('x','v','input_x','input_v','force')}
        count_buffer=wp.zeros(substeps,dtype=wp.int32,device=args.device)
        cursor=0
        def capture_record(state_in,state_out,control,contacts,dt):
            nonlocal cursor
            if dt!=example.sim_dt or cursor>=substeps:raise ValueError('Unexpected captured substep schedule')
            for key,array in (('input_x',state_in.particle_q),('input_v',state_in.particle_qd),('force',state_in.particle_f)):
                wp.copy(capture_buffers[key],array,dest_offset=cursor*n,count=n)
            original(state_in,state_out,control,contacts,dt)
            wp.copy(capture_buffers['x'],state_out.particle_q,dest_offset=cursor*n,count=n)
            wp.copy(capture_buffers['v'],state_out.particle_qd,dest_offset=cursor*n,count=n)
            if contacts is not None:wp.copy(count_buffer,contacts.soft_contact_count,dest_offset=cursor,count=1)
            cursor+=1
        solver.step=capture_record
        example.capture()
        if cursor!=substeps or example.graph is None:raise ValueError('Authored frame capture did not record every substep')
    else:solver.step=record
    begin=time.perf_counter()
    for frame in range(args.frames):
        example.step()
        if capture_buffers is not None:
            values={key:buffer.numpy().reshape(example.sim_substeps,model.particle_count,3).copy() for key,buffer in capture_buffers.items()}
            max_force=max(max_force,float(np.abs(values['force']).max()))
            if max_force>1e-12:raise ValueError('Captured external force needs a declared input adapter')
            max_contact_count=max(max_contact_count,int(count_buffer.numpy().max()))
            for offset in range(example.sim_substeps):
                before=trajectory[-1]
                jump=max(float(np.abs(values['input_x'][offset]-before['x']).max()),float(np.abs(values['input_v'][offset]-before['v']).max()))
                max_control_jump=max(max_control_jump,jump)
                if jump>1e-6:raise ValueError('Captured controller changed state outside the admitted material contract')
                trajectory.append({'x':values['x'][offset],'v':values['v'][offset],'q':before['q'].copy(),'omega':before['omega'].copy()})
        if (frame+1)%60==0:print(f'{args.demo} factor={args.factor} budget={args.iterations}: {frame+1}/{args.frames}',flush=True)
    states={key:np.stack([s[key] for s in trajectory]) for key in trajectory[0]}
    if commands_x:states.update(input_x=np.stack(commands_x),input_q=np.stack(commands_q))
    if not all(np.isfinite(value).all() for value in states.values()):raise ValueError('Nonfinite teacher state')
    final_check='NOT_REQUESTED'
    if args.report:
        try:
            example.test_final();final_check='PASS'
        except Exception as error:final_check='FAIL: '+str(error)
    metadata.update(teacher_iterations=solver.iterations,teacher_frames=args.frames,teacher_substeps=len(trajectory)-1,max_contact_count=max_contact_count,max_external_force=max_force,max_control_jump=max_control_jump,teacher_seconds=time.perf_counter()-begin,teacher_device=args.device,reference_final_check=final_check,substep_recording='authored_frame_graph_with_per_substep_copies' if capture_buffers is not None else 'eager_original_step',collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,**static,**states,metadata=np.asarray(json.dumps(metadata)))
    if args.report:
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps({'demo':args.demo,'factor':args.factor,'frames':args.frames,'physical_substeps':len(trajectory)-1,'iterations':solver.iterations,'device':args.device,'terminal_check':final_check,'seconds':metadata['teacher_seconds'],'data_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'training_admission_granted':False},indent=2)+'\n')
    print('saved',args.output,len(trajectory)-1,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo',required=True);parser.add_argument('--factor',type=float,default=1.)
    parser.add_argument('--frames',type=int,default=120);parser.add_argument('--iterations',type=int)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--report',type=Path,help='Also run the authored final assertion and record the result')
    parser.add_argument('--capture-substeps',action='store_true',help='Offline particle reference optimization; keep every physical-substep output in the authored frame graph')
    args=parser.parse_args();worker(args)

if __name__=='__main__':main()
