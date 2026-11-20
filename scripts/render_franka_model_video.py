"""Render actual Franka states with explicit model/reference labeling."""
import argparse
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from wp_vbd.newton_source import activate_newton_source, source_root, newton_revision
NEWTON_ROOT=source_root()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--snapshot-only',action='store_true')
    p.add_argument('--reference-recording',action='store_true',help='Explicitly label a separately computed offline VBD recording')
    a=p.parse_args()
    activate_newton_source()
    if a.output.exists():raise FileExistsError(a.output)
    run=json.loads((a.recording/'run.json').read_text())
    recorded_revision=run.get('newton_source',{}).get('revision')
    if recorded_revision is not None and recorded_revision!=newton_revision():
        raise ValueError('Render with the recording Newton revision')
    reference=run['mode']=='offline_vbd_reference'
    if reference!=a.reference_recording:
        raise ValueError('Offline VBD recordings require --reference-recording; model recordings must omit it')
    if run['mode'] not in ('shared_network_inference','offline_vbd_reference') or any(run['forbidden_solver_calls'].values()) or run['teacher_trajectory_reads']:
        raise ValueError('An independent model recording is required')
    if reference and (run['nn_backbone_calls'] or run['checkpoint_step'] is not None or run['reference_vbd_steps']<=0):
        raise ValueError('A separately computed native VBD reference is required')
    os.environ.update(CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',__GLX_VENDOR_LIBRARY_NAME='mesa')
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    sys.path.append(str(NEWTON_ROOT/'.venv/lib/python3.12/site-packages'))
    from wp_vbd.franka_vbd_scene import build
    rigid_cloth=run['scene']=='research.franka_vbd_rigid_cloth'
    if rigid_cloth:
        from wp_vbd.franka_rigid_cloth_scene import build
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    parameters=run.get('physical_parameters',{});material=parameters.get('tet_material') or [None,None]
    e=build(wp,newton,device='cpu',soft_asset=run['scene'].rsplit('_',1)[-1],initialize_controls=False,
            material_mu=material[0],material_lambda=material[1],arm_drive_ke=parameters.get('arm_drive_ke',2000.),
            friction=parameters.get('friction_override'))
    if rigid_cloth:
        # Display-only fabric checks make the recorded bending visible.
        # Geometry, contacts and all recorded states remain untouched.
        palette=[(.12,.50,.80) if ((i//2)%8//2+(i//2)//8//2)%2 else (.38,.72,.91)
                 for i in range(e.model.tri_count)]
        e.model.tri_color.assign(np.asarray(palette,dtype=np.float32))
    def forbidden(*unused,**kwargs):raise AssertionError('Rendering must not execute dynamics')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=forbidden;cls.step=forbidden
    with np.load(a.recording/'trajectory.npz') as z:
        particle=z['particle_position'];body=z['body_pose'];times=z['physical_time'];faces=z['faces']
    peak_lift=float((particle.mean(1)[:,2]-particle[0].mean(0)[2]).max())
    from wp_vbd.franka_lift_metrics import table_height,whole_object_lift
    lift=whole_object_lift(particle,times,table_height(e.model,newton)) if not rigid_cloth else None
    assert np.array_equal(faces,e.model.tri_indices.numpy())
    viewer=newton.viewer.ViewerGL(width=944,height=748,vsync=False,headless=True,enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE)
    viewer.set_model(e.model);viewer.show_ui=False;viewer.show_particles=False;viewer.show_joints=False;viewer.show_contacts=False
    viewer.renderer.draw_sky=False;viewer.renderer.draw_fps=False;viewer.renderer.draw_edges=False
    viewer.renderer.background_color=(.90,.93,.97)
    viewer.renderer.spotlight_enabled=False
    camera=[((.70,-1.25,.86),(-.22,-.50,.30),'Franka / offline VBD reference' if reference else 'Franka / articulated response' if run.get('articulation_modes') else 'Franka / dynamic joints'),
            ((.32,-.50,.32),(0.,-.50,.28),'Gripper / elastic contact')]
    if rigid_cloth:
        camera=[((.76,-1.27,.89),(-.16,-.50,.28),'Franka / pick and place'),
                ((.53,-.02,.56),(.055,-.50,.235),'Free rigid box / deformable cloth')]
    font_path='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    font=ImageFont.truetype(font_path,27);small=ImageFont.truetype(font_path,21)
    def compose(index):
        e.state_0.particle_q.assign(particle[index]);e.state_0.body_q.assign(body[index])
        frame=Image.new('RGB',(1920,900),(18,28,45));draw=ImageDraw.Draw(frame)
        dt=run['physical_dt'];preview=' (requested interval; adaptive)' if run.get('solver_parameters',{}).get('automatic_rejected_step_recovery') else ' (single macro step)' if run.get('requested_single_step_dt') else ' (coarse preview)' if dt>1/600+1e-12 else ''
        mode='Offline VBD reference' if reference else f'Shared model step {run["checkpoint_step"]}'
        if run.get('analytic_predictor_rounds',0):
            analytic=run['analytic_predictor_rounds'];neural=run['solver_parameters'].get('model_unroll',1)
            mode=f'Shared model s{run["checkpoint_step"]} / A{analytic}+N{neural}'
            if run.get('analytic_elastic_postsolve',False):mode+=' / elastic correction'
        device='CUDA' if run['device'].startswith('cuda') else 'CPU'
        suffix=''
        if reference:
            relaxation=run.get('solver_parameters',{}).get('offline_reference_mimic_relaxation',1.)
            if relaxation!=1.:suffix=f' | mimic relax={relaxation:g}'
            if run.get('solver_parameters',{}).get('offline_reference_mimic_schedule')=='final_iteration':suffix+=' | mimic: final iteration'
            if run.get('ideal_mimic_count')==0:suffix+=' | independent fingers'
        asset='Franka + rigid box + cloth' if rigid_cloth else 'Franka + elastic body'
        draw.text((22,16),f'{mode} | {asset} | {device} | h={dt:.5f}s'+preview+suffix,font=font,fill='white')
        for j,(position,target,label) in enumerate(camera):
            viewer.set_camera(wp.vec3(*position));viewer.camera.look_at(np.array(target))
            viewer.begin_frame(float(times[index]));viewer.log_state(e.state_0);viewer.end_frame()
            panel=viewer.get_frame(render_ui=False).numpy()[...,:3]
            frame.paste(Image.fromarray(panel),(8+j*960,96))
            draw.text((18+j*960,59),label,font=small,fill=(190,213,240))
        complete=run['status']=='COMPLETE' and run.get('initial_command_time',0.)==0. and times[-1]>=4.-1e-9
        lifting=run.get('initial_command_time',0.)>=1.3 and run.get('initial_command_time',0.)+times[-1]>=3.3
        label='Full approach/pinch/lift/hold recording' if complete else 'From pinch pose / attempted lift' if lifting else 'Contact test / PARTIAL sequence'
        if lifting and run.get('initial_command_time',0.)<=1.3+1e-9:label='Open jaws / pinch / lift / hold'
        if lifting and run.get('soft_center_lift_m',0.)<.02:
            label='From pinch pose: object NOT lifted'
        if not rigid_cloth and complete:
            label=('Full control sequence / sustained whole object lift' if lift['whole_object_lift_sustained_250ms'] else
                   'Full control sequence / transient clearance, no sustained lift' if lift['whole_object_lift_above_5mm_observed'] else
                   'Full control sequence / COM rose, full lift NOT observed' if peak_lift>.02 else
                   'Full control sequence / object NOT lifted')
            release=run.get('teacher_release')
            if release and times[index]+run.get('initial_command_time',0.)>=release['start_time_s']:
                label+=' / release command'
        if run['status']=='FAILED':label=f'FAILED at {times[-1]:.3f}s / partial recording'
        if rigid_cloth:
            from wp_vbd.franka_rigid_cloth_scene import command
            _,_,phase=command(float(times[index])+run.get('initial_command_time',0.))
            label=f'Pick / lift / transfer / release / settle | {phase}'
            if not run.get('full_authored_timeline_recorded'):label+=' / PARTIAL'
        draw.text((22,855),f'Physical time {times[index]:.3f} s | '+label+' | 0.5x playback; not realtime',font=small,fill=(235,205,150))
        return np.asarray(frame)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    fps=30;writer=None;written=0
    if not a.snapshot_only:
        writer=imageio_ffmpeg.write_frames(str(a.output),(1920,900),fps=fps,codec='libx264',quality=8,macro_block_size=2,
                                           output_params=['-movflags','+faststart','-preset','fast','-threads','2'])
        writer.send(None)
    try:
        first=compose(0)
        Image.fromarray(first).save(a.output.with_suffix('.poster.png'))
        if writer:
            for _ in range(fps):writer.send(first);written+=1
            # Display at 60 physical samples/s even for native substep tapes.
            display_times=np.arange(0.,float(times[-1])+1e-9,1/60)
            right=np.clip(np.searchsorted(times,display_times),0,len(times)-1)
            left=np.maximum(right-1,0)
            display_indices=np.where(np.abs(times[right]-display_times)<np.abs(times[left]-display_times),right,left)
            if display_indices[-1]!=len(times)-1:display_indices=np.append(display_indices,len(times)-1)
            for index in display_indices:
                last=compose(index);writer.send(last);written+=1
                if index%30==0:print(f'render {index}/{len(times)-1}',flush=True)
            for _ in range(fps*2):writer.send(last);written+=1
        else:last=compose(len(times)-1)
        Image.fromarray(last).save(a.output.with_suffix('.final.png'))
    finally:
        if writer:writer.close()
        viewer.close()
    a.output.with_suffix('.json').write_text(json.dumps({'status':('RECORDED_REFERENCE_SNAPSHOT' if a.snapshot_only else 'RECORDED_REFERENCE_VIDEO') if reference else ('RECORDED_MODEL_SNAPSHOT' if a.snapshot_only else 'RECORDED_MODEL_VIDEO'),'recording':str(a.recording),
         'recording_mode':run['mode'],'model_inference_result':not reference,
         'recording_status':run['status'],'physical_seconds':float(times[-1]),'video_frames':written,'fps':fps,
         'peak_unweighted_particle_com_lift_m':peak_lift,'lift_above_2cm_observed':peak_lift>.02,
         'whole_object_lift':lift,
         'full_vbd_calls_during_render':0,'model_execution_during_render':False,'camera_views':camera,
         'articulation_modes':run.get('articulation_modes',False),'separate_coarse_step':run.get('separate_coarse_step',False),
         'material_motion_modes':run.get('material_motion_modes',False),'solver_parameters':run.get('solver_parameters',{}),
         'physical_parameters':run.get('physical_parameters',{})},indent=2)+'\n')


if __name__=='__main__':main()
