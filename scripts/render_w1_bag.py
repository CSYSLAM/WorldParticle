"""Render recorded W1 bag packing: overview, transfer and mouth views."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--recording',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--snapshot-only',action='store_true')
    a=p.parse_args()
    a.recording=a.recording.expanduser().resolve();a.output=a.output.expanduser().resolve()
    run=json.loads((a.recording/'run.json').read_text())
    if hashlib.sha256(Path(run['asset']).read_bytes()).hexdigest()!=run['asset_sha256']:
        raise ValueError('The source robot URDF has changed since recording')
    preview=run['mode']=='asset_geometry_preview'
    if preview and not a.snapshot_only:raise ValueError('Geometry previews are snapshots only')
    if not preview and (run['mode']!='shared_network_inference' or any(run['forbidden_solver_calls'].values()) or run['teacher_trajectory_reads']):
        raise ValueError('An independent shared-network recording is required')
    os.environ.update(CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',__GLX_VENDOR_LIBRARY_NAME='mesa',LP_NUM_THREADS='2')
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    if not preview and run['newton_source']['revision']!=newton_revision():raise ValueError('Render with the recorded Newton source revision')
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    from wp_vbd.w1_bag_scene import build,command,deformation_metrics
    torch.set_num_threads(1);wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    def forbidden(*args,**kwargs):raise AssertionError('Rendering must not execute dynamics')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):cls.__init__=forbidden;cls.step=forbidden
    e=build(wp,newton,asset=run['asset'],output=a.recording/'render_asset',device='cpu',initialize_controls=False,
            friction=run.get('friction',1.5),bag_profile=run.get('bag_profile','stiff'),bag_grip=run.get('bag_grip','horizontal'))
    tape=a.recording/'trajectory.npz'
    if not tape.exists() and a.snapshot_only:
        tape=a.recording/('progress.npz' if (a.recording/'progress.npz').exists() else 'initial_state.npz')
    with np.load(tape) as z:
        bodies=z['body_pose'];particles=z['particle_position'];times=z['physical_time']
    if bodies.shape!=(len(times),e.model.body_count,7) or particles.shape!=(len(times),e.model.particle_count,3):
        raise ValueError('Recording topology differs from the rendered asset')
    viewer=newton.viewer.ViewerGL(width=944,height=748,vsync=False,headless=True,enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE)
    viewer.set_model(e.model);viewer.show_ui=False;viewer.show_particles=False;viewer.show_joints=False;viewer.show_contacts=False
    viewer.renderer.draw_sky=False;viewer.renderer.draw_fps=False;viewer.renderer.spotlight_enabled=False
    viewer.renderer.draw_edges=False
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',20)
    viewer.show_triangles=True
    cameras=[((2.1,-1.7,1.95),(.23,0.,1.05),'W1 / bag packing / shared model'),
             ((1.30,-.85,1.95),(.52,-.03,1.22),'Packing / transfer and release'),
             ((.73,.04,2.2),(.53,.03,1.08),'Top view / cube containment')]
    if run.get('bag_grip')=='oblique':
        cameras[2]=((1.05,.78,1.95),(.53,.20,1.25),'Bag rim / oblique pinch')
    def frame(index):
        e.state_0.body_q.assign(bodies[index]);e.state_0.particle_q.assign(particles[index])
        canvas=Image.new('RGB',(1920,900),(18,28,45));draw=ImageDraw.Draw(canvas)
        title='W1 asset geometry preview' if preview else f'W1 bag packing | shared model step {run["checkpoint_step"]} | {run["device"]} inference'
        draw.text((20,15),title,font=font,fill='white')
        for j,(position,target,label) in enumerate(cameras):
            if run.get('bag_grip')=='oblique' and j==1 and times[index]>=3.35:
                position,target,label=(.73,.04,2.2),(.53,.03,1.08),'Inside bag / released cube'
            viewer.set_camera(wp.vec3(*position));viewer.camera.look_at(np.array(target))
            viewer.begin_frame(float(times[index]));viewer.log_state(e.state_0);viewer.end_frame()
            panel=Image.fromarray(viewer.get_frame(render_ui=False).numpy()[...,:3])
            left=8 if j==0 else 968+(j-1)*472
            if j:panel=panel.crop((236,0,708,748))
            canvas.paste(panel,(left,95));draw.text((left+10,60),label,font=small,fill=(190,213,240))
        phase=command(float(times[index]))
        if run.get('bag_profile')=='flexible':
            if .85<=times[index]<1.6:phase='lift cube / gently compress bag rim'
            if 3.7<=times[index]<4.35:phase='raise loaded rim / material recovery'
        scope='Geometry only' if preview else 'Full grasp / transfer / release / settle' if run['full_authored_timeline_recorded'] else 'PARTIAL recording'
        draw.text((20,855),f'Physical time {times[index]:.3f} s | {phase} | {scope} | 0.25x replay; not realtime',font=small,fill=(235,205,150))
        if run.get('bag_profile')=='flexible':
            shape=deformation_metrics(e.bag_rest,particles[index])
            draw.text((20,878),f'Actual nonrigid deformation: RMS {1000*shape["rms_m"]:.1f} mm | maximum {1000*shape["maximum_m"]:.1f} mm | rigid motion removed',
                      font=small,fill=(170,220,235))
        return np.asarray(canvas)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    writer=None;count=0;fps=30;thumbnails=[]
    selected=set(np.argmin(abs(times-t)) for t in (0.,.85,1.6,2.6,3.35,4.8))
    try:
        first=frame(0);Image.fromarray(first).save(a.output.with_suffix('.poster.png'))
        if a.snapshot_only:last=frame(len(times)-1)
        else:
            if a.output.exists():raise FileExistsError(a.output)
            writer=imageio_ffmpeg.write_frames(str(a.output),(1920,900),fps=fps,codec='libx264',quality=8,macro_block_size=2,
                output_params=['-movflags','+faststart','-preset','fast','-threads','2'])
            writer.send(None)
            for _ in range(15):writer.send(first);count+=1
            for index in range(len(times)):
                last=frame(index)
                if index in selected:thumbnails.append(Image.fromarray(last).resize((640,300)))
                for _ in range(2):writer.send(last);count+=1
            for _ in range(30):writer.send(last);count+=1
        Image.fromarray(last).save(a.output.with_suffix('.final.png'))
        if thumbnails:
            sheet=Image.new('RGB',(1920,600),(18,28,45))
            for i,panel in enumerate(thumbnails):sheet.paste(panel,((i%3)*640,(i//3)*300))
            sheet.save(a.output.with_suffix('.contact_sheet.png'))
    finally:
        if writer:writer.close()
        viewer.close()
    a.output.with_suffix('.json').write_text(json.dumps({'recording':str(a.recording.resolve()),'physical_seconds':float(times[-1]),
         'video_frames':count,'fps':fps,'cameras':cameras,
         'inside_bag_camera_switch_s':3.35 if run.get('bag_grip')=='oblique' else None,
         'model_inference_result':not preview,'full_dynamics_calls_during_render':0},indent=2)+'\n')


if __name__=='__main__':main()
