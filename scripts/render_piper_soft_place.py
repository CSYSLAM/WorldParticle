"""Render accepted Piper states with overview and grasp/placement close-up."""
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
    run=json.loads((a.recording/'run.json').read_text())
    preview=run['mode']=='asset_geometry_preview'
    if preview and not a.snapshot_only:
        p.error('Geometry previews are snapshots only')
    if any(run['forbidden_solver_calls'].values()) or run['teacher_trajectory_reads']:
        raise ValueError('Only independent model trajectories may be rendered')
    if hashlib.sha256(Path(run['asset']).read_bytes()).hexdigest()!=run['asset_sha256']:
        raise ValueError('The source Piper asset has changed')
    os.environ.update(CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',
        __GLX_VENDOR_LIBRARY_NAME='mesa',LP_NUM_THREADS='2')
    from wp_vbd.newton_source import activate_newton_source,newton_revision
    activate_newton_source()
    if newton_revision()!=run['newton_source']['revision']:
        raise ValueError('Use the recorded Newton revision')
    import numpy as np
    import torch
    import warp as wp
    import newton
    import newton.viewer
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    from wp_vbd.piper_soft_place_scene import build,phase
    torch.set_num_threads(1)
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache')
    wp.init()
    wp.set_device('cpu')
    def forbidden(*args,**kwargs):
        raise AssertionError('No dynamics during video rendering')
    for cls in (newton.solvers.SolverVBD,newton.solvers.SolverFeatherstone):
        cls.__init__=forbidden
        cls.step=forbidden
    e=build(wp,newton,asset=run['asset'],output=a.recording/'render_asset',device='cpu',
        initialize_controls=False,friction=run['physical_parameters']['friction'])
    with np.load(a.recording/'trajectory.npz') as tape:
        particles=tape['particle_position'];bodies=tape['body_pose'];times=tape['physical_time']
        assert np.array_equal(tape['faces'],e.model.tri_indices.numpy())
    viewer=newton.viewer.ViewerGL(width=944,height=748,vsync=False,headless=True,
        enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE)
    viewer.set_model(e.model)
    viewer.show_ui=False;viewer.show_particles=False;viewer.show_joints=False;viewer.show_contacts=False
    viewer.renderer.draw_sky=False;viewer.renderer.draw_fps=False;viewer.renderer.draw_edges=False
    viewer.renderer.background_color=(.90,.93,.97);viewer.renderer.spotlight_enabled=False
    cameras=[((1.05,-1.0,.80),(.28,0.,.25),'Piper / full robot and work area'),
        ((.80,-.55,.55),(.40,0.,.21),'Grasp and place / soft block on free rigid cube')]
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',26)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    def frame(index):
        e.state_0.particle_q.assign(particles[index]);e.state_0.body_q.assign(bodies[index])
        canvas=Image.new('RGB',(1920,900),(18,28,45));draw=ImageDraw.Draw(canvas)
        title='Piper / GEOMETRY PREVIEW ONLY' if preview else f'Piper | shared model s{run["checkpoint_step"]} | {run["device"]} | A{run["analytic_predictor_rounds"]}+N{run["model_unroll"]}'
        draw.text((20,15),title,font=font,fill='white')
        for j,(position,target,label) in enumerate(cameras):
            viewer.set_camera(wp.vec3(*position));viewer.camera.look_at(np.array(target))
            viewer.begin_frame(float(times[index]));viewer.log_state(e.state_0);viewer.end_frame()
            canvas.paste(Image.fromarray(viewer.get_frame(render_ui=False).numpy()[...,:3]),(8+j*960,95))
            draw.text((18+j*960,60),label,font=small,fill=(190,213,240))
        draw.text((20,851),f'Physical time {times[index]:.3f} s | {phase(float(times[index]))} | blue: soft tetrahedral block / orange: dynamic rigid cube',font=small,fill=(225,220,200))
        draw.text((20,878),'Motor targets only; no object attachment | 0.5x offline replay; not realtime' if not preview else 'Kinematic clearance check; no physics inference in this preview',font=small,fill=(170,220,235))
        return np.asarray(canvas)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    writer=None;count=0;thumbnails=[]
    selected=set(np.argmin(abs(times-t)) for t in (0.,1.4,2.1,2.9,4.,5.2))
    try:
        first=frame(0);Image.fromarray(first).save(a.output.with_suffix('.poster.png'))
        if a.snapshot_only:
            last=frame(len(times)-1)
        else:
            if a.output.exists():raise FileExistsError(a.output)
            writer=imageio_ffmpeg.write_frames(str(a.output),(1920,900),fps=30,codec='libx264',quality=8,
                macro_block_size=2,output_params=['-movflags','+faststart','-preset','fast','-threads','2'])
            writer.send(None)
            for _ in range(15):writer.send(first);count+=1
            for i in range(len(times)):
                last=frame(i)
                if i in selected:thumbnails.append(Image.fromarray(last).resize((640,300)))
                dt=times[min(i+1,len(times)-1)]-times[i] if i<len(times)-1 else 1/60
                for _ in range(max(1,round(dt*60))):writer.send(last);count+=1
            for _ in range(30):writer.send(last);count+=1
        Image.fromarray(last).save(a.output.with_suffix('.final.png'))
        if thumbnails:
            sheet=Image.new('RGB',(1920,600),(18,28,45))
            for i,panel in enumerate(thumbnails):sheet.paste(panel,((i%3)*640,(i//3)*300))
            sheet.save(a.output.with_suffix('.contact_sheet.png'))
    finally:
        if writer:writer.close()
        viewer.close()
    a.output.with_suffix('.json').write_text(json.dumps(dict(recording=str(a.recording.resolve()),
        physical_seconds=float(times[-1]),video_frames=count,fps=30,playback_speed=.5,cameras=cameras,
        geometry_only=preview,full_dynamics_calls_during_render=0),indent=2)+'\n')


if __name__=='__main__':
    main()
