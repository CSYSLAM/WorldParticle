"""Native two-camera video from recorded model states, with explicit physical time."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),'/home/oem/code/engine/newton']


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--recording',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);parser.add_argument('--allow-partial',action='store_true')
    parser.add_argument('--thumbnail-only',action='store_true');parser.add_argument('--fps',type=int,default=30)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.json').exists():raise FileExistsError(args.output)
    record=args.recording/'recording.json';record_raw=record.read_bytes();run=json.loads(record_raw)
    if run['status']!='MODEL_RECORDING_COMPLETE' and not args.allow_partial:raise ValueError('Complete model recording required')
    if run['frames']<1 or run['teacher_trajectory_reads'] or run['full_vbd_prediction_calls']!={'construct':0,'step':0}:raise ValueError('Actual non-teacher model states required')
    os.environ.update(CUDA_VISIBLE_DEVICES='',LIBGL_ALWAYS_SOFTWARE='1',MESA_LOADER_DRIVER_OVERRIDE='llvmpipe',__GLX_VENDOR_LIBRARY_NAME='mesa')
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    import imageio_ffmpeg
    import warp as wp
    import newton
    import newton.viewer
    from wp_vbd.official import make_example
    wp.config.kernel_cache_dir=str(ROOT/'.local/p0/warp-cache');wp.init();wp.set_device('cpu')
    width,height=936,810
    viewer=newton.viewer.ViewerGL(width=width,height=height,vsync=False,headless=True,enable_cuda_interop=newton.viewer.ViewerGL.CudaInterop.NONE)
    example,_,_=make_example('cloth.example_cloth_twist',reference=False,viewer=viewer,preserve_defaults=True)
    if run.get('preview_asset'):
        from wp_vbd.cloth_twist_preview import resize_example
        resized=resize_example(example,run['preview_asset']['resolution'],wp,newton)
        if resized!=run['preview_asset']:raise ValueError('Recorded preview asset changed')
    def forbidden(*unused,**kwargs):raise AssertionError('No dynamics during offline video rendering')
    newton.solvers.SolverVBD.__init__=forbidden;newton.solvers.SolverVBD.step=forbidden
    example.step=forbidden;example.solver.step=forbidden
    from pyglet import gl
    renderer=gl.gl_info.get_renderer()
    if 'llvmpipe' not in renderer.lower() and 'softpipe' not in renderer.lower():raise RuntimeError('Software rendering required')
    viewer.show_ui=False;viewer.show_particles=False;viewer.show_joints=False;viewer.show_contacts=False
    viewer.renderer.draw_sky=False;viewer.renderer.draw_fps=False;viewer.renderer.draw_edges=False
    viewer.renderer.sky_upper=(.90,.93,.97);viewer.renderer.sky_lower=(.90,.93,.97)
    viewer.renderer.background_color=(.90,.93,.97);viewer.renderer.spotlight_enabled=False
    positions=np.load(args.recording/'particle_position.npy',mmap_mode='r')[:run['frames']+1]
    with np.load(args.recording/'static.npz') as static:
        assert np.array_equal(static['rest'],example.model.particle_q.numpy()) and np.array_equal(static['faces'],example.faces)
        faces=static['faces'];rest=static['rest'].astype(float);pinned=static['pinned']
    edges=np.unique(np.sort(np.concatenate((faces[:,[0,1]],faces[:,[1,2]],faces[:,[2,0]])),axis=1),axis=0)
    grid_mask=np.zeros(len(edges),dtype=bool)
    for dimension in (1,2):
        levels=np.unique(rest[:,dimension]);chosen=np.unique(np.r_[levels[::5],levels[-1]])
        aligned=np.isclose(rest[edges[:,0],dimension],rest[edges[:,1],dimension],rtol=0,atol=1e-7)
        grid_mask |= aligned & np.isclose(rest[edges[:,0],dimension,None],chosen[None],rtol=0,atol=1e-7).any(-1)
    grid_edges=edges[grid_mask]
    pin_colors=wp.array(np.tile(np.array([.08,.42,.91],np.float32),(len(pinned),1)),dtype=wp.vec3,device='cpu')
    pin_radii=wp.full(len(pinned),.005,dtype=wp.float32,device='cpu')
    if not np.isfinite(positions).all():raise ValueError('Finite recorded vertices required')
    resolution=run.get('preview_asset',{}).get('resolution',50) if run.get('preview_asset') else 50
    if len(pinned)!=2*resolution:raise ValueError('Both original-style driven edges required')
    reference=rest[pinned]
    positive_edge=reference[:,1]>(reference[:,1].min()+reference[:,1].max())/2
    if int(positive_edge.sum())!=resolution:raise ValueError('Matched driven edges on opposite Y sides required')
    angle=np.arctan2(reference[None,:,2]*positions[:,pinned,0]-reference[None,:,0]*positions[:,pinned,2],
        reference[None,:,0]*positions[:,pinned,0]+reference[None,:,2]*positions[:,pinned,2])
    angle=np.unwrap(angle,axis=0)
    relative_turns=np.abs(angle[:,positive_edge].mean(1)-angle[:,~positive_edge].mean(1))/(2*np.pi)
    bounds=np.stack((positions.min((0,1)),positions.max((0,1)))).astype(float);center=bounds.mean(0)
    points=np.asarray(positions).reshape(-1,3).astype(float)-center
    cameras=[]
    for axis in [[1.,.42,.28],[.48,-1.,.24]]:
        axis=np.array(axis);axis/=np.linalg.norm(axis);front=-axis
        right=np.cross(front,[0,0,1.]);right/=np.linalg.norm(right);up=np.cross(right,front)
        tan_v=np.tan(np.deg2rad(viewer.camera.fov)/2);tan_h=tan_v*width/height
        depth=points@front
        distance=max(float((np.abs(points@right)/(.82*tan_h)-depth).max()),float((np.abs(points@up)/(.82*tan_v)-depth).max()))+.04
        eye=center+axis*distance
        clip_x=np.abs(points@right)/(distance+depth)/tan_h;clip_y=np.abs(points@up)/(distance+depth)/tan_v
        if max(clip_x.max(),clip_y.max())>=.9:raise ValueError('Recorded motion extends outside fitted camera')
        cameras.append({'eye':eye.tolist(),'target':center.tolist(),'maximum_normalized_horizontal_extent':float(clip_x.max()),'maximum_normalized_vertical_extent':float(clip_y.max()),'fov':viewer.camera.fov})
    font_path='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
    large=ImageFont.truetype(font_path,37);medium=ImageFont.truetype(font_path,25);small=ImageFont.truetype(font_path,22)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    source_fps=1/run['frame_dt'];playback=args.fps/source_fps
    complete_recording=run['status']=='MODEL_RECORDING_COMPLETE'
    def compose(index):
        frame=Image.new('RGB',(1920,1080),(15,23,42));draw=ImageDraw.Draw(frame)
        draw.text((30,20),f'Cloth Twist · 共享模型 / 第 {run["checkpoint_step"]} 步权重',font=large,fill=(240,245,252))
        preview='大时间步预览' if run.get('explicit_numerical_preview') else '原始时间步'
        precision=run.get('neural_precision',{}).get('policy','fp32').upper()
        geometry=f'低分辨率 {run["preview_asset"]["resolution"]}×{run["preview_asset"]["resolution"]} 预览' if run.get('preview_asset') else f'{len(rest)} 粒子'
        schedule=run.get('time_step_schedule',[])
        step_text=f'h = {run["physical_dt"]:.6g}s（{preview}）'
        if len(schedule)>1:step_text='h = '+' → '.join(f'{row["physical_dt"]:.6g}' for row in schedule)+'s（分段时间步）'
        draw.text((32,77),f'离线模型推理回放  |  {geometry}  |  {step_text} | {precision}神经计算',font=small,fill=(167,188,212))
        for camera,x,label in zip(cameras,[24,960],['斜正面 · 看整体形态','侧视角 · 看扭转与褶皱']):
            draw.text((x+12,122),label,font=medium,fill=(196,219,243))
            example.state_0.particle_q.assign(positions[index]);example.sim_time=index*run['frame_dt']
            viewer.set_camera(wp.vec3(*camera['eye']));viewer.camera.look_at(tuple(center))
            viewer.begin_frame(example.sim_time);viewer.log_state(example.state_0)
            # Annotation only: a submillimetre view-facing offset avoids
            # z-fighting. The rendered cloth and all recorded states stay exact.
            view_axis=np.array(camera['eye'])-center;view_axis/=np.linalg.norm(view_axis)
            line_points=positions[index]+view_axis*.0002
            viewer.log_lines('cloth_reading_grid',wp.array(line_points[grid_edges[:,0]],dtype=wp.vec3,device='cpu'),
                wp.array(line_points[grid_edges[:,1]],dtype=wp.vec3,device='cpu'),(.45,.32,.10))
            viewer.log_points('driven_boundary',wp.array(positions[index,pinned],dtype=wp.vec3,device='cpu'),radii=pin_radii,colors=pin_colors)
            viewer.end_frame()
            panel=viewer.get_frame(render_ui=False).numpy()[...,:3]
            if panel.shape!=(height,width,3):raise ValueError('Exact renderer dimensions required')
            frame.paste(Image.fromarray(panel),(x,164))
        physical=index*run['frame_dt']
        draw.text((32,995),f'物理时间 {physical:5.2f} / {run["physical_seconds"]:.2f} s   ·   相对边界扭转 {relative_turns[index]:.2f} 圈   ·   {playback:.2f}×播放',font=medium,fill=(235,241,250))
        footer='蓝点：驱动边界。无完整 VBD 推理 / 无教师轨迹输入；播放速度不代表实时求解速度。'
        if not complete_recording:
            footer=f'有效片段：{run["physical_seconds"]:.2f}s 后录制中止。'
            if run['status']=='RECORDING_INDEPENDENT_MODEL_STATES':
                footer='录制中的相机检查。'
            if 'self-contact path or detection envelope' in str(run.get('failure')):
                footer=f'有效片段：{run["physical_seconds"]:.2f}s 后驱动输入触发自接触保护。'
            footer+=' 蓝点：驱动边界；非完整场景结果，播放速度不代表实时求解速度。'
        draw.text((32,1037),footer,font=small,fill=(151,174,203) if complete_recording else (248,198,130))
        return np.asarray(frame)
    first=compose(0);Image.fromarray(first).save(args.output.with_suffix('.poster.png'))
    snapshots=[0,run['frames']//2,run['frames']]
    for index in snapshots:Image.fromarray(compose(index)).save(args.output.with_name(args.output.stem+f'_frame{index:04d}.png'))
    written=0
    try:
        if not args.thumbnail_only:
            writer=imageio_ffmpeg.write_frames(str(args.output),(1920,1080),fps=args.fps,codec='libx264',quality=8,macro_block_size=2,
                output_params=['-movflags','+faststart','-preset','fast','-threads','2'])
            writer.send(None)
            try:
                for _ in range(args.fps):writer.send(first);written+=1
                for index in range(len(positions)):
                    latest=compose(index);writer.send(latest);written+=1
                    if index%30==0:print(f'rendered={index}/{run["frames"]}',flush=True)
                for _ in range(2*args.fps):writer.send(latest);written+=1
            finally:writer.close()
    finally:viewer.close()
    result={'status':'RECORDED_CLOTH_MODEL_VIDEO_RENDERED' if not args.thumbnail_only else 'CAMERA_PREVIEW_RENDERED',
        'video':str(args.output),'video_frames':written,'video_fps':args.fps,'physical_seconds':run['physical_seconds'],
        'playback_physical_speed':playback,'start_hold_seconds':1,'end_hold_seconds':2,
        'source_recording':str(record),'source_recording_sha256':hashlib.sha256(record_raw).hexdigest(),'checkpoint_sha256':run['checkpoint_sha256'],
        'recorded_frames':run['frames'],'camera_views':cameras,'bounds':bounds.tolist(),'renderer':renderer,
        'complete_recording':complete_recording,'source_recording_status':run['status'],'source_recording_failure':run.get('failure'),
        'grid_annotation_view_facing_offset_m':.0002,'rendered_cloth_vertices_unmodified':True,
        'full_vbd_prediction_calls':0,'teacher_trajectory_reads':0,'model_execution_during_render':False,
        'explicit_numerical_preview':run.get('explicit_numerical_preview',False),'physical_dt':run['physical_dt'],
        'preview_asset':run.get('preview_asset'),'explicit_geometry_preview':run.get('explicit_geometry_preview',False),
        'measured_final_relative_boundary_turns':float(relative_turns[-1]),
        'time_step_schedule':run.get('time_step_schedule',[]),'own_state_resume':run.get('resume'),
        'script_sha256':digest(Path(__file__)),'scope':'Native rendering of actual model states with two fixed cameras fitted to the entire recorded motion. No teacher, dynamics substitution, physical interpolation or realtime performance claim. Numerical preview h is explicitly labeled when changed.'}
    args.output.with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    args.output.with_suffix('.recording_snapshot.json').write_bytes(record_raw)


if __name__=='__main__':main()
