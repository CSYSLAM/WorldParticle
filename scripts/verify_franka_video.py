"""Fully decode a model video and export a phase contact sheet for review."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--report',type=Path,required=True)
    a=p.parse_args()
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image,ImageDraw,ImageFont
    rendering=json.loads(a.video.with_suffix('.json').read_text())
    run=json.loads((Path(rendering['recording'])/'run.json').read_text())
    if run['mode']!='shared_network_inference' or any(run['forbidden_solver_calls'].values()) or run['teacher_trajectory_reads']:
        raise ValueError('An independent shared-network recording is required')
    seconds=rendering['physical_seconds']
    sampled=[0.,.95,1.6,2.3,3.5,4.3,5.,5.8,seconds] if run['scene'].endswith('rigid_cloth') else [0.,1.3,2.1,3.3,seconds]
    sampled=sorted(set(t for t in sampled if t<=seconds+1e-9))
    wanted={int(round(rendering['fps']+60*t)):t for t in sampled}
    reader=imageio_ffmpeg.read_frames(str(a.video),pix_fmt='rgb24')
    meta=next(reader);size=meta['size'];images={};count=0
    try:
        for i,payload in enumerate(reader):
            if len(payload)!=size[0]*size[1]*3:raise ValueError('Incomplete video frame')
            if i in wanted:images[wanted[i]]=Image.frombytes('RGB',size,payload)
            count=i+1
    finally:reader.close()
    if count!=rendering['video_frames']:raise ValueError(f'Decoded {count} frames, expected {rendering["video_frames"]}')
    if set(images)!=set(sampled):raise ValueError('Missing phase images')
    width,height=640,324
    sheet=Image.new('RGB',(width*3,height*int(np.ceil(len(sampled)/3))),(18,28,45))
    draw=ImageDraw.Draw(sheet);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for i,t in enumerate(sampled):
        left=(i%3)*width;top=(i//3)*height
        sheet.paste(images[t].resize((width,300)),(left,top))
        draw.text((left+8,top+302),f'Physical time {t:.2f} s',font=font,fill='white')
    contact_sheet=a.video.with_name(a.video.stem+'.contact_sheet.png');sheet.save(contact_sheet)
    result={'status':'PASS','decoded_frames':count,'fps':meta['fps'],'video_seconds':count/meta['fps'],
            'physical_seconds':seconds,'recording_status':run['status'],
            'full_authored_timeline_recorded':run['full_authored_timeline_recorded'],
            'presentation':'0.5x playback plus 1 s initial / 2 s final holds; not realtime inference',
            'sha256':hashlib.sha256(a.video.read_bytes()).hexdigest(),
            'sampled_physical_times':sampled,'contact_sheet':str(contact_sheet),
            'scope':'Full codec decode and phase coverage; physical plausibility requires geometry and visual review.'}
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n')
    print(a.report.read_text())


if __name__=='__main__':main()
