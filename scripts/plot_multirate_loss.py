"""Render training loss curves without requiring a plotting package."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--window',type=int,default=25)
    args=parser.parse_args()
    from PIL import Image,ImageDraw,ImageFont
    report=json.loads(args.training.read_text())
    samples=report['samples']
    grouped=defaultdict(list)
    for row in samples:grouped[row['dataset']].append((row['update'],row['loss']))
    colors=['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf']
    width,height=2100,1050;left,right,top,bottom=100,385,90,80
    image=Image.new('RGB',(width,height),'white');draw=ImageDraw.Draw(image)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',17)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',30)
    draw.text((left,18),'Multirate Training Loss: 200 Updates',font=title,fill='black')
    panel_gap=90;panel_w=(width-left-right-panel_gap)//2
    x0,x1=left,left+panel_w;y0,y1=top,height-bottom
    values=np.asarray([max(float(row['loss']),1e-10) for row in samples])
    low,high=max(np.quantile(values,.015)*.65,1e-10),np.quantile(values,.985)*1.4
    def px(update,a,b):return a+(update-1)/max(report['completed_updates']-1,1)*(b-a)
    def py(value,lo,hi):
        value=np.clip(value,lo,hi)
        return y1-(np.log10(value)-np.log10(lo))/(np.log10(hi)-np.log10(lo))*(y1-y0)
    draw.rectangle((x0,y0,x1,y1),outline='#555')
    for tick in np.linspace(np.log10(low),np.log10(high),8):
        value=10**tick;y=py(value,low,high);draw.line((x0,y,x1,y),fill='#e7e7e7');draw.text((8,y-10),f'{value:.2g}',font=small,fill='#444')
    for update in range(0,report['completed_updates']+1,25):
        x=px(update,x0,x1);draw.line((x,y0,x,y1),fill='#eeeeee');draw.text((x-12,y1+12),str(update),font=small,fill='#444')
    draw.text((x0,y0-42),'Total loss by update (log scale)',font=font,fill='black')
    draw.text((x0,y1+40),'optimizer update',font=small,fill='black')
    raw=[(px(row['update'],x0,x1),py(row['loss'],low,high)) for row in samples]
    for a,b in zip(raw,raw[1:]):draw.line((*a,*b),fill='#b0b0b0',width=1)
    total=np.asarray([max(float(row['loss']),1e-10) for row in samples])
    trend=np.asarray([np.median(total[max(0,i-args.window+1):i+1]) for i in range(len(total))])
    smooth=[(px(row['update'],x0,x1),py(value,low,high)) for row,value in zip(samples,trend)]
    for a,b in zip(smooth,smooth[1:]):draw.line((*a,*b),fill='#245f9e',width=4)
    draw.text((12,y0-42),'loss',font=small,fill='black')
    x0b,x1b=x1+panel_gap,width-right
    draw.rectangle((x0b,y0,x1b,y1),outline='#555')
    for update in range(0,report['completed_updates']+1,25):
        x=px(update,x0b,x1b);draw.line((x,y0,x,y1),fill='#eeeeee');draw.text((x-12,y1+12),str(update),font=small,fill='#444')
    draw.text((x0b,y0-42),f'Dataset loss / first-{args.window} update median',font=font,fill='black')
    draw.text((x0b,y1+40),'optimizer update',font=small,fill='black')
    series=[]
    for index,(dataset,rows) in enumerate(sorted(grouped.items())):
        rows=sorted(rows);updates=np.asarray([r[0] for r in rows]);loss=np.asarray([max(r[1],1e-10) for r in rows])
        smooth=np.asarray([np.median(loss[max(0,i-args.window+1):i+1]) for i in range(len(loss))])
        baseline=np.median(loss[:min(args.window,len(loss))]);normalized=smooth/max(baseline,1e-12)
        points=[(px(int(u),x0b,x1b),py(v,.35,4.0)) for u,v in zip(updates,normalized)]
        color=colors[index%len(colors)]
        for a,b in zip(points,points[1:]):draw.line((*a,*b),fill=color,width=3)
        series.append((dataset,color))
    for tick in (.5,1.,2.,3.,4.):
        y=py(tick,.35,4.);draw.line((x0b,y,x1b,y),fill='#e7e7e7')
        draw.text((x0b-42,y-9),str(tick),font=small,fill='#444')
    for index,(dataset,color) in enumerate(series):
        y=y0+index*38;draw.line((x1b+16,y+10,x1b+56,y+10),fill=color,width=4)
        label=dataset.replace('_train_default','').replace('_mu5000_lambda5000_grasp_v310','').replace('_multirate_20261118','')
        label=label if len(label)<40 else label[:37]+'...'
        draw.text((x1b+65,y),label,font=small,fill='black')
    draw.text((x0b-42,y0-4),'ratio',font=small,fill='black')
    args.output.parent.mkdir(parents=True,exist_ok=True);image.save(args.output)
    print(args.output)


if __name__=='__main__':main()
