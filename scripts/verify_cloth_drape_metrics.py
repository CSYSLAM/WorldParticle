"""An almost area-preserving rolled sheet must fail broad-drape screening."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    import numpy as np
    from wp_vbd.cloth_drape_metrics import drape_metrics
    n=64;length=.18
    rest=np.array([[i*length/n,j*length/n,0.] for i in range(n+1) for j in range(n+1)])
    faces=[]
    for i in range(n):
        for j in range(n):
            k=i*(n+1)+j;faces.extend(((k,k+n+1,k+n+2),(k,k+n+2,k+1)))
    faces=np.array(faces)
    flat=drape_metrics(rest,rest,faces,0)
    assert abs(flat['panel_width_ratio']-1)<1e-12 and abs(flat['panel_normal_coherence']-1)<1e-12
    radius=length/(2*np.pi*1.25);angle=rest[:,0]/radius
    rolled=np.stack((radius*np.sin(angle),rest[:,1],radius*(1-np.cos(angle))),-1)
    measured=drape_metrics(rest,rolled,faces,0)
    def area(x):
        t=x[faces];return np.linalg.norm(np.cross(t[:,1]-t[:,0],t[:,2]-t[:,0]),axis=-1)
    ratio=area(rolled)/area(rest)
    assert ratio.min()>.995 and ratio.max()<1.005
    assert measured['panel_width_ratio']<.55 and measured['panel_normal_coherence']<.60
    q,_=np.linalg.qr(np.random.default_rng(41).normal(size=(3,3)))
    moved=drape_metrics(rest@q.T+[1.,-.2,.4],rolled@q.T+[1.,-.2,.4],faces,0)
    # Only invariant geometric quantities are compared, not eigenvectors.
    for key in ['panel_width_ratio','panel_normal_coherence']:
        assert abs(measured[key]-moved[key])<1e-10
    result=dict(status='PASS',flat=flat,rolled=measured,rolled_area_ratio_range=[float(ratio.min()),float(ratio.max())],
        rotation_translation_invariant=True,scope='Appearance diagnostic for broad hanging cloth, not a universal fold prohibition.')
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(a.report.read_text())


if __name__=='__main__':main()
