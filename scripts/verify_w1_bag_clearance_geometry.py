"""Independent solid/surface cases for the offline W1 visual geometry audit."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    import numpy as np
    import trimesh
    from scipy.spatial.transform import Rotation
    from wp_vbd.w1_bag_clearance import GripperBagClearance

    class Array:
        def __init__(self,value):self.value=np.asarray(value)
        def numpy(self):return self.value

    box=trimesh.creation.box(extents=[.02,.02,.02])
    def audit(faces):
        source=SimpleNamespace(vertices=box.vertices,indices=np.asarray(faces).reshape(-1))
        model=SimpleNamespace(shape_body=Array([0]),shape_transform=Array([[0,0,0,0,0,0,1]]),
                              shape_scale=Array([[1,1,1]]),shape_source=[source])
        return GripperBagClearance(model,owners=(0,),spacing=.001)

    solid=audit(box.faces);pose=np.array([[0,0,0,0,0,0,1.]])
    face=np.array([[0,1,2]]);outside=np.array([[.03,0,0],[.05,0,0],[.04,.02,0]])
    assert solid.measure(pose,outside,face)['maximum_sampled_paper_inside_visual_depth_m']==0
    # Every triangle vertex is outside, but its interior crosses the box.
    crossing=np.array([[-.03,-.02,0],[.03,-.02,0],[0,.03,0]])
    original=solid.measure(pose,crossing,face)
    assert .009<original['maximum_sampled_paper_inside_visual_depth_m']<=.01000001
    rotation=Rotation.from_euler('xyz',[.2,.5,.7]);shift=np.array([.2,.3,.4])
    moved=solid.measure(np.r_[shift,rotation.as_quat()][None],rotation.apply(crossing)+shift,face)
    error=abs(original['maximum_sampled_paper_inside_visual_depth_m']-moved['maximum_sampled_paper_inside_visual_depth_m'])
    assert error<1e-12
    open_mesh=audit(box.faces[:-1])
    assert not open_mesh.measure(pose,outside,face)['unresolved_open_visual_shapes']
    assert open_mesh.measure(pose,crossing,face)['unresolved_open_visual_shapes']==[0]
    result=dict(status='PASS',separated_surface=True,interior_crossing_with_all_vertices_outside_detected=True,
                maximum_crossing_depth_m=original['maximum_sampled_paper_inside_visual_depth_m'],
                rigid_transform_error_m=error,open_mesh_separation_proved=True,
                open_mesh_overlap_reported_as_unresolved=True,native_dynamics_calls=0)
    a.report.parent.mkdir(parents=True,exist_ok=True);a.report.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
