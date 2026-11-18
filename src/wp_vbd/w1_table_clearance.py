"""Measure original robot visual surfaces against the tabletop footprint."""
import numpy as np
from scipy.spatial.transform import Rotation


def clip_to_table(triangle, bounds):
    """Clip in XY, retaining interpolated height at every table edge."""
    polygon=list(triangle)
    for axis,edge,sign in ((0,bounds[0],1),(0,bounds[1],-1),(1,bounds[2],1),(1,bounds[3],-1)):
        if not polygon:break
        output=[];previous=polygon[-1];a=sign*(previous[axis]-edge)
        for point in polygon:
            b=sign*(point[axis]-edge)
            if (a>=0)!=(b>=0):output.append(previous+(point-previous)*(a/(a-b)))
            if b>=0:output.append(point)
            previous=point;a=b
        polygon=output
    return np.asarray(polygon).reshape(-1,3)


class TableClearance:
    def __init__(self,model,*,payload_id,bounds=(.445,.985,-.6,.6),top=1.08):
        self.bounds=np.asarray(bounds);self.top=float(top);self.meshes=[]
        owners=model.shape_body.numpy();poses=model.shape_transform.numpy();scales=model.shape_scale.numpy()
        for i,source in enumerate(model.shape_source):
            if source is None or owners[i]==payload_id:continue
            vertices=np.asarray(source.vertices,dtype=np.float64)*scales[i]
            local=Rotation.from_quat(poses[i,3:]).apply(vertices)+poses[i,:3]
            self.meshes.append((int(owners[i]),local,np.asarray(source.indices).reshape(-1,3)))

    def measure(self,body_pose):
        best=float('inf');worst=None;bounds=self.bounds
        for owner,local,faces in self.meshes:
            points=local if owner<0 else Rotation.from_quat(body_pose[owner,3:]).apply(local)+body_pose[owner,:3]
            low=points.min(0);high=points.max(0)
            if high[0]<bounds[0] or low[0]>bounds[1] or high[1]<bounds[2] or low[1]>bounds[3]:continue
            if low[2]-self.top>=best:continue
            triangles=points[faces];lo=triangles.min(1);hi=triangles.max(1)
            selected=(hi[:,0]>=bounds[0])&(lo[:,0]<=bounds[1])&(hi[:,1]>=bounds[2])&(lo[:,1]<=bounds[3])
            ids=np.flatnonzero(selected)
            for index in ids[np.argsort(lo[ids,2])]:
                if lo[index,2]-self.top>=best:break
                clipped=clip_to_table(triangles[index],bounds)
                if not len(clipped):continue
                gap=float(clipped[:,2].min()-self.top)
                if gap<best:best=gap;worst=owner
        return {'minimum_robot_visual_table_clearance_m':best if np.isfinite(best) else None,
                'closest_robot_body':worst,'robot_visual_meshes':len(self.meshes)}
