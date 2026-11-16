"""Independent sampled geometry checks for authored native primitives.

All lengths use physical dimensions, without contact margins or solver rows.
This module is an offline observer and cannot modify a predicted trajectory.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def dot(a,b):return np.sum(a*b,axis=-1)


def point_segment_distance(point,a,b):
    edge=b-a;denominator=dot(edge,edge)
    t=np.divide(dot(point-a,edge),denominator,out=np.zeros_like(dot(point-a,edge)),where=denominator>0)
    return np.linalg.norm(point-(a+np.clip(t,0,1)[...,None]*edge),axis=-1)


def point_triangle_distance(point,triangle):
    a,b,c=np.moveaxis(triangle,-2,0);ab=b-a;ac=c-a;ap=point-a
    normal=np.cross(ab,ac);square=dot(normal,normal)
    safe=np.where(square>0,square,1.)
    v=dot(np.cross(ap,ac),normal)/safe;w=dot(np.cross(ab,ap),normal)/safe
    inside=(square>0)&(v>=0)&(w>=0)&(v+w<=1)
    plane=np.abs(dot(ap,normal))/np.sqrt(safe)
    return np.minimum(np.where(inside,plane,np.inf),np.minimum.reduce([
        point_segment_distance(point,a,b),point_segment_distance(point,b,c),point_segment_distance(point,c,a)]))


def segment_segment_distance(a,b,c,d):
    u=b-a;v=d-c;offset=a-c
    aa=dot(u,u);bb=dot(u,v);cc=dot(v,v);du=dot(u,offset);dv=dot(v,offset)
    denominator=dot(np.cross(u,v),np.cross(u,v));safe=np.where(denominator>0,denominator,1.)
    s=(bb*dv-cc*du)/safe;t=(aa*dv-bb*du)/safe
    inside=(denominator>0)&(s>=0)&(s<=1)&(t>=0)&(t<=1)
    interior=np.linalg.norm(a+s[...,None]*u-c-t[...,None]*v,axis=-1)
    return np.minimum(np.where(inside,interior,np.inf),np.minimum.reduce([
        point_segment_distance(a,c,d),point_segment_distance(b,c,d),
        point_segment_distance(c,a,b),point_segment_distance(d,a,b)]))


def segment_triangle_distance(a,b,triangle):
    p,q,r=np.moveaxis(triangle,-2,0);normal=np.cross(q-p,r-p);direction=b-a
    denominator=dot(direction,normal);safe=np.where(denominator!=0,denominator,1.)
    t=dot(p-a,normal)/safe;hit=a+t[...,None]*direction
    square=dot(normal,normal);safe_square=np.where(square>0,square,1.)
    v=dot(np.cross(hit-p,r-p),normal)/safe_square;w=dot(np.cross(q-p,hit-p),normal)/safe_square
    intersects=(denominator!=0)&(square>0)&(t>=0)&(t<=1)&(v>=0)&(w>=0)&(v+w<=1)
    distance=np.minimum.reduce([point_triangle_distance(a,triangle),point_triangle_distance(b,triangle),
        segment_segment_distance(a,b,p,q),segment_segment_distance(a,b,q,r),segment_segment_distance(a,b,r,p)])
    return np.where(intersects,0.,distance)


def triangle_box_axis_gap(triangle,half_extents):
    """Signed largest separating-axis gap; negative means triangle in box.

    This SAT quantity detects overlap, including a face with all vertices
    outside the box. Its positive value is a lower bound on Euclidean gap.
    """
    edges=np.roll(triangle,-1,axis=-2)-triangle
    axes=np.concatenate((np.broadcast_to(np.eye(3),(*triangle.shape[:-2],3,3)),
        np.cross(edges[...,0,:],edges[...,1,:])[...,None,:],
        np.cross(edges[..., :,None,:],np.eye(3)).reshape(*triangle.shape[:-2],9,3)),axis=-2)
    length=np.linalg.norm(axes,axis=-1);valid=length>0
    unit=axes/np.where(valid,length,1.)[...,None]
    projection=np.einsum('...ai,...vi->...av',unit,triangle)
    radius=np.sum(np.abs(unit)*half_extents,axis=-1)
    gap=np.maximum(projection.min(-1)-radius,-radius-projection.max(-1))
    return np.where(valid,gap,-np.inf).max(-1)


def point_box_signed_distance(point,half_extents):
    q=np.abs(point)-half_extents
    return np.linalg.norm(np.maximum(q,0),axis=-1)+np.minimum(q.max(-1),0)


def segment_box_distance(a,b,half_extents):
    """Exact segment/AABB distance via its piecewise quadratic intervals."""
    a,b=np.broadcast_arrays(a,b);direction=b-a
    safe=np.where(direction!=0,direction,1.)
    cuts=np.concatenate((np.zeros((*a.shape[:-1],1)),np.ones((*a.shape[:-1],1)),
        ((-half_extents-a)/safe),((half_extents-a)/safe)),axis=-1)
    cuts=np.sort(np.clip(cuts,0,1),axis=-1)
    lower,upper=cuts[...,:-1],cuts[...,1:];middle=(lower+upper)/2
    q=a[...,None,:]+middle[...,None]*direction[...,None,:]
    boundary=np.clip(q,-half_extents,half_extents);active=np.abs(q)>half_extents
    offset=np.where(active,a[...,None,:]-boundary,0.)
    slope=np.where(active,direction[...,None,:],0.);denominator=dot(slope,slope)
    parameter=np.divide(-dot(offset,slope),denominator,out=middle.copy(),where=denominator>0)
    parameter=np.clip(parameter,lower,upper)
    candidates=np.concatenate((cuts,parameter),axis=-1)
    points=a[...,None,:]+candidates[...,None]*direction[...,None,:]
    return np.linalg.norm(np.maximum(np.abs(points)-half_extents,0),axis=-1).min(-1)


def world_shapes(source,pose):
    kinds=np.asarray(source['shape_type']);owners=np.asarray(source['shape_body'])
    local=np.asarray(source['shape_transform'],float);result=[]
    for index,kind in enumerate(kinds):
        if kind==1:continue
        if kind not in (3,4,7):raise ValueError('Pair audit supports sphere, capsule and box shapes')
        owner=int(owners[index]);lr=Rotation.from_quat(local[index,3:]).as_matrix()
        if owner<0:
            center=np.broadcast_to(local[index,:3],(len(pose),3));rotation=np.broadcast_to(lr,(len(pose),3,3))
        else:
            rb=Rotation.from_quat(pose[:,owner,3:]).as_matrix()
            center=pose[:,owner,:3]+np.einsum('kij,j->ki',rb,local[index,:3]);rotation=rb@lr
        shape={'shape':index,'body':owner,'kind':int(kind),'center':center,'rotation':rotation,
               'scale':np.asarray(source['shape_scale'][index],float)}
        half=shape['scale'][1] if kind==4 else 0.
        shape['a']=center-rotation[:,:,2]*half;shape['b']=center+rotation[:,:,2]*half
        result.append(shape)
    return result


def to_local(shape,points):
    return np.einsum('kji,k...j->k...i',shape['rotation'],points-shape['center'].reshape(len(points),*(1,)*(points.ndim-2),3))


def shape_point_gap(shape,points):
    local=to_local(shape,points);kind=shape['kind'];scale=shape['scale']
    if kind==7:return point_box_signed_distance(local,scale)
    if kind==4:
        local=local.copy();local[...,2]-=np.clip(local[...,2],-scale[1],scale[1])
    return np.linalg.norm(local,axis=-1)-scale[0]


def box_box_axis_gap(first,second):
    ra,rb=first['rotation'],second['rotation'];delta=second['center']-first['center']
    axes=np.concatenate((ra.transpose(0,2,1),rb.transpose(0,2,1),
        np.cross(ra.transpose(0,2,1)[:,:,None],rb.transpose(0,2,1)[:,None]).reshape(len(ra),9,3)),axis=1)
    length=np.linalg.norm(axes,axis=-1);valid=length>0;unit=axes/np.where(valid,length,1.)[...,None]
    radius_a=(np.abs(unit@ra)*first['scale']).sum(-1);radius_b=(np.abs(unit@rb)*second['scale']).sum(-1)
    gap=np.abs(np.einsum('kai,ki->ka',unit,delta))-radius_a-radius_b
    return np.where(valid,gap,-np.inf).max(-1)


def pair_gap(first,second):
    if first['kind']==second['kind']==7:return box_box_axis_gap(first,second),'separating_axis_gap'
    if first['kind']==7:first,second=second,first
    if second['kind']==7:
        a=to_local(second,first['a']);b=to_local(second,first['b'])
        if first['kind']==3:return point_box_signed_distance(a,second['scale'])-first['scale'][0],'sphere_box_signed_gap'
        return segment_box_distance(a,b,second['scale'])-first['scale'][0],'capsule_box_axis_distance_minus_radius'
    return segment_segment_distance(first['a'],first['b'],second['a'],second['b'])-first['scale'][0]-second['scale'][0],'axis_distance_minus_radii'


def summarize(gap,tolerance):
    flat=gap.reshape(len(gap),-1);per_step=flat.min(-1);step,feature=np.unravel_index(flat.argmin(),flat.shape)
    return {'initial_minimum_m':float(per_step[0]),'minimum_recorded_m':float(per_step[step]),
            'minimum_step':int(step),'minimum_feature':int(feature),
            'recorded_steps_with_strict_overlap':int((per_step < -tolerance).sum()),
            'initial_strict_overlap':bool(per_step[0] < -tolerance),
            'first_strict_overlap_step':int(np.flatnonzero(per_step < -tolerance)[0]) if (per_step < -tolerance).any() else None}


def tet_containment(points,tetrahedra,vertices):
    """Strict barycentric witnesses, with explicit degenerate-cell omissions."""
    matrix=(tetrahedra[:,:,1:]-tetrahedra[:,:,:1]).transpose(0,1,3,2)
    determinant=np.linalg.det(matrix);valid=np.abs(determinant)>1e-18
    inverse=np.linalg.inv(np.where(valid[...,None,None],matrix,np.eye(3)))
    relative=points[:,None]-tetrahedra[:,:,None,0]
    coordinates=np.einsum('ktij,ktpj->ktpi',inverse,relative)
    bary=np.concatenate((1-coordinates.sum(-1,keepdims=True),coordinates),axis=-1)
    inside=(bary.min(-1)>1e-9)&valid[:,:,None]
    if vertices is not None:
        # Sharing a mesh vertex cannot be a strict interior witness for that cell.
        inside&=~(np.arange(points.shape[1])[None,:,None]==vertices[:,None,:]).any(-1)[None]
    return inside.any(1),int((~valid).sum())


def measure(source,points,pose,*,tolerance=1e-7):
    if not 0<tolerance<1e-3:raise ValueError('Explicit physical-length roundoff tolerance required')
    if points.ndim!=3 or points.shape[-1]!=3 or pose.ndim!=3 or pose.shape[-1]!=7 or len(points)!=len(pose) or not len(points):
        raise ValueError('Matched contiguous particle/body records required')
    if not np.isfinite(points).all() or not np.isfinite(pose).all():raise ValueError('Finite recorded geometry required')
    shapes=world_shapes(source,pose);faces=source.get('tri_indices',np.empty((0,3),int))
    vertex_rows=[];surface_rows=[];rigid_rows=[];contained=[]
    triangles=points[:,faces]
    for shape in shapes:
        identity={key:shape[key] for key in ('shape','body','kind')}
        if points.shape[1]:
            gap=shape_point_gap(shape,points)
            vertex_rows.append({**identity,'surface_vertex':summarize(gap,tolerance),
                'particle_radius_surface':summarize(gap-source['particle_radius'][None],tolerance)})
        if len(faces):
            if shape['kind']==7:
                gap=triangle_box_axis_gap(to_local(shape,triangles),shape['scale']);kind='triangle_box_separating_axis_gap'
            else:
                a=shape['a'][:,None];b=shape['b'][:,None]
                gap=segment_triangle_distance(a,b,triangles)-shape['scale'][0];kind='triangle_axis_distance_minus_radius'
            surface_rows.append({**identity,'measurement':kind,**summarize(gap,tolerance)})
    connected={tuple(sorted((int(p),int(c)))) for p,c in zip(source.get('joint_parent',[]),source.get('joint_child',[])) if p>=0 and c>=0}
    for i,first in enumerate(shapes):
        for second in shapes[i+1:]:
            if first['body']==second['body']:continue
            gap,kind=pair_gap(first,second)
            rigid_rows.append({'shapes':[first['shape'],second['shape']],'bodies':[first['body'],second['body']],
                'direct_joint_connection':tuple(sorted((first['body'],second['body']))) in connected,
                'measurement':kind,**summarize(gap,tolerance)})
    tets=source.get('tet_indices',np.empty((0,4),int));particle_containment=None;degenerate_omissions=0
    if len(tets):
        tetrahedra=points[:,tets]
        inside,degenerate_omissions=tet_containment(points,tetrahedra,tets)
        particle_containment={'initial_strict_interior_vertices':int(inside[0].sum()),
            'maximum_strict_interior_vertices':int(inside.sum(-1).max()),
            'recorded_steps_with_strict_interior_vertices':int(inside.any(-1).sum())}
        for shape in shapes:
            if shape['kind']==7:
                signs=np.array([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)])
                vertices=shape['center'][:,None]+np.einsum('kij,vj->kvi',shape['rotation'],signs*shape['scale'])
                witnesses=np.concatenate((shape['center'][:,None],vertices),axis=1)
            elif shape['kind']==4:witnesses=np.stack((shape['a'],shape['center'],shape['b']),axis=1)
            else:witnesses=shape['center'][:,None]
            inside,_=tet_containment(witnesses,tetrahedra,None)
            contained.append({'shape':shape['shape'],'body':shape['body'],'witnesses_per_state':witnesses.shape[1],
                'initial_strict_interior_witnesses':int(inside[0].sum()),
                'maximum_strict_interior_witnesses':int(inside.sum(-1).max()),
                'recorded_steps_with_strict_interior_witnesses':int(inside.any(-1).sum())})
    return {'schema':'native_sampled_primitive_pairs_v252','physical_roundoff_tolerance_m':tolerance,
        'shape_vertex_gaps':vertex_rows,'shape_triangle_overlaps':surface_rows,'rigid_primitive_pairs':rigid_rows,
        'rigid_witnesses_inside_soft_tets':contained,'deformable_vertices_inside_nonincident_tets':particle_containment,
        'degenerate_tet_state_rows_omitted_from_containment':degenerate_omissions,
        'pair_filter_policy':'All different-body primitives checked. Direct joint neighbors labeled separately, not silently removed. Native collision filter tables are not reconstructed.',
        'geometry_convention':'Rendered zero-thickness triangles and closed physical primitive surfaces. Particle radii reported separately; contact margin/gap is never added to physical dimensions.',
        'gap_semantics':'Negative gaps witness overlap. Positive SAT gaps are lower bounds on Euclidean separation. Capsule-box negative gaps witness overlap but are not full penetration depths when the centerline is inside the box.',
        'unchecked_geometry':['Continuous paths between recorded states','Volume overlap without a sampled vertex/rigid witness or surface intersection',
            'Finite cloth thickness away from vertices','Coplanar deformable/deformable overlap','Nonprimitive rigid meshes'],
        'quality_accepted':False}
