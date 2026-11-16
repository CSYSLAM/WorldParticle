"""Offline measurements from native geometry, with explicit incomplete scope."""
import numpy as np
from scipy.spatial.transform import Rotation


def transverse_crossings(points,faces):
    pairs=np.array([(a,b) for a in range(len(faces)) for b in range(a+1,len(faces))
                    if not set(faces[a])&set(faces[b])],dtype=int).reshape(-1,2)
    if not len(pairs):return np.zeros(len(points),dtype=int),0
    hit=np.zeros((len(points),len(pairs)),dtype=bool)
    for first,second in ((0,1),(1,0)):
        a=points[:,faces[pairs[:,first]]];b=points[:,faces[pairs[:,second]]]
        direction=np.roll(a,-1,axis=2)-a
        edge1=b[:,:,1]-b[:,:,0];edge2=b[:,:,2]-b[:,:,0]
        h=np.cross(direction,edge2[:,:,None]);det=np.einsum('kpei,kpi->kpe',h,edge1)
        good=np.abs(det)>1e-16;inverse=np.divide(1.,det,out=np.zeros_like(det),where=good)
        s=a-b[:,:,None,0];u=inverse*np.einsum('kpei,kpei->kpe',s,h)
        q=np.cross(s,edge1[:,:,None]);v=inverse*np.einsum('kpei,kpei->kpe',direction,q)
        t=inverse*np.einsum('kpi,kpei->kpe',edge2,q)
        hit|=(good&(u>1e-8)&(v>1e-8)&(u+v<1-1e-8)&(t>1e-8)&(t<1-1e-8)).any(-1)
    return hit.sum(-1),len(pairs)


def measure(source,points,pose,dt):
    if points.ndim!=3 or points.shape[-1]!=3 or pose.ndim!=3 or pose.shape[-1]!=7 or len(pose)!=len(points) or not len(points):
        raise ValueError('Contiguous native particle/body pose arrays required')
    if not np.isfinite(points).all() or not np.isfinite(pose).all():raise ValueError('Nonfinite native poses')
    types=source['shape_type'];bodies=source['shape_body'];scale=source['shape_scale'].astype(float)
    local=source['shape_transform'].astype(float);ground=np.flatnonzero((types==1)&(bodies==-1))
    if len(ground)!=1 or not np.array_equal(local[ground[0],[0,1,3,4,5,6]],np.array([0,0,0,0,0,1])):
        raise ValueError('One authored horizontal plane is required for this audit')
    height=local[ground[0],2]
    rotation=(Rotation.from_quat(pose[:,:,3:].reshape(-1,4)).as_matrix().reshape(*pose.shape[:2],3,3)
              if pose.shape[1] else np.empty((len(pose),0,3,3)))
    gaps=[]
    for shape,body in enumerate(bodies):
        if shape in ground:continue
        if body<0 or types[shape] not in (3,4,7):raise ValueError('Unsupported authored ground-gap shape')
        center=pose[:,body,:3]+np.einsum('kij,j->ki',rotation[:,body],local[shape,:3])
        world=rotation[:,body]@Rotation.from_quat(local[shape,3:]).as_matrix()
        if types[shape]==3:support=np.full(len(points),scale[shape,0])
        elif types[shape]==4:support=scale[shape,0]+np.abs(world[:,2,2])*scale[shape,1]
        else:support=(np.abs(world[:,2])*scale[shape]).sum(-1)
        separation=center[:,2]-support-height
        gaps.append({'shape':shape,'body':int(body),'initial_surface_ground_separation_m':float(separation[0]),
            'minimum_recorded_surface_ground_separation_m':float(separation.min()),
            'minimum_after_first_step_surface_ground_separation_m':float(separation[1:].min()) if len(points)>1 else None,
            'maximum_recorded_ground_penetration_m':float(np.maximum(-separation,0).max())})
    particle=None
    if points.shape[1]:
        gap=points[:,:,2]-source['particle_radius'][None]-height
        particle={'initial_minimum_m':float(gap[0].min()),'all_recorded_minimum_m':float(gap.min()),
            'after_first_step_minimum_m':float(gap[1:].min()) if len(points)>1 else None,
            'convention':'Authored particle radius subtracted from vertex height; rendered triangle surface heights differ by that radius.'}
    tets=source.get('tet_indices',np.empty((0,4),int));rest=source.get('particle_q',np.empty((0,3))).astype(float)
    tetrahedra=None
    if len(tets):
        dm=(rest[tets[:,1:]]-rest[tets[:,:1]]).transpose(0,2,1)
        ds=(points[:,tets[:,1:]]-points[:,tets[:,:1]]).transpose(0,1,3,2)
        deformation=ds@np.linalg.inv(dm);det=np.linalg.det(deformation);stretch=np.linalg.svd(deformation,compute_uv=False)
        tetrahedra={'minimum_determinant':float(det.min()),'maximum_determinant':float(det.max()),
            'inverted_recorded_tets':int((det<=0).sum()),'minimum_principal_stretch':float(stretch.min()),'maximum_principal_stretch':float(stretch.max())}
    faces=source.get('tri_indices',np.empty((0,3),int));soft=set(tets.flatten())
    cloth=faces[[not any(int(i) in soft for i in face) for face in faces]] if len(faces) else faces
    membrane=None
    if len(cloth):
        reference=(rest[cloth[:,1:]]-rest[cloth[:,:1]]).transpose(0,2,1)
        gram=reference.transpose(0,2,1)@reference;eigen,vectors=np.linalg.eigh(gram)
        inverse_root=(vectors/np.sqrt(eigen)[:,None,:])@vectors.transpose(0,2,1)
        current=(points[:,cloth[:,1:]]-points[:,cloth[:,:1]]).transpose(0,1,3,2)
        stretch=np.linalg.svd(current@inverse_root,compute_uv=False)
        membrane={'minimum_principal_stretch':float(stretch.min()),'maximum_principal_stretch':float(stretch.max()),
            'minimum_area_ratio':float(stretch.prod(-1).min()),'maximum_area_ratio':float(stretch.prod(-1).max())}
    counts,pair_count=transverse_crossings(points,faces)
    return {'body_surface_ground_gaps':gaps,'particle_radius_surface_ground_gap':particle,
        'tet_deformation':tetrahedra,'cloth_deformation':membrane,
        'strict_transverse_triangle_crossings':{'checked_nonadjacent_face_pairs':pair_count,
            'recorded_steps_with_crossings':int((counts>0).sum()),
            'maximum_pairs_crossing_in_one_recorded_state':int(counts.max()),'initial_pairs_crossing':int(counts[0])},
        'maximum_particle_finite_difference_speed_m_s':float(np.linalg.norm(np.diff(points,axis=0)/dt,axis=-1).max()) if len(points)>1 and points.shape[1] else None,
        'unchecked_geometry':['Continuous paths between recorded states','Coplanar/edge-only intersections and containment',
                              'Deformable/rigid and rigid/rigid pair intersections beyond ground']}
