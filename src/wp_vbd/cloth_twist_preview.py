"""Explicit resolution variant of the native cloth-twist asset and controller."""
import numpy as np


def resize_example(example,resolution,wp,newton):
    if type(resolution) is not int or not 4<=resolution<50:
        raise ValueError('Explicit preview grid in [4,49] required')
    from newton.examples.cloth.example_cloth_twist import initialize_rotation
    source=example.model;rest=source.particle_q.numpy()
    if rest.shape!=(2500,3):raise ValueError('Original 50x50 cloth template required')
    grid=rest.reshape(50,50,3);s=np.linspace(0.,1.,resolution)
    points=np.stack([(1-a)*((1-b)*grid[0,0]+b*grid[0,-1])+a*((1-b)*grid[-1,0]+b*grid[-1,-1])
                     for a in s for b in s]).astype(np.float32)
    faces=[]
    for i in range(resolution-1):
        for j in range(resolution-1):
            a=i*resolution+j;b=a+1;c=a+resolution;d=c+1
            faces.extend(((a,d,b),(a,c,d)) if (i+j)%2==0 else ((a,c,b),(c,d,b)))
    faces=np.asarray(faces,dtype=np.int32)
    original_faces=example.faces
    area=np.linalg.norm(np.cross(rest[original_faces[:,1]]-rest[original_faces[:,0]],rest[original_faces[:,2]]-rest[original_faces[:,0]]),axis=-1).sum()/2
    density=float(source.particle_mass.numpy().sum()/area)
    tri=source.tri_materials.numpy()[0];bend=source.edge_bending_properties.numpy()[0]
    builder=newton.ModelBuilder(gravity=tuple(source.gravity.numpy()[0]))
    builder.add_cloth_mesh(pos=wp.vec3(0.),rot=wp.quat_identity(),scale=1.,vel=wp.vec3(0.),vertices=[wp.vec3(*p) for p in points],
        indices=faces.flatten().tolist(),density=density,tri_ke=float(tri[0]),tri_ka=float(tri[1]),tri_kd=float(tri[2]),
        tri_drag=float(tri[3]),tri_lift=float(tri[4]),edge_ke=float(bend[0]),edge_kd=float(bend[1]),
        particle_radius=float(source.particle_radius.numpy()[0]))
    builder.color();model=builder.finalize()
    for field in ['soft_contact_ke','soft_contact_kd','soft_contact_mu']:setattr(model,field,getattr(source,field))
    left=[resolution-1+i*resolution for i in range(resolution)];right=[i*resolution for i in range(resolution)]
    indices=left+right;flags=model.particle_flags.numpy();flags[indices]&=~int(newton.ParticleFlags.ACTIVE)
    model.particle_flags=wp.array(flags,dtype=model.particle_flags.dtype,device=model.device)
    example.model=model;example.faces=faces;example.state_0=model.state();example.state_1=model.state();example.control=model.control()
    example.collision_pipeline=newton.CollisionPipeline(model);example.contacts=example.collision_pipeline.contacts()
    example.rot_point_indices=wp.array(indices,dtype=wp.int32);example.t=wp.zeros(1,dtype=wp.float32)
    example.rot_centers=wp.zeros(len(indices),dtype=wp.vec3)
    example.rot_axes=wp.array([[0.,1.,0.]]*resolution+[[0.,-1.,0.]]*resolution,dtype=wp.vec3)
    example.roots=wp.zeros_like(example.rot_centers);example.roots_to_ps=wp.zeros_like(example.rot_centers)
    wp.launch(initialize_rotation,dim=len(indices),inputs=[example.rot_point_indices,example.state_0.particle_q,
        example.rot_centers,example.rot_axes,example.t],outputs=[example.roots,example.roots_to_ps])
    example.viewer.set_model(model)
    return {'resolution':resolution,'particles':resolution**2,'triangles':len(faces),'driven_vertices':len(indices),
        'original_particles':2500,'density':density,'original_density_inferred_from_total_mass_and_area':True,
        'angular_velocity':example.rot_angular_velocity,'rotation_end_seconds':example.rot_end_time,
        'scope':'Explicit lower resolution preview; same extent, material coefficients, density, 60Hz frame clock and native rotation controller. Checkerboard triangle grid; not the original2500-particle asset.'}
