"""Small native Newton asset with all five material/object families."""
import math


def build_asset(newton,wp,*,device='cpu',friction=.35,elastic_scale=1.,mass_scale=1.):
    if any(not math.isfinite(v) for v in (friction,elastic_scale,mass_scale)) or friction<0 or min(elastic_scale,mass_scale)<=0:
        raise ValueError('Finite nonnegative friction and positive elastic/mass scales required')
    builder=newton.ModelBuilder(gravity=(0.,0.,-9.8))
    builder.default_shape_cfg.ke=2000.
    builder.default_shape_cfg.kd=.1
    builder.default_shape_cfg.mu=friction
    builder.default_shape_cfg.margin=.003
    builder.default_shape_cfg.density=1000.*mass_scale
    builder.add_cloth_grid(pos=wp.vec3(0.,0.,.06),rot=wp.quat_identity(),vel=wp.vec3(0.),
        dim_x=2,dim_y=2,cell_x=.1,cell_y=.1,mass=.001*mass_scale,
        tri_ke=300.*elastic_scale,tri_ka=300.*elastic_scale,tri_kd=.1,
        edge_ke=.03*elastic_scale,edge_kd=.01,fix_left=True,particle_radius=.005)
    builder.add_soft_grid(pos=wp.vec3(.025,.025,.09),rot=wp.quat_identity(),vel=wp.vec3(0.),
        dim_x=1,dim_y=1,dim_z=1,cell_x=.04,cell_y=.04,cell_z=.04,density=50.*mass_scale,
        k_mu=100.*elastic_scale,k_lambda=100.*elastic_scale,k_damp=.1,particle_radius=.005)
    cable=newton.Rod.create_straight(start=wp.vec3(-.04,.03,.075),direction=wp.vec3(1.,0.,0.),
        length=.3,segment_count=4,radius=.012)
    rod_bodies,rod_joints=builder.add_rod(rod=cable,stretch_stiffness=1000.*elastic_scale,
        shear_stiffness=1000.*elastic_scale,bend_stiffness=.1*elastic_scale,
        twist_stiffness=.1*elastic_scale,bend_damping=.01,body_frame_origin='start')
    free=builder.add_link(xform=wp.transform((.28,.03,.025),wp.quat_identity()))
    builder.add_shape_box(free,hx=.025,hy=.025,hz=.03)
    free_joint=builder.add_joint_free(free);builder.add_articulation([free_joint])
    hinge=builder.add_link(xform=wp.transform((-.07,.03,.075),wp.quat_identity()))
    builder.add_shape_sphere(hinge,radius=.015)
    hinge_joint=builder.add_joint_revolute(parent=-1,child=hinge,axis=(0.,0.,1.),
        parent_xform=wp.transform((-.07,.03,.075),wp.quat_identity()))
    builder.add_articulation([hinge_joint])
    builder.add_ground_plane();builder.color()
    asset=builder.finalize(device=device)
    asset.soft_contact_ke,asset.soft_contact_kd,asset.soft_contact_mu=100.,.1,friction
    return asset,{'rod_bodies':list(rod_bodies),'rod_source_joints':list(rod_joints),
                  'free_body':free,'hinge_body':hinge,'hinge_source_joint':hinge_joint,
                  'friction':friction,'elastic_scale':elastic_scale,'mass_scale':mass_scale}
