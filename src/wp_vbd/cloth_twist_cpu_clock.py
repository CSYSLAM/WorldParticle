"""Uniform source rotation time for CPU-native cloth-twist controls."""
import numpy as np
import warp as wp


@wp.kernel
def rotate_at_time(indices:wp.array[wp.int32],axes:wp.array[wp.vec3],roots:wp.array[wp.vec3],
    arms:wp.array[wp.vec3],time:float,angular_velocity:float,end_time:float,
    output0:wp.array[wp.vec3],output1:wp.array[wp.vec3]):
    if time>end_time:return
    i=wp.tid();axis=axes[i];ux=axis[0];uy=axis[1];uz=axis[2];theta=time*angular_velocity
    cosine=wp.cos(theta);sine=wp.sin(theta)
    rotation=wp.mat33(cosine+ux*ux*(1.0-cosine),ux*uy*(1.0-cosine)-uz*sine,ux*uz*(1.0-cosine)+uy*sine,
        uy*ux*(1.0-cosine)+uz*sine,cosine+uy*uy*(1.0-cosine),uy*uz*(1.0-cosine)-ux*sine,
        uz*ux*(1.0-cosine)-uy*sine,uz*uy*(1.0-cosine)+ux*sine,cosine+uz*uz*(1.0-cosine))
    position=roots[i]+rotation*arms[i]
    output0[indices[i]]=position;output1[indices[i]]=position


def install(example):
    if not example.model.device.is_cpu:raise ValueError('CPU-native controller adapter only')
    def simulate():
        example.collision_pipeline.collide(example.state_0,example.contacts)
        example.solver.rebuild_bvh(example.state_0)
        for _ in range(example.sim_substeps):
            example.state_0.clear_forces();example.viewer.apply_forces(example.state_0)
            phase=np.float32(example.t.numpy()[0])
            wp.launch(rotate_at_time,dim=len(example.rot_point_indices),inputs=[example.rot_point_indices,example.rot_axes,
                example.roots,example.roots_to_ps,float(phase),example.rot_angular_velocity,example.rot_end_time],
                outputs=[example.state_0.particle_q,example.state_1.particle_q])
            if phase<=example.rot_end_time:example.t.assign(np.array([phase+np.float32(example.sim_dt)],dtype=np.float32))
            example.solver.step(example.state_0,example.state_1,example.control,example.contacts,example.sim_dt)
            example.state_0,example.state_1=example.state_1,example.state_0
    example.simulate=simulate
    return {'kind':'uniform_scalar_source_phase_cpu','angular_velocity':example.rot_angular_velocity,
        'end_time':example.rot_end_time,'scope':'Same Rodrigues rotation and once-per-substep FP32 clock. Scalar phase prevents the source CPU loop from observing its own time update at different vertex indices. No dynamics/force/iteration changes.'}
