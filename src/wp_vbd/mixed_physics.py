"""Finite elastic mixed residuals and coupled 6D blocks, without a solver.

Contract: finite-elastic joints/rod, regularized Coulomb plane contact. The
offline teacher explicitly selects the corresponding VBD soft-penalty mode.
This is not the default ALM contact/drive/history contract.
"""

import torch

from . import math3d as so3
from .data import State
from .physics import elastic_nodes
from .response_operator import skew
from .attachments import particle_body_attachment


def attachment_system(scene,state):
    pairs=scene.attachments
    evaluation=particle_body_attachment(state.x[:,pairs[:,0]],state.x[:,pairs[:,1]],state.q[:,pairs[:,1]],scene.attachment_anchor,scene.attachment_stiffness)
    residual=state.x.new_zeros(state.x.shape[0],scene.n,6).index_add(1,pairs.flatten(),evaluation.residual.flatten(1,2))
    diagonal=state.x.new_zeros(state.x.shape[0],scene.n,6,6).index_add(1,pairs.flatten(),evaluation.diagonal.flatten(1,2))
    return residual,diagonal,evaluation.gap,evaluation.jacobian


def rod_measure(parent, child):
    rp,rc=so3.matrix(parent),so3.matrix(child)
    t0,t1=rp[...,2],rc[...,2]
    m0,m1=rp[...,0],rc[...,0]
    denominator=(1+(t0*t1).sum(-1)).clamp_min(1e-8)
    curvature=2*torch.linalg.cross(t0,t1)/denominator[...,None]
    curvature=curvature*(20/(curvature.square().sum(-1,keepdim=True)+1e-20).sqrt()).clamp_max(1)
    transported=so3.normalize(m0-((m0*t1).sum(-1)/denominator)[...,None]*(t0+t1))
    twist=torch.atan2((t1*torch.linalg.cross(transported,m1)).sum(-1),(transported*m1).sum(-1))
    bend=so3.rotate(so3.conjugate(parent),curvature)
    return torch.cat((bend[...,:2],twist[...,None]),-1)


def rod_domain(scene,state):
    if not len(scene.joints):return torch.ones(state.x.shape[0],device=state.x.device,dtype=torch.bool)
    qa=so3.multiply(state.q[:,scene.joints[:,0]],scene.anchor_qp)
    qb=so3.multiply(state.q[:,scene.joints[:,1]],scene.anchor_qc)
    cosine=(so3.matrix(qa)[...,2]*so3.matrix(qb)[...,2]).sum(-1)
    return ((cosine>-.9)|(scene.joint_tag!=1)[None]).all(-1)


def joint_coordinates(scene, pair_x, pair_q):
    qa=so3.multiply(pair_q[:,:,0],scene.anchor_qp)
    qb=so3.multiply(pair_q[:,:,1],scene.anchor_qc)
    pa=pair_x[:,:,0]+so3.rotate(pair_q[:,:,0],scene.anchor_p)
    pb=pair_x[:,:,1]+so3.rotate(pair_q[:,:,1],scene.anchor_c)
    gap=so3.rotate(so3.conjugate(qa),pb-pa)
    angular=so3.log(so3.multiply(so3.multiply(so3.conjugate(qa),qb),so3.conjugate(scene.joint_rest_q)))
    angular=torch.where((scene.joint_tag==0)[None,:,None],angular-(angular*scene.joint_axis).sum(-1,keepdim=True)*scene.joint_axis,angular)
    rod=rod_measure(qa,qb)-scene.rod_rest
    rod=torch.cat((rod[...,:2],torch.atan2(torch.sin(rod[...,2:]),torch.cos(rod[...,2:]))),-1)
    angular=torch.where((scene.joint_tag==1)[None,:,None],rod,angular)
    return torch.cat((gap,angular),-1)


def joint_system(scene,state,linearize=True,batched=True):
    if not len(scene.joints):
        gradient=state.x.new_zeros(state.x.shape[0],scene.n,6)
        blocks=state.x.new_zeros(state.x.shape[0],scene.n,6,6)
        return gradient,blocks,state.x.new_empty(state.x.shape[0],0,6),state.x.new_empty(state.x.shape[0],0,6,2,6)
    enabled=torch.is_grad_enabled()
    with torch.enable_grad():
        delta=state.x.new_zeros(state.x.shape[0],len(scene.joints),2,6,requires_grad=True)
        x=state.x[:,scene.joints]+delta[...,:3]
        q=so3.multiply(so3.exp(delta[...,3:]),state.q[:,scene.joints])
        coordinates=joint_coordinates(scene,x,q)
        if not linearize:
            energy=.5*(coordinates.square()*scene.joint_stiffness).sum()
            force=torch.autograd.grad(energy,delta,create_graph=enabled)[0]
            if not enabled:force=force.detach()
            gradient=force.new_zeros(force.shape[0],scene.n,6).index_add(1,scene.joints.flatten(),force.flatten(1,2))
            return gradient,None,None,None
        if batched:
            basis=torch.eye(6,device=delta.device,dtype=delta.dtype)[:,None,None,:].expand(6,*coordinates.shape)
            derivative=torch.autograd.grad(coordinates,delta,grad_outputs=basis,retain_graph=True,create_graph=enabled,is_grads_batched=True)[0]
            jacobian=derivative.permute(1,2,0,3,4)
        else:
            rows=[torch.autograd.grad(coordinates[...,i].sum(),delta,retain_graph=True,create_graph=enabled)[0] for i in range(6)]
            jacobian=torch.stack(rows,2)
    if not enabled:coordinates,jacobian=coordinates.detach(),jacobian.detach()
    force=torch.einsum('brdsu,brd->brsu',jacobian,coordinates*scene.joint_stiffness)
    j=jacobian.detach()
    metric=torch.einsum('brdsi,rd,brdsj->brsij',j,scene.joint_stiffness,j)
    gradient=force.new_zeros(force.shape[0],scene.n,6).index_add(1,scene.joints.flatten(),force.flatten(1,2))
    blocks=metric.new_zeros(metric.shape[0],scene.n,6,6).index_add(1,scene.joints.flatten(),metric.flatten(1,2))
    return gradient,blocks,coordinates,jacobian.detach()


def contact_kinematics(scene,state,start):
    owner=scene.site_owner
    arm=so3.rotate(state.q[:,owner],scene.site_local)
    point=state.x[:,owner]+arm
    normal=scene.axis_z
    offset=-scene.site_radius[:,None]*normal
    # PRE_INIT contact anchors are initialized from the PHYSICAL STEP START
    # and transported with the body throughout every inner iteration. Re-
    # creating the local offset from the current pose changes rolling friction.
    local_offset=so3.rotate(so3.conjugate(start.q[:,owner]),offset)
    old=start.x[:,owner]+so3.rotate(start.q[:,owner],scene.site_local+local_offset)
    surface=state.x[:,owner]+so3.rotate(state.q[:,owner],scene.site_local+local_offset)
    slip=surface-old
    distance=point[...,2]-scene.site_radius-scene.plane_height
    selected=torch.ones_like(distance,dtype=torch.bool)
    for ids in scene.box_sites:
        # Exactly the four deepest corners used by Newton plane-box contact.
        order=torch.argsort(distance[:,ids],dim=1,stable=True)
        chosen=torch.zeros_like(order,dtype=torch.bool).scatter(1,order[:,:4],True)
        selected[:,ids]=chosen
    return arm,point,slip,distance,selected


def plane_system(scene,state,start,dt,forecast=None):
    arm,point,slip,distance,selected=contact_kinematics(scene,state,start)
    penetration=(scene.site_margin-distance).clamp_min(0)
    active=(penetration>0)&selected
    metric_active=active
    if forecast is not None:
        future=forecast.x[:,scene.site_owner]+so3.rotate(forecast.q[:,scene.site_owner],scene.site_local)
        future_distance=future[...,2]-scene.site_radius-scene.plane_height
        # A contact can activate within the proposed physical step. Its solve
        # metric must anticipate that stiffness even while its physical force
        # is still zero. The forecast is NEVER used as an unchecked state.
        metric_active=active|(future_distance<scene.site_margin)
    normal_load=scene.contact_ke*penetration*selected
    old_axis=start.x[:,scene.site_owner]+so3.rotate(start.q[:,scene.site_owner],scene.site_local)
    dn=point[...,2]-old_axis[...,2]
    damping=scene.contact_kd/dt*active*(dn<0)
    normal=scene.axis_z
    tangent=slip*scene.axis_xy
    norm=(tangent.square().sum(-1)+1e-30).sqrt()
    eps=scene.friction_epsilon*dt
    factor=torch.where(norm>eps,1/norm,(2-norm/eps)/eps)
    scale=scene.site_mu*normal_load*factor*(norm>1e-15)
    normal_force=(normal_load-damping*dn)[...,None]*normal
    tangent_force=-scale[...,None]*tangent
    eye=torch.eye(3,device=point.device,dtype=point.dtype)
    jn=torch.cat((eye.expand(*arm.shape[:-1],3,3),-skew(arm)),-1)
    offset=so3.rotate(so3.conjugate(start.q[:,scene.site_owner]),-scene.site_radius[:,None]*normal)
    surface_arm=so3.rotate(state.q[:,scene.site_owner],scene.site_local+offset)
    jt=torch.cat((eye.expand(*arm.shape[:-1],3,3),-skew(surface_arm)),-1)
    gradient=-torch.einsum('bsdi,bsd->bsi',jn,normal_force)-torch.einsum('bsdi,bsd->bsi',jt,tangent_force)
    kn=(scene.contact_ke*metric_active+damping)[...,None,None]*torch.diag(scene.axis_z)
    kt=scale[...,None,None]*torch.diag(scene.axis_xy)
    block=jn.transpose(-1,-2)@kn@jn+jt.transpose(-1,-2)@kt@jt
    gradient=point.new_zeros(point.shape[0],scene.n,6).index_add(1,scene.site_owner,gradient)
    blocks=point.new_zeros(point.shape[0],scene.n,6,6).index_add(1,scene.site_owner,block)
    response_jacobian=torch.cat((jt[...,:2,:],jn[...,2:,:]),-2)
    return gradient,blocks,response_jacobian,slip,distance,active


def predictor(scene,state,dt,force=None,torque=None):
    force=torch.zeros_like(state.x) if force is None else force
    torque=torch.zeros_like(state.omega) if torque is None else torque
    target_x=state.x+dt*state.v+dt**2*(scene.gravity+force/scene.mass.clamp_min(1e-8)[None,:,None])
    wb=so3.rotate(so3.conjugate(state.q),state.omega)
    tb=so3.rotate(so3.conjugate(state.q),torque)-torch.linalg.cross(wb,(scene.inertia@wb[...,None]).squeeze(-1))
    regularizer=(~scene.movable).to(state.x.dtype)[:,None,None] if getattr(scene,'use_exact_inertia',False) else 1e-9
    dw=torch.linalg.solve_ex(scene.inertia+regularizer*torch.eye(3,device=state.x.device),tb[...,None],check_errors=False).result.squeeze(-1)
    omega=so3.rotate(state.q,wb+dt*dw)
    derivative=so3.multiply(torch.cat((omega,torch.zeros_like(omega[...,:1])),-1),state.q)
    target_q=so3.normalize(state.q+.5*dt*derivative)
    return torch.where(scene.movable[None,:,None],target_x,state.x),torch.where((scene.movable&scene.body)[None,:,None],target_q,state.q)


def mass_whitening(scene,state,dt):
    rotation=so3.matrix(state.q)
    angular=dt*rotation@scene.inverse_inertia_sqrt@rotation.transpose(-1,-2)
    b=state.x.shape[0]
    w=state.x.new_zeros(b,scene.n,6,6)
    w[...,:3,:3]=dt/scene.mass.clamp_min(1e-8).sqrt()[None,:,None,None]*torch.eye(3,device=state.x.device)
    w[...,3:,3:]=angular
    return w*scene.dof[None,:,:,None]*scene.dof[None,:,None,:]


def residual_system(scene,state,start,target_x,target_q,dt,linearize=True,batched=True,return_metric_base=False):
    m=scene.mass/dt**2
    residual=state.x.new_zeros(*state.x.shape[:-1],6)
    residual[...,:3]=m[None,:,None]*(state.x-target_x)
    rotation=so3.matrix(state.q)
    inertia=rotation@scene.inertia@rotation.transpose(-1,-2)/dt**2
    theta=so3.log(so3.multiply(state.q,so3.conjugate(target_q)))
    residual[...,3:]=(inertia@theta[...,None]).squeeze(-1)
    blocks=state.x.new_zeros(*state.x.shape[:-1],6,6)
    blocks[...,:3,:3]=m[None,:,None,None]*torch.eye(3,device=state.x.device)
    blocks[...,3:,3:]=inertia
    metric_base=blocks.clone() if return_metric_base else None
    elastic,elastic_metric=elastic_nodes(scene,state.x)
    damping_terms=[]
    if any(scene.material_damping_flags):
        from .material_damping import damping_system
        viscous,viscous_metric,damping_terms=damping_system(scene,state.x,start.x,dt)
        elastic=elastic+viscous;elastic_metric=elastic_metric+viscous_metric
    residual[...,:3]=residual[...,:3]+elastic
    blocks[...,:3,:3]=blocks[...,:3,:3]+elastic_metric
    joint,joint_metric,coordinates,jacobian=joint_system(scene,state,linearize,batched)
    attached=attachment_metric=None
    attachment_gap=state.x.new_empty(state.x.shape[0],0,3)
    attachment_j=state.x.new_empty(state.x.shape[0],0,3,2,6)
    if len(scene.attachments):attached,attachment_metric,attachment_gap,attachment_j=attachment_system(scene,state)
    forecast=State(target_x,target_q,start.v,start.omega)
    contact,contact_metric,point_j,slip,distance,active=plane_system(scene,state,start,dt,forecast)
    residual=residual+joint+contact
    if attached is not None:residual=residual+attached
    residual=residual*scene.dof[None]
    if not linearize:return residual,None,None
    blocks=blocks+joint_metric+contact_metric
    if attachment_metric is not None:blocks=blocks+attachment_metric
    result=(residual,blocks,(coordinates,jacobian,point_j,slip,distance,active,attachment_gap,attachment_j,damping_terms))
    return (*result,metric_base+contact_metric) if return_metric_base else result


def energy_owner(scene,ids):
    """Assign a structural potential to a participating mobile endpoint.

    Mixed contact graphs keep fixed nodes outside dynamic components. Their
    incident material energy must still enter the mobile component's trial
    merit. Older static-object backends retain their existing distribution.
    """
    if not getattr(scene,'redistribute_structural_energy',False):return ids[:,0]
    mobile=scene.movable[ids]
    return ids.gather(1,mobile.long().argmax(-1,keepdim=True)).squeeze(-1)


def incremental_objective(scene,state,start,target_x,target_q,dt,lagged_load):
    """Per-entity incremental model with Coulomb normal loads frozen per trial.

    This is a declared local acceptance surrogate. Friction is not recast as
    a global conservative physical energy, and loads refresh next unroll.
    """
    if hasattr(scene,'objective_scene'):
        # Tiny equilibrium improvements are below FP32 objective cancellation
        # at world coordinates near one metre. Keep dynamics in FP32, evaluate
        # acceptance in FP64 rather than silently relaxing the merit threshold.
        cast=lambda s:State(s.x.double(),s.q.double(),s.v,s.omega)
        return incremental_objective(scene.objective_scene,cast(state),cast(start),target_x.double(),target_q.double(),dt,lagged_load.double())
    energy=.5*scene.mass[None]/dt**2*(state.x-target_x).square().sum(-1)
    if any(scene.material_damping_flags):
        from .material_damping import damping_potential
        energy=energy+damping_potential(scene,state.x,start.x,dt)
    angle=so3.log(so3.multiply(so3.conjugate(state.q),target_q))
    energy=energy+.5*(angle*(scene.inertia@angle[...,None]).squeeze(-1)).sum(-1)/dt**2*scene.body
    for key in ('tets','triangles'):
        ids=getattr(scene,key)
        if not len(ids):continue
        dm=scene.tet_Dm if key=='tets' else scene.tri_Dm
        mat=scene.tet_material if key=='tets' else scene.tri_material
        p=state.x[:,ids];f=(p[:,:,1:]-p[:,:,:1]).transpose(-1,-2)@dm
        mu=mat[:,0];lam=mat[:,:2].sum(-1)
        if key=='tets':
            determinant=(f[...,0]*so3.cofactor(f)[...,0]).sum(-1)
            measure=scene.tet_volume
        else:
            determinant=(torch.linalg.cross(f[...,0],f[...,1]).square().sum(-1)+1e-20).sqrt()
            measure=scene.tri_area
        term=measure*(.5*mu*f.square().sum((-1,-2))+.5*lam*(determinant-1-mu/lam.clamp_min(1e-20)).square())
        energy=energy.index_add(1,energy_owner(scene,ids),term)
    ids=scene.bend_edges[scene.bending_indices]
    if len(ids):
        p=state.x[:,ids]
        n1=so3.normalize(torch.linalg.cross(p[:,:,2]-p[:,:,0],p[:,:,3]-p[:,:,0]))
        n2=so3.normalize(torch.linalg.cross(p[:,:,3]-p[:,:,1],p[:,:,2]-p[:,:,1]))
        edge=so3.normalize(p[:,:,3]-p[:,:,2])
        theta=torch.atan2((torch.linalg.cross(n1,n2)*edge).sum(-1),(n1*n2).sum(-1))
        valid=scene.bending_indices
        from .bending_angle import difference
        term=.5*scene.bend_material[valid,0]*scene.bend_length[valid]*difference(scene,theta,scene.bend_angle[valid]).square()
        energy=energy.index_add(1,energy_owner(scene,ids),term)
    c=joint_coordinates(scene,state.x[:,scene.joints],state.q[:,scene.joints])
    term=.5*(scene.joint_stiffness*c.square()).sum(-1)
    energy=energy.index_add(1,scene.joints[:,1],term)
    if len(scene.attachments):
        pairs=scene.attachments
        attachment=particle_body_attachment(state.x[:,pairs[:,0]],state.x[:,pairs[:,1]],state.q[:,pairs[:,1]],scene.attachment_anchor,scene.attachment_stiffness)
        energy=energy.index_add(1,pairs[:,1],attachment.energy)
    _,_,slip,distance,selected=contact_kinematics(scene,state,start)
    normal=.5*scene.contact_ke*(scene.site_margin-distance).clamp_min(0).square()*selected
    norm=(slip[...,:2].square().sum(-1)+1e-30).sqrt();eps=scene.friction_epsilon*dt
    primitive=torch.where(norm>eps,norm-eps/3,norm.square()/eps-norm.pow(3)/(3*eps**2))
    friction=scene.site_mu*lagged_load*primitive
    old_axis=start.x[:,scene.site_owner]+so3.rotate(start.q[:,scene.site_owner],scene.site_local)
    arm=so3.rotate(state.q[:,scene.site_owner],scene.site_local)
    delta_n=state.x[:,scene.site_owner,2]+arm[...,2]-old_axis[...,2]
    damping=.5*scene.contact_kd/dt*delta_n.clamp_max(0).square()*(lagged_load>0)
    energy=energy.index_add(1,scene.site_owner,normal+friction+damping)
    return energy
