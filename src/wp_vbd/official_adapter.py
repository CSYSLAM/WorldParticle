"""Fail-closed import of authored Newton scenes into the shared state schema.

The material contract contains objective tet/shell damping and free rigid
inertia; the separate pure-rod contract carries compliant ALM/Dahl history.
Self-contact, springs, drives and rigid contacts need their own operators
before admission; they are never silently turned off here.
"""

import numpy as np

from .adapters import export_static


ADMITTED=(
    'softbody.example_softbody_hanging',
    'vbd.example_cloth_stiff_material_hanging',
    'vbd.example_cloth_stiff_material_stretch',
    'basic.example_basic_dzhanibekov',
    'vbd.example_cable_bend_stiffness',
    'vbd.example_cable_bend_twist_analytic',
    'vbd.example_cable_dahl_hysteresis',
    'vbd.example_cable_michell_threshold',
    'vbd.example_cable_torsion_material_mapping',
    'vbd.example_cable_twist_transfer',
)


def export_official(example,configuration,identifier,newton):
    if identifier not in ADMITTED:raise ValueError(f'{identifier}: physical operators not yet admitted')
    if identifier.startswith('vbd.example_cable_'):
        from .rod_adapter import export_model
        return export_model(example.model, configuration, identifier, newton, example.sim_dt)
    model=example.model
    if configuration['particle_enable_self_contact'] or configuration['rigid_contact_history'] or configuration['integrate_with_external_rigid_solver']:
        raise ValueError('Self-contact, persistent contact history or solver coupling requires a new operator')
    if model.spring_count or model.attachment_particle_body_count:raise ValueError('Unsupported additional relations')
    if model.joint_count and np.any(model.joint_type.numpy()!=int(newton.JointType.FREE)):
        raise ValueError('Default ALM joint state is not a finite elastic joint')
    gravity=model.gravity.numpy().reshape(-1,3)
    if len(gravity)!=1:raise ValueError('Multi-world gravity requires an explicit adapter')
    node=export_static('softbody',model,{},gravity[0],newton)
    body=export_static('joint',model,{},gravity[0],newton)
    shared=('rest_x','rest_q','mass','movable','inertia','kind','object_id')
    values={key:np.concatenate((node[key],body[key])) for key in shared}
    values.update({key:value for key,value in node.items() if key not in shared})
    values.update({key:value for key,value in body.items() if key not in shared})
    n=model.particle_count;b=model.body_count;total=n+b
    values['body_com']=np.concatenate((np.zeros((n,3),np.float32),body['body_com']))
    values['joints']=body['joints']+n
    values['joint_stiffness']=np.empty((0,6),np.float32)
    if identifier=='vbd.example_cloth_stiff_material_stretch':values['kinematic_ids']=np.where(~values['movable'])[0].astype(np.int64)
    values.update(attachments=np.empty((0,2),np.int64),attachment_anchor=np.empty((0,3),np.float32),attachment_stiffness=np.empty(0,np.float32))
    values['kind'][:n]=1
    if model.tet_count:values['kind'][np.unique(values['tets'])]=0
    # Connectivity, not fixture-specific group numbers or proximity, defines
    # independent objects. Fixed mesh vertices belong to their own mesh.
    parent=list(range(total))
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    for key in ('tets','triangles'):
        for row in values[key]:
            a=root(int(row[0]))
            for v in row[1:]:parent[root(int(v))]=a
    labels={};groups=[]
    for i in range(total):
        r=root(i)
        if r not in labels:
            label=len(labels);labels[r]=label
            family={0:'softbody',1:'cloth',2:'rigid'}[int(values['kind'][i])]
            groups.append({'name':f'{family}_{label}','family':family})
        values['object_id'][i]=labels[r]
    shape_flags=model.shape_flags.numpy();shape_types=model.shape_type.numpy();shape_body=model.shape_body.numpy()
    collide=(int(newton.ShapeFlags.COLLIDE_SHAPES) if b else 0)|(int(newton.ShapeFlags.COLLIDE_PARTICLES) if n else 0)
    active=np.where((shape_flags&collide)!=0)[0]
    if any(int(shape_types[s])!=int(newton.GeoType.PLANE) or shape_body[s]>=0 for s in active):
        raise ValueError('Active non-plane colliders require another geometry adapter')
    if len(active)>1:raise ValueError('Multiple planes not yet supported')
    owners=np.arange(n,dtype=np.int64) if len(active) else np.empty(0,np.int64)
    plane_height=0.;margin=np.zeros(len(owners),np.float32);mu=margin.copy();ke=kd=0.
    if len(active):
        s=int(active[0]);xf=model.shape_transform.numpy()[s]
        from .adapters import np_rotate
        if np.linalg.norm(np_rotate(xf[3:],np.array([0.,0.,1.]))-np.array([0,0,1]))>1e-6:
            raise ValueError('Only a Z-up plane is admitted')
        plane_height=float(xf[2]);margin=model.particle_radius.numpy()+float(model.shape_margin.numpy()[s]) if n else margin
        mu=np.full(len(owners),np.sqrt(model.soft_contact_mu*model.shape_material_mu.numpy()[s]),np.float32)
        ke=.5*(model.soft_contact_ke+model.shape_material_ke.numpy()[s]);kd=.5*(model.soft_contact_kd+model.shape_material_kd.numpy()[s])
    values.update(site_owner=owners,site_local=np.zeros((len(owners),3),np.float32),site_radius=np.zeros(len(owners),np.float32),site_margin=margin,site_mu=mu,plane_height=np.asarray(plane_height,np.float32),contact_ke=np.asarray(ke,np.float32),contact_kd=np.asarray(kd,np.float32),friction_epsilon=np.asarray(configuration['friction_epsilon'],np.float32))
    # Use the authored extent as a normalization length, never scale geometry.
    values['reference_length']=np.asarray(max(np.ptp(values['rest_x'],axis=0).max(),.1),np.float32)
    metadata={'family':'mixed','official_demo':identifier,'contract':'official_objective_materials_v5','contacts':'analytic_plane','box_sites':[],'groups':groups,'particle_count':n,'body_count':b,'dt':example.sim_dt,'frame_dt':example.frame_dt,'sim_substeps':example.sim_substeps,'solver_configuration':configuration,'contact_contract':'Particle-plane finite material law at convergence (reference retains original within-step penalty ramp); active rigid ALM contacts are not admitted.'}
    return values,metadata


def snapshot(static,model,state):
    n=model.particle_count;b=model.body_count;total=len(static['rest_x'])
    result={key:np.zeros((total,4 if key=='q' else 3),np.float32) for key in ('x','q','v','omega')}
    result['q'][:,3]=1
    if n:result['x'][:n]=state.particle_q.numpy();result['v'][:n]=state.particle_qd.numpy()
    if b:
        poses=state.body_q.numpy();q=poses[:,3:];com=static['body_com'][n:n+b]
        result['x'][n:n+b]=poses[:,:3]+com+2*np.cross(q[:,:3],np.cross(q[:,:3],com)+q[:,3:]*com)
        result['q'][n:n+b]=q
        result['v'][n:n+b]=state.body_qd.numpy()[:,:3];result['omega'][n:n+b]=state.body_qd.numpy()[:,3:]
    return result
