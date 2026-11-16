"""Build native assets from geometry and physical parameters, without a solver."""
from copy import deepcopy
import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np

SCHEMA='native_shared_asset_v1'


def fields(value,allowed,required=(),label='asset'):
    if not isinstance(value,dict) or set(value)-set(allowed) or set(required)-set(value):
        raise ValueError(f'Invalid or missing {label} fields')
    return value


def number(value,label,*,positive=False,nonnegative=False):
    if type(value) not in (int,float) or not math.isfinite(value):raise ValueError('Finite number required: '+label)
    if positive and value<=0 or nonnegative and value<0:raise ValueError('Invalid physical value: '+label)
    return float(value)


def vector(value,count,label):
    if not isinstance(value,(list,tuple)) or len(value)!=count:raise ValueError('Invalid vector: '+label)
    return [number(v,label) for v in value]


def quaternion(value,label):
    result=vector(value,4,label)
    if abs(np.linalg.norm(result)-1)>1e-6:raise ValueError('Unit xyzw quaternion required: '+label)
    return result


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path,overrides=()):
    path=Path(path).resolve();spec=json.loads(path.read_text())
    for entry in overrides:
        key,separator,raw=entry.partition('=')
        if not separator:raise ValueError('Use --set path.to.value=JSON')
        parts=key.split('.');target=spec
        for part in parts[:-1]:target=target[int(part)] if isinstance(target,list) else target[part]
        leaf=int(parts[-1]) if isinstance(target,list) else parts[-1]
        if isinstance(target,dict) and leaf not in target:raise ValueError('Override must name an existing field: '+key)
        target[leaf]=json.loads(raw)
    fields(spec,('schema','name','gravity','simulation','contact','solver','objects','joints'),('schema','objects'))
    if spec['schema']!=SCHEMA or not isinstance(spec['objects'],list) or not spec['objects']:raise ValueError('Explicit nonempty native shared asset required')
    spec['_directory']=str(path.parent);spec['_asset_file']=str(path);spec['_asset_sha256']=digest(path)
    return spec


def build(spec,newton,wp,*,device='cpu'):
    """Return a native Model, physical configuration and source provenance."""
    spec=deepcopy(spec);objects=spec['objects'];directory=Path(spec['_directory'])
    gravity=vector(spec.get('gravity',[0,0,-9.8]),3,'gravity')
    simulation=fields(spec.get('simulation',{}),('dt','substeps','motion_budget'))
    dt=number(simulation.get('dt',1/600),'dt',positive=True)
    substeps=simulation.get('substeps',10)
    if type(substeps) is not int or substeps<=0:raise ValueError('Positive integer substeps required')
    motion=number(simulation.get('motion_budget',.02),'motion budget',positive=True)
    contact=fields(spec.get('contact',{}),('friction','rigid_ke','rigid_kd','shape_margin','soft_ke','soft_kd','self_margin','self_gap','ground_height'))
    defaults={'friction':.35,'rigid_ke':2000.,'rigid_kd':.1,'shape_margin':.003,
              'soft_ke':100.,'soft_kd':.1,'self_margin':.01,'self_gap':.005,'ground_height':0.}
    contact={**defaults,**contact}
    for key,value in contact.items():number(value,key,nonnegative=key!='ground_height')
    if contact['self_margin']<=0 or not 0<contact['self_gap']<=contact['self_margin']:
        raise ValueError('Require positive self gap <= interaction margin')
    configuration={k:v.default for k,v in inspect.signature(newton.solvers.SolverVBD).parameters.items() if k not in ('model','collision_pipeline')}
    joint_material=fields(spec.get('solver',{}),('rigid_joint_linear_ke','rigid_joint_angular_ke','rigid_joint_linear_kd','rigid_joint_angular_kd','friction_epsilon'))
    for key,value in joint_material.items():number(value,key,positive=key=='friction_epsilon',nonnegative=True)
    configuration.update(joint_material)
    configuration.update(particle_enable_self_contact=True,particle_self_contact_margin=contact['self_margin'],
                         particle_self_contact_gap=contact['self_gap'])
    builder=newton.ModelBuilder(gravity=tuple(gravity));body_names={};object_records=[];sources={};rod_present=False
    names=set()
    def transform(value,label):
        p=vector(value[:3],3,label);q=quaternion(value[3:],label)
        return wp.transform(wp.vec3(*p),wp.quat(*q))
    def shape_configuration(material,density=None):
        allowed=('density','mass','inertia','center_of_mass','friction','ke','kd','margin')
        fields(material,allowed)
        return newton.ModelBuilder.ShapeConfig(density=material.get('density',1000.) if density is None else density,
            ke=number(material.get('ke',contact['rigid_ke']),'shape ke',nonnegative=True),
            kd=number(material.get('kd',contact['rigid_kd']),'shape kd',nonnegative=True),
            mu=number(material.get('friction',contact['friction']),'shape friction',nonnegative=True),
            margin=number(material.get('margin',contact['shape_margin']),'shape margin',nonnegative=True))
    for item in objects:
        fields(item,('name','family','geometry','material','position','rotation_xyzw','velocity','angular_velocity','particle_radius','pins','fixed_segments','fixed','shape_transform'),('name','family','geometry','material'),'object')
        name=item['name'];family=item['family'];geometry=item['geometry'];material=item['material']
        if not isinstance(name,str) or not name or ':' in name or name=='world' or name in names:raise ValueError('Unique nonempty object names without reserved colon required')
        if not isinstance(geometry,dict) or not isinstance(material,dict):raise ValueError('Explicit object geometry and material dictionaries required')
        names.add(name)
        position=vector(item.get('position',[0,0,0]),3,name+' position');rotation=quaternion(item.get('rotation_xyzw',[0,0,0,1]),name+' rotation')
        velocity=vector(item.get('velocity',[0,0,0]),3,name+' velocity');omega=vector(item.get('angular_velocity',[0,0,0]),3,name+' angular velocity')
        pos=wp.vec3(*position);rot=wp.quat(*rotation);vel=wp.vec3(*velocity)
        particle_start=len(builder.particle_q);body_start=len(builder.body_q)
        if family in ('cloth','softbody'):
            if any(omega):raise ValueError('Particle angular velocity needs an explicit velocity field')
            if item.get('fixed') or 'fixed_segments' in item or 'shape_transform' in item:raise ValueError('Use explicit particle pins for deformable assets')
            radius=number(item.get('particle_radius',.005),name+' particle radius',positive=True)
            if family=='cloth':
                fields(material,('stretch','area','bending','damping','bend_damping','mass','areal_density'),('stretch','area'))
                physical={k:number(v,name+' '+k,positive=k in ('mass','areal_density'),nonnegative=True) for k,v in material.items()}
                common=dict(pos=pos,rot=rot,vel=vel,tri_ke=physical['stretch'],tri_ka=physical['area'],
                    tri_kd=physical.get('damping',0.),edge_ke=physical.get('bending',0.),edge_kd=physical.get('bend_damping',0.),particle_radius=radius,label=name)
            else:
                fields(material,('mu','lambda','young_modulus','poisson_ratio','density','damping'),('density',))
                if 'young_modulus' in material or 'poisson_ratio' in material:
                    if not {'young_modulus','poisson_ratio'}<=set(material) or {'mu','lambda'}&set(material):raise ValueError('Choose E/nu or Lame mu/lambda')
                    young=number(material['young_modulus'],'Young modulus',positive=True);nu=number(material['poisson_ratio'],'Poisson ratio',nonnegative=True)
                    if nu>=.5:raise ValueError('Require Poisson ratio < .5')
                    material={k:v for k,v in material.items() if k not in ('young_modulus','poisson_ratio')}
                    material.update(mu=young/(2*(1+nu)),**{'lambda':young*nu/((1+nu)*(1-2*nu))})
                if not {'mu','lambda'}<=set(material):raise ValueError('Softbody needs complete elastic parameters')
                physical={k:number(v,name+' '+k,positive=k=='density',nonnegative=True) for k,v in material.items()}
                common=dict(pos=pos,rot=rot,vel=vel,density=physical['density'],k_mu=physical['mu'],
                    k_lambda=physical['lambda'],k_damp=physical.get('damping',0.),particle_radius=radius,label=name)
            if geometry.get('type')=='grid':
                fields(geometry,('type','resolution','size'),('resolution','size'))
                dim=2 if family=='cloth' else 3;resolution=geometry['resolution'];size=vector(geometry['size'],dim,name+' size')
                if not isinstance(resolution,list) or len(resolution)!=dim or any(type(n) is not int or n<1 for n in resolution) or min(size)<=0:raise ValueError('Positive grid dimensions required')
                if family=='cloth':
                    if 'areal_density' in physical or 'mass' not in physical:raise ValueError('Cloth grid requires explicit mass per node')
                    builder.add_cloth_grid(**common,dim_x=resolution[0],dim_y=resolution[1],cell_x=size[0]/resolution[0],cell_y=size[1]/resolution[1],mass=physical['mass'])
                else:builder.add_soft_grid(**common,dim_x=resolution[0],dim_y=resolution[1],dim_z=resolution[2],cell_x=size[0]/resolution[0],cell_y=size[1]/resolution[1],cell_z=size[2]/resolution[2])
            elif family=='cloth' and geometry.get('type')=='obj':
                fields(geometry,('type','path','scale'),('path',));path=(directory/geometry['path']).resolve()
                from .assets import read_obj
                vertices,indices=read_obj(path);sources[str(path)]=digest(path)
                if ('mass' in physical)==('areal_density' in physical):raise ValueError('OBJ cloth requires mass per node or areal density')
                builder.add_cloth_mesh(**common,vertices=[wp.vec3(*p) for p in vertices],indices=indices.flatten().tolist(),
                    scale=number(geometry.get('scale',1.),'mesh scale',positive=True),density=physical.get('areal_density',1.),validate_mesh=True)
                if 'mass' in physical:builder.particle_mass[particle_start:]=[physical['mass']]*(len(builder.particle_q)-particle_start)
            elif family=='softbody' and geometry.get('type')=='tet_npz':
                fields(geometry,('type','path','scale'),('path',));path=(directory/geometry['path']).resolve()
                with np.load(path,allow_pickle=False) as archive:vertices=archive['vertices'].copy();indices=archive['tets'].copy()
                if vertices.ndim!=2 or vertices.shape[1]!=3 or not np.isfinite(vertices).all() or indices.ndim!=2 or indices.shape[1]!=4 or indices.dtype.kind not in 'iu' or not len(indices):raise ValueError('Finite vertices and integer tetrahedra required')
                if indices.min()<0 or indices.max()>=len(vertices):raise ValueError('Tet index outside vertices')
                determinant=np.linalg.det((vertices[indices[:,1:]]-vertices[indices[:,:1]]).transpose(0,2,1))
                if np.any(determinant<=1e-18):raise ValueError('Positively oriented nondegenerate tetrahedra required')
                sources[str(path)]=digest(path)
                builder.add_soft_mesh(**common,vertices=[wp.vec3(*p) for p in vertices],indices=indices.flatten().tolist(),
                    scale=number(geometry.get('scale',1.),'mesh scale',positive=True),validate_mesh=True)
            else:raise ValueError('Unsupported deformable geometry')
            pins=item.get('pins',[]);count=len(builder.particle_q)-particle_start
            if not isinstance(pins,list) or any(type(i) is not int or not 0<=i<count for i in pins) or len(set(pins))!=len(pins):raise ValueError('Explicit unique local particle pin indices required')
            for pin in pins:builder.particle_mass[particle_start+pin]=0.;builder.particle_qd[particle_start+pin]=wp.vec3(0.)
        elif family=='rod':
            if 'pins' in item or 'particle_radius' in item or 'shape_transform' in item or 'fixed' in item:raise ValueError('Rod uses radius and fixed_segments')
            fields(geometry,('type','points','radius'),('type','points','radius'))
            if geometry['type']!='polyline' or len(geometry['points'])<3:raise ValueError('Rod requires at least three authored points')
            points=np.asarray([vector(p,3,name+' rod point') for p in geometry['points']])
            if np.any(np.linalg.norm(np.diff(points,axis=0),axis=-1)<1e-8):raise ValueError('Rod segments must have nonzero length')
            fields(material,('stretch','shear','bending','twist','density','stretch_damping','shear_damping','bend_damping','twist_damping','friction'),('stretch','shear','bending','twist'))
            physical={k:number(v,name+' '+k,positive=k=='density',nonnegative=True) for k,v in material.items()}
            rod=newton.Rod([pos+wp.quat_rotate(rot,wp.vec3(*p)) for p in points],radius=number(geometry['radius'],'rod radius',positive=True));rod.compute_frames()
            cfg=newton.ModelBuilder.ShapeConfig(density=physical.get('density',1000.),ke=contact['rigid_ke'],kd=contact['rigid_kd'],mu=physical.get('friction',contact['friction']),margin=contact['shape_margin'])
            bodies,joints=builder.add_rod(rod=rod,cfg=cfg,label=name,body_frame_origin='start',
                stretch_stiffness=physical['stretch'],shear_stiffness=physical['shear'],bend_stiffness=physical['bending'],twist_stiffness=physical['twist'],
                **{key:physical[key] for key in ('stretch_damping','shear_damping','bend_damping','twist_damping') if key in physical})
            fixed=item.get('fixed_segments',[])
            if not isinstance(fixed,list) or any(type(i) is not int or not 0<=i<len(bodies) for i in fixed) or len(set(fixed))!=len(fixed):raise ValueError('Invalid fixed rod segments')
            for index in fixed:
                body=bodies[index];builder.body_mass[body]=builder.body_inv_mass[body]=0.;builder.body_inertia[body]=builder.body_inv_inertia[body]=wp.mat33(0.)
            for index,body in enumerate(bodies):body_names[f'{name}:{index}']=body
            rod_present=True
        elif family=='rigid':
            if any(k in item for k in ('pins','fixed_segments','particle_radius')):raise ValueError('Rigid object has unsupported particle fields')
            fields(material,('mass','density','inertia','center_of_mass','friction','ke','kd','margin'))
            if 'mass' in material and 'density' in material:raise ValueError('Choose rigid mass or density')
            kind=geometry.get('type')
            if kind=='box':
                fields(geometry,('type','half_extents'),('half_extents',));half=vector(geometry['half_extents'],3,name+' half extents')
                if min(half)<=0:raise ValueError('Positive box half extents required')
                volume=8*math.prod(half)
            elif kind in ('sphere','capsule'):
                fields(geometry,('type','radius','half_height'),('radius',));radius=number(geometry['radius'],name+' radius',positive=True)
                height=number(geometry.get('half_height',0.),name+' half height',nonnegative=True)
                if kind=='sphere' and 'half_height' in geometry:raise ValueError('Sphere has no half height')
                volume=4*math.pi*radius**3/3+(2*math.pi*radius**2*height if kind=='capsule' else 0.)
            else:raise ValueError('Rigid asset loader currently supports box/sphere/capsule shapes')
            fixed=item.get('fixed',False)
            if type(fixed) is not bool:raise ValueError('Explicit Boolean fixed body flag required')
            if fixed and (any(velocity) or any(omega)):raise ValueError('Fixed bodies require zero initial velocity')
            density=number(material.get('density',1000.),name+' density',positive=True)
            if 'mass' in material:density=number(material['mass'],name+' mass',positive=True)/volume
            kwargs={}
            if 'inertia' in material or 'center_of_mass' in material:
                if 'inertia' not in material or 'mass' not in material or fixed:raise ValueError('Custom inertia requires positive explicit mass and a dynamic body')
                inertia=np.asarray(material['inertia'],dtype=float)
                if inertia.shape!=(3,3) or not np.isfinite(inertia).all() or not np.allclose(inertia,inertia.T,rtol=0,atol=1e-12) or np.linalg.eigvalsh(inertia).min()<=0:raise ValueError('Symmetric positive definite inertia required')
                kwargs.update(mass=material['mass'],inertia=wp.mat33(*inertia.flatten()),lock_inertia=True,com=wp.vec3(*vector(material.get('center_of_mass',[0,0,0]),3,name+' COM')));density=0.
            body=builder.add_link(xform=wp.transform(pos,rot),label=name,**kwargs);body_names[name]=body
            cfg=shape_configuration(material,0. if fixed else density)
            local=transform(item.get('shape_transform',[0,0,0,0,0,0,1]),name+' shape transform')
            if kind=='box':builder.add_shape_box(body,hx=half[0],hy=half[1],hz=half[2],cfg=cfg,xform=local)
            elif kind=='sphere':builder.add_shape_sphere(body,radius=radius,cfg=cfg,xform=local)
            else:builder.add_shape_capsule(body,radius=radius,half_height=height,cfg=cfg,xform=local)
            physical=deepcopy(material)
        else:raise ValueError('Unsupported native asset family: '+str(family))
        for body in range(body_start,len(builder.body_q)):
            builder.body_qd[body]=wp.spatial_vector(*velocity,*omega)
        object_records.append({'name':name,'family':family,'particle_ids':list(range(particle_start,len(builder.particle_q))),
                               'body_ids':list(range(body_start,len(builder.body_q))),'physical_parameters':physical})
    joint_records=[];authored=[];children=set()
    for joint in spec.get('joints',[]):
        fields(joint,('type','parent','child','parent_frame','child_frame','axis','target_pos','target_vel','target_ke','target_kd','limit_lower','limit_upper','limit_ke','limit_kd','linear_axes','angular_axes'),('type','parent','child'),'joint')
        parent=-1 if joint['parent']=='world' else body_names[joint['parent']];child=body_names[joint['child']]
        if child in children:raise ValueError('Multiple incoming authored joints need a separate closed-loop asset contract')
        children.add(child);kind=joint['type'];common={'parent':parent,'child':child,
            'parent_xform':transform(joint.get('parent_frame',[0,0,0,0,0,0,1]),'parent frame'),
            'child_xform':transform(joint.get('child_frame',[0,0,0,0,0,0,1]),'child frame')}
        if kind in ('revolute','prismatic'):
            common['axis']=wp.vec3(*vector(joint.get('axis',[0,1,0]),3,'joint axis'))
            if float(wp.length(common['axis']))<1e-8:raise ValueError('Nonzero joint axis required')
            for key in ('target_pos','target_vel','target_ke','target_kd','limit_lower','limit_upper','limit_ke','limit_kd'):
                if key in joint:common[key]=number(joint[key],key,nonnegative=key.endswith(('_ke','_kd')))
        elif kind=='d6':
            for key in ('linear_axes','angular_axes'):
                axes=joint.get(key,[])
                if not isinstance(axes,list) or len(axes)>3:raise ValueError('D6 has at most three axes of each kind')
                common[key]=[builder.JointDofConfig(axis=wp.vec3(*vector(axis,3,key))) for axis in axes]
        elif kind not in ('fixed','ball'):raise ValueError('Unsupported authored joint type')
        allowed={'type','parent','child','parent_frame','child_frame'}|({'axis','target_pos','target_vel','target_ke','target_kd','limit_lower','limit_upper','limit_ke','limit_kd'} if kind in ('revolute','prismatic') else {'linear_axes','angular_axes'} if kind=='d6' else set())
        if set(joint)-allowed:raise ValueError('Joint fields do not apply to its type')
        index=getattr(builder,'add_joint_'+kind)(**common);authored.append(index);joint_records.append({'source_joint':index,**joint})
    rod_body_ids={i for row in object_records if row['family']=='rod' for i in row['body_ids']}
    for body in body_names.values():
        if body not in children and body not in rod_body_ids:authored.append(builder.add_joint_free(body))
    if authored:builder.add_articulation(authored)
    builder.add_ground_plane(height=contact['ground_height'],cfg=newton.ModelBuilder.ShapeConfig(ke=contact['rigid_ke'],kd=contact['rigid_kd'],mu=contact['friction'],margin=contact['shape_margin']))
    builder.color();model=builder.finalize(device=device)
    model.soft_contact_ke,model.soft_contact_kd,model.soft_contact_mu=contact['soft_ke'],contact['soft_kd'],contact['friction']
    if model.particle_count:
        from .elastic_rod_configuration import reciprocal_configuration
        configuration,topology=reciprocal_configuration(model,configuration)
    else:topology=None
    provenance={'schema':SCHEMA,'name':spec.get('name',Path(spec['_asset_file']).stem),'asset_file':spec['_asset_file'],
        'asset_sha256':spec['_asset_sha256'],'resolved_spec':{k:v for k,v in spec.items() if not k.startswith('_')},
        'mesh_sha256':sources,'objects':object_records,'joints':joint_records,'contact':contact,'dt':dt,
        'substeps':substeps,'motion_budget':motion,'body_names':body_names,'source_topology':topology,
        'has_native_rods':rod_present,'scope':'Static authored geometry and physical parameters only; no solver construction, trajectory, training admission or quality claim.'}
    return model,configuration,provenance
