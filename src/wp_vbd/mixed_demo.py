"""Five-object mixed fixture. Solver selection belongs to the external runner."""

from copy import deepcopy
import math

from .assets import build_asset


def build_mixed(spec, device, newton, wp, *, builder_only=False):
    if spec["family"] != "mixed":
        raise ValueError("Expected an explicit mixed asset")
    if spec.get('contacts')!='plane':
        raise ValueError('The current mixed contract requires explicit plane contact')
    families = {part["family"] for part in spec["objects"]}
    if families != {"joint","rigid","softbody","cloth","rod"} or len(spec['objects'])!=5:
        raise ValueError("Mixed fixture requires all five requested object families")
    builder = newton.ModelBuilder(gravity=tuple(spec["gravity"]))
    groups = []
    plane = spec["plane"]
    if not all(math.isfinite(float(value)) for value in plane.values()):
        raise ValueError('Plane parameters must be finite')
    if any(value<0 for key,value in plane.items() if key!='height') or plane['activation_radius']<=0 or plane['friction_epsilon']<=0 or plane['ke']<=0:
        raise ValueError('Invalid plane material or numerical offset')
    for part in spec["objects"]:
        part = deepcopy(part)
        if not math.isfinite(part.get('friction',plane['particle_mu'])) or part.get('friction',plane['particle_mu'])<0:
            raise ValueError('Body friction must be finite and nonnegative')
        part.update(gravity=spec["gravity"],contacts=False,_directory=spec.get("_directory","."))
        template, params = build_asset(part,device,newton,wp,builder_only=True)
        template.particle_radius[:] = [plane["activation_radius"]]*len(template.particle_q)
        for i in range(template.shape_count):
            template.shape_material_ke[i] = plane["ke"]
            template.shape_material_kd[i] = plane["kd"]
            template.shape_material_mu[i] = part.get("friction",plane["particle_mu"])
        if spec.get('interobject_contacts') is False:
            for i in range(template.shape_count):template.shape_flags[i]&=~int(newton.ShapeFlags.COLLIDE_PARTICLES)
        entry={"name":part["name"],"family":part["family"],"particle_start":builder.particle_count,"body_start":builder.body_count,"shape_start":builder.shape_count,"params":params}
        builder.add_builder(template,label_prefix=part["name"]+"/")
        entry.update(particle_end=builder.particle_count,body_end=builder.body_count,shape_end=builder.shape_count)
        groups.append(entry)
    if spec.get('interobject_contacts') is False:
        for i,left in enumerate(groups):
            if not spec.get('self_contact',False):
                for a in range(left['shape_start'],left['shape_end']):
                    for b in range(a+1,left['shape_end']):builder.add_shape_collision_filter_pair(a,b)
            for right in groups[i+1:]:
                for a in range(left['shape_start'],left['shape_end']):
                    for b in range(right['shape_start'],right['shape_end']):builder.add_shape_collision_filter_pair(a,b)
    lookup={g['name']:g for g in groups}
    for attachment in spec.get('attachments',[]):
        particle_group=lookup[attachment['particle_object']];body_group=lookup[attachment['body_object']]
        particle=int(attachment['particle']);body=int(attachment.get('body',0))
        if not 0<=particle<particle_group['particle_end']-particle_group['particle_start']:raise ValueError('Attachment particle out of range')
        if not 0<=body<body_group['body_end']-body_group['body_start']:raise ValueError('Attachment body out of range')
        if attachment.get('damping',0)!=0:raise ValueError('Initial attachment contract supports zero material damping')
        particle+=particle_group['particle_start'];body+=body_group['body_start']
        anchor=attachment.get('body_point')
        if anchor is None:
            anchor=wp.transform_point(wp.transform_inverse(builder.body_q[body]),wp.vec3(*builder.particle_q[particle]))
        builder.add_attachment_particle_body(particle,body,body_point=anchor,stiffness=attachment['stiffness'],damping=0.,enabled=attachment.get('enabled',True))
    builder.add_ground_plane(height=plane["height"],cfg=newton.ModelBuilder.ShapeConfig(ke=plane["ke"],kd=plane["kd"],mu=plane["ground_mu"],margin=plane["rigid_margin"]))
    builder.color()
    if builder_only:return builder,groups
    model=builder.finalize(device=device)
    model.soft_contact_ke=plane["ke"]
    model.soft_contact_kd=plane["kd"]
    model.soft_contact_mu=plane["particle_mu"]
    return model,groups
