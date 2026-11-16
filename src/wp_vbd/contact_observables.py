"""Transactional native readout of the actual final constitutive contact rows.

The explicit v29 contract exports the independently detected expanded contact
geometry into the caller's Contacts only after acceptance. Geometry and force
therefore share slots; readout performs no solve, collision matching or extra
Coulomb update. Wrenches are on body 0, about its COM (world origin for world).
"""
import copy
import torch

GEOMETRY_FIELDS=('rigid_contact_shape0','rigid_contact_shape1','rigid_contact_point0','rigid_contact_point1','rigid_contact_offset0','rigid_contact_offset1','rigid_contact_normal','rigid_contact_margin0','rigid_contact_margin1','rigid_contact_surface_velocity','rigid_contact_point_id','rigid_contact_tids')


def enabled(scene):return scene.metadata.get('native_contact_export')=='expanded_rigid_stateless_v29'


def validate(runtime,observables,contacts):
    if enabled(runtime.scene) and contacts is runtime.oracle.contacts:raise ValueError('The private collision staging buffer cannot also be the native export target')
    if observables is not None:
        # Base validation also updates layout metadata. Validate a shallow
        # container copy so a later rejection cannot publish that update.
        candidate=copy.copy(observables);candidate._contact_layouts=observables._contact_layouts.copy()
        runtime.validate_observables(candidate,contacts)
    requested=observables is not None and observables.is_requested(runtime.ObservableKind.CONTACT_F)
    runtime.scene.request_contact_wrench=requested
    if not enabled(runtime.scene):
        if requested:raise ValueError('Contact readout requires the explicit native geometry export contract')
        return
    if contacts is None:raise ValueError('Native geometry export requires the caller Contacts buffer')
    if contacts.device!=runtime.source_model.device:raise ValueError('Contact export device mismatch')
    if getattr(contacts,'rigid_contact_match_index',None) is not None or getattr(contacts,'contact_report',False):raise ValueError('Matched contact history/reporting needs its own export identity lifecycle')
    for name in GEOMETRY_FIELDS:
        target=getattr(contacts,name,None)
        if name!='rigid_contact_surface_velocity' and target is None:raise ValueError('Missing native geometry array: '+name)


def prepare(runtime,diagnostic,contacts,observables):
    if not enabled(runtime.scene):return None
    c=runtime.scene.candidates;source=runtime.oracle.contacts;wp=runtime.wp
    count=c.source_count
    if count>contacts.rigid_contact_max:raise ValueError(f'Native contact export overflow: {count}/{contacts.rigid_contact_max}')
    if getattr(contacts,'rigid_contact_surface_velocity',None) is None and bool(c.surface_velocity.any()):raise ValueError('Target Contacts lacks nonzero surface velocity storage')
    copies=[]
    for name in GEOMETRY_FIELDS:
        target=getattr(contacts,name,None);value=getattr(source,name,None)
        if target is None:continue
        if value is not None:
            if value.dtype!=target.dtype or value.device!=target.device:raise ValueError('Native geometry type/device mismatch: '+name)
            copies.append((name,target,value,count))
        else:copies.append((name,target,None,0))
    values=None
    if observables is not None and observables.is_requested(runtime.ObservableKind.CONTACT_F):
        pair=diagnostic['contact_wrench'];assert pair.shape==(1,len(c.owners),2,6)
        if not bool(torch.isfinite(pair).all()):raise ValueError('Nonfinite contact output')
        # Use actual operator values, never infer contact forces from the
        # remaining inertial residual or attribute NN corrections to contacts.
        values=wp.to_torch(observables.contact_f).new_zeros(observables.contact_f.shape[0],6)
        values[c.source_slots]=pair[0,:,0].to(values.dtype)
    return copies,values,int(wp.to_torch(contacts.contact_generation)[0])+1


def commit(runtime,prepared,contacts,observables):
    if prepared is None:return
    copies,values,generation=prepared;wp=runtime.wp
    for name,target,source,count in copies:
        if name in ('rigid_contact_shape0','rigid_contact_shape1','rigid_contact_tids'):target.fill_(-1)
        else:target.zero_()
        if count:wp.copy(target,source,count=count)
    contacts.rigid_contact_count.fill_(runtime.scene.candidates.source_count)
    contacts.soft_contact_count.zero_();contacts.contact_generation.fill_(generation)
    # This deprecated vector array is not an observable and must not retain
    # values referring to the geometry overwritten by this native backend.
    if contacts.rigid_contact_force is not None:contacts.rigid_contact_force.zero_()
    if values is not None:
        wp.to_torch(observables.contact_f).copy_(values)
        observables._contact_layouts[runtime.ObservableKind.CONTACT_F]=contacts
    runtime.contact_exports+=1
