"""Unmasked physical constraint loads from already evaluated material rows.

Spatial vectors are world-frame [force, torque about the owner's COM].
Packets retain real owner incidence until proxy-vs-owned eligibility is
known. No inertia defect, neural correction or extra constitutive call is
turned into a physical force. Elastic continuum forces are not exported here.
"""
from dataclasses import dataclass

import torch

CONTACT_FAMILIES=frozenset(('rigid_contact','surface_contact','self_contact'))
FAMILIES=CONTACT_FAMILIES|{'joint','rod_joint'}


@dataclass(frozen=True)
class PhysicalLoadPacket:
    family: str
    node_count: int
    owners: torch.Tensor
    participating: torch.Tensor
    wrench: torch.Tensor
    valid: torch.Tensor

    def __post_init__(self):
        if self.family not in FAMILIES or type(self.node_count) is not int or self.node_count<1:
            raise ValueError('Known physical family and positive owner count required')
        if self.owners.ndim!=2 or self.owners.dtype not in (torch.int32,torch.int64):
            raise ValueError('Real-owner row incidence required')
        rows,slots=self.owners.shape
        if self.participating.shape!=(rows,slots) or self.participating.dtype!=torch.bool:
            raise ValueError('Explicit participating physical slots required')
        if self.wrench.ndim!=4 or self.wrench.shape[1:]!=(rows,slots,6):
            raise ValueError('World force/COM torque required for each real owner slot')
        if not self.wrench.is_floating_point():raise ValueError('Physical loads require floating-point storage')
        if self.valid.shape!=self.wrench.shape[:2] or self.valid.dtype!=torch.bool:
            raise ValueError('Per-batch physical row validity required')
        if any(t.device!=self.wrench.device for t in (self.owners,self.participating,self.valid)):
            raise ValueError('Packet storage must use one device')
        if bool(((self.owners< -1)|(self.owners>=self.node_count)).any()):raise ValueError('Physical owner out of range')
        if bool((self.participating&(self.owners<0)).any()):raise ValueError('Padding cannot participate')
        if not bool(torch.isfinite(self.wrench).all()):raise ValueError('Nonfinite physical load')
        if bool((self.wrench*(~self.participating)[None,:,:,None]).any()):
            raise ValueError('A nonparticipating slot cannot carry physical load')
        for name in ('owners','participating','wrench','valid'):
            object.__setattr__(self,name,getattr(self,name).detach().clone())
        object.__setattr__(self,'owners',self.owners.long())


def _direct(family,node_count,owners,result,reference):
    count=len(owners)
    if not count:
        wrench=reference.new_zeros(len(reference),0,owners.shape[1],6)
        valid=torch.ones(len(reference),0,device=reference.device,dtype=torch.bool)
    else:
        if 'force' not in result or 'torque' not in result:raise ValueError('Missing evaluated pair force/COM torque')
        wrench=torch.cat((result['force'],result['torque']),-1)
        valid=result['valid'][:,:count]
    return PhysicalLoadPacket(family,node_count,owners,owners>=0,wrench,valid)


def _weighted(family,node_count,contacts,result):
    ports=contacts.ports
    wrench=ports.owner_wrenches(result['jacobian'],result['force'])
    participating=(ports.owners>=0)&(ports.weights>0).any(1)
    return PhysicalLoadPacket(family,node_count,ports.owners,participating,wrench,result['valid'])


def packets_from_rows(scene,contacts,result,reference):
    """Snapshot last evaluated contact/finite-joint/rod loads without re-solving."""
    if getattr(scene,'mimic_spec',None) is not None:
        raise NotImplementedError('Hard mimic projection reactions need an explicit readout')
    packets=[]
    if 'rigid_rows' in result:
        packets.append(_direct('rigid_contact',scene.n,contacts.rigid.owners,result['rigid_rows'],reference))
        packets.append(_weighted('surface_contact',scene.n,contacts.soft,result['soft_rows']))
    else:
        direct=_direct('rigid_contact',scene.n,contacts.owners,result,reference)
        if hasattr(contacts,'source_kind'):
            for kind,family in ((0,'rigid_contact'),(1,'surface_contact')):
                mask=contacts.source_kind==kind
                packets.append(PhysicalLoadPacket(family,scene.n,direct.owners[mask],direct.participating[mask],direct.wrench[:,mask],direct.valid[:,mask]))
        else:packets.append(direct)
    if 'self_rows' in result:
        packets.append(_weighted('self_contact',scene.n,contacts.self_contacts,result['self_rows']))
    for key,owners,family,residual_key in (
        ('joint_rows',getattr(getattr(scene,'joint_spec',None),'owners',None),'joint','pair_residual'),
        ('rod_rows',getattr(getattr(scene,'rod',None),'joints',None),'rod_joint','residual')):
        if key in result:
            if owners is None:raise ValueError('Missing physical joint owner schema')
            row=result[key];valid=row['valid'] if 'valid' in row else row['in_domain']
            packets.append(PhysicalLoadPacket(family,scene.n,owners,owners>=0,-row[residual_key],valid))
    if len({p.family for p in packets})!=len(packets):raise ValueError('Duplicate physical load family')
    return tuple(packets)


def owner_loads(packet):
    if not bool(packet.valid.all()):raise ValueError('Invalid physical rows cannot be exported')
    value=packet.wrench*packet.participating[None,:,:,None]
    return value.new_zeros(len(value),packet.node_count,6).index_add(1,packet.owners.clamp_min(0).flatten(),value.flatten(1,2))


def proxy_feedback(packets,proxy_global,owned_response,output_count,*,accepted,families=CONTACT_FAMILIES):
    """Export only proxy rows incident on an owned responding physical node.

    Proxy-only and proxy-vs-static rows may strengthen a destination solve,
    but must not be fed back as cross-owner interactions. Joint feedback is
    opt-in: Newton VBD's proxy hooks use contact-only feedback.
    """
    if not packets or len({p.family for p in packets})!=len(packets):raise ValueError('One explicit packet per physical family required')
    first=packets[0];n=first.node_count;batch=len(first.wrench)
    if any(value.device!=first.wrench.device for value in (proxy_global,owned_response,accepted)):
        raise ValueError('Feedback roles and accepted flags must use the packet device')
    if accepted.dtype!=torch.bool or accepted.shape!=(batch,) or not bool(accepted.all()):
        raise ValueError('Only accepted physical steps can export coupling feedback')
    if proxy_global.shape!=(n,) or proxy_global.dtype!=torch.long or owned_response.shape!=(n,) or owned_response.dtype!=torch.bool:
        raise ValueError('Complete physical ownership and proxy mapping required')
    if type(output_count) is not int or output_count<0 or bool(((proxy_global< -1)|(proxy_global>=output_count)).any()):
        raise ValueError('Invalid proxy global output mapping')
    if bool(((proxy_global>=0)&owned_response).any()):raise ValueError('Proxy and owned responding roles overlap')
    if not set(families)<=FAMILIES:raise ValueError('Unknown feedback family')
    output=first.wrench.new_zeros(batch,output_count,6)
    for packet in packets:
        if packet.node_count!=n or len(packet.wrench)!=batch or packet.wrench.device!=first.wrench.device or packet.wrench.dtype!=first.wrench.dtype:
            raise ValueError('Packet owner/batch/device/dtype contract changed')
        if not bool(packet.valid.all()):raise ValueError('Invalid rows cannot supply physical feedback')
        if packet.family not in families:continue
        ids=packet.owners.clamp_min(0);active=packet.participating
        proxy=(proxy_global[ids]>=0)&active
        owned=owned_response[ids]&active
        cross=proxy.any(-1)&owned.any(-1)
        keep=proxy&cross[:,None]
        targets=proxy_global[ids][keep]
        if len(targets):output=output.index_add(1,targets,packet.wrench[:,keep])
    return output
