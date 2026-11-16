"""Optional inference routing of tensor-only neural blocks to a CUDA device.

Physical state, geometry, history, acceptance and sparse operators retain their
caller's device. Only the existing neural modules cross this boundary.
"""
import torch
from torch import nn


class NeuralDeviceBridge:
    def __init__(self, response, device):
        from .heterogeneous_backbone import AttentionBlock, ConditionalFFN
        if response.training:raise ValueError('Neural device routing is inference only')
        self.device=torch.device(device)
        if self.device.type!='cuda':raise ValueError('Explicit CUDA neural device required')
        self.handles=[];self.modules=[];self.stacks={}
        self.calls=self.host_to_device_bytes=self.device_to_host_bytes=0
        classes=(AttentionBlock,ConditionalFFN,nn.Sequential,nn.Linear,nn.LayerNorm,nn.Embedding)
        def select(module,name):
            if isinstance(module,classes):
                self.modules.append((name,module));return
            for child_name,child in module.named_children():select(child,name+'.'+child_name)
        select(response,'response')
        for name,module in self.modules:
            module.to(self.device);self.stacks[id(module)]=[]
            self.handles.append(module.register_forward_pre_hook(self._before,with_kwargs=True))
            self.handles.append(module.register_forward_hook(self._after,always_call=True))

    @staticmethod
    def _tensors(value):
        if isinstance(value,torch.Tensor):return [value]
        if isinstance(value,(tuple,list)):return [t for item in value for t in NeuralDeviceBridge._tensors(item)]
        if isinstance(value,dict):return [t for item in value.values() for t in NeuralDeviceBridge._tensors(item)]
        return []

    def _move(self,value,device,*,incoming):
        if isinstance(value,torch.Tensor):
            if value.device!=device:
                size=value.numel()*value.element_size()
                if incoming:self.host_to_device_bytes+=size
                else:self.device_to_host_bytes+=size
            return value.to(device)
        if isinstance(value,tuple):return tuple(self._move(v,device,incoming=incoming) for v in value)
        if isinstance(value,list):return [self._move(v,device,incoming=incoming) for v in value]
        if isinstance(value,dict):return {k:self._move(v,device,incoming=incoming) for k,v in value.items()}
        return value

    def _before(self,module,args,kwargs):
        if module.training or torch.is_grad_enabled():raise RuntimeError('Neural device routing is inference only')
        tensors=self._tensors((args,kwargs))
        if not tensors:raise ValueError('Tensor-only neural block inputs required')
        origins={t.device for t in tensors}
        if len(origins)!=1:raise ValueError('Neural block inputs must start on one device')
        origin=next(iter(origins))
        if origin.type!='cpu' and origin!=self.device:raise ValueError('Unexpected neural block caller device')
        self.stacks[id(module)].append(origin);self.calls+=1
        return self._move(args,self.device,incoming=True),self._move(kwargs,self.device,incoming=True)

    def _after(self,module,args,result):
        stack=self.stacks[id(module)]
        if not stack:return result
        origin=stack.pop()
        return self._move(result,origin,incoming=False)

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
        if any(self.stacks.values()):raise RuntimeError('Neural device call did not finish')

    def report(self):
        return {'enabled':True,'neural_device':str(self.device),'tensor_only_blocks':[name for name,_ in self.modules],
            'calls':self.calls,'host_to_device_bytes':self.host_to_device_bytes,'device_to_host_bytes':self.device_to_host_bytes,
            'scope':'Existing tensor-only neural modules on CUDA; original parameters retained. Physical operators and accepted history stay on the caller device. Inference only; no dynamics replacement or realtime claim.'}
