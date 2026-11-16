"""Optional inference autocast inside the shared neural response only."""
import torch


class NeuralPrecision:
    def __init__(self,response,policy,device):
        if policy not in ('fp32','bf16'):raise ValueError('Explicit fp32 or bf16 neural precision required')
        self.policy=policy;self.device_type=str(device).split(':')[0];self.stack=[];self.handles=[]
        self.calls=0;self.linear_calls=0;self.bf16_outputs=0
        self.physical_checks=0
        if policy=='fp32':return
        if self.device_type not in ('cpu','cuda'):raise ValueError('CPU or CUDA inference required')
        if self.device_type=='cuda' and not torch.cuda.is_bf16_supported():raise ValueError('This device does not support BF16 inference')
        linear=[module for module in response.modules() if isinstance(module,torch.nn.Linear)]
        if not linear or any(module.weight.dtype!=torch.float32 for module in linear):
            raise ValueError('BF16 inference retains original FP32 neural weights')
        def begin(module,unused):
            if module.training or torch.is_grad_enabled():raise RuntimeError('Neural BF16 policy is inference only')
            context=torch.autocast(self.device_type,dtype=torch.bfloat16,cache_enabled=True)
            context.__enter__();self.stack.append(context);self.calls+=1
        def finish(module,unused,result):
            if self.stack:self.stack.pop().__exit__(None,None,None)
        def projection(module,unused,result):
            self.linear_calls+=1;self.bf16_outputs+=int(result.dtype==torch.bfloat16)
            # Sparse token/feature buffers retain the original FP32 interface.
            # Without this cast, BF16 Linear outputs cannot enter index_copy
            # buffers allocated from FP32 physical feature tensors.
            return result.to(module.weight.dtype)
        self.handles.append(response.register_forward_pre_hook(begin))
        for module in linear:self.handles.append(module.register_forward_hook(projection))
        self.handles.append(response.register_forward_hook(finish,always_call=True))

    def close(self):
        for handle in self.handles:handle.remove()
        self.handles=[]
        if self.stack:
            while self.stack:self.stack.pop().__exit__(None,None,None)
            raise RuntimeError('Neural precision context escaped its forward call')

    def verify_physical_boundary(self):
        if self.stack or torch.is_autocast_enabled(self.device_type):
            raise RuntimeError('Neural autocast may not escape into the physical step')
        self.physical_checks+=1

    def report(self):
        return {'policy':self.policy,'device_type':self.device_type,'autocast_neural_calls':self.calls,
            'linear_calls':self.linear_calls,'observed_bf16_linear_outputs':self.bf16_outputs,
            'stored_weight_dtype':'float32','projection_output_dtype':'float32',
            'physical_operator_autocast_outside_neural_response':False,
            'verified_physical_boundary_checks':self.physical_checks,
            'scope':'Optional inference arithmetic; no weight conversion or training policy change. BF16 outputs require separate trajectory/physics validation and timing on each device.'}
