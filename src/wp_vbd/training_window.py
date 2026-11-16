"""Transfer only the supervised causal window, retaining every input/history."""
import torch


def transfer_window(states, start, horizon, device):
    if start<0 or horizon<1:raise ValueError('A nonnegative start and positive horizon are required')
    if any(len(states[key])<start+horizon+1 for key in ('x','q','v','omega')):
        raise ValueError('Complete states through the supervised endpoint required')
    result={}
    for key,value in states.items():
        required=horizon if key.startswith(('input_','history__')) else horizon+1
        if not isinstance(value,torch.Tensor) or value.ndim<1 or len(value)<start+required:
            raise ValueError('Incomplete state/input history: '+key)
        result[key]=value[start:start+horizon+1].to(device)
    return result
