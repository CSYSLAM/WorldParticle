"""Convert native arrays, including valid zero-length CPU allocations."""
import torch


def native_tensor(wp, array):
    # Warp's CPU array interface cannot expose a null data pointer. numpy()
    # still preserves the valid empty array's dtype and component dimensions.
    return wp.to_torch(array) if array.size else torch.as_tensor(array.numpy(), device=str(array.device))
