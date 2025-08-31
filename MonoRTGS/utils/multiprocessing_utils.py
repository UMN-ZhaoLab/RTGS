import copy

import torch
import torch.multiprocessing as mp


class FakeQueue:
    def put(self, arg):
        del arg

    def get_nowait(self):
        raise mp.queues.Empty

    def qsize(self):
        return 0

    def empty(self):
        return True


def clone_obj(obj):
    """Clone object while handling CUDA tensors safely"""
    try:
        # For CUDA tensors, create a CPU copy first to avoid serialization issues
        if hasattr(obj, '__dict__'):
            clone_obj = copy.deepcopy(obj)
            for attr in clone_obj.__class__.__dict__.keys():
                # check if its a property
                if hasattr(clone_obj.__class__, attr) and isinstance(
                    getattr(clone_obj.__class__, attr), property
                ):
                    continue
                if isinstance(getattr(clone_obj, attr), torch.Tensor):
                    tensor = getattr(clone_obj, attr)
                    if tensor.is_cuda:
                        # Move to CPU first, then clone, then move back to GPU
                        cpu_tensor = tensor.cpu().detach().clone()
                        setattr(clone_obj, attr, cpu_tensor)
                    else:
                        setattr(clone_obj, attr, tensor.detach().clone())
            return clone_obj
        else:
            return copy.deepcopy(obj)
    except Exception as e:
        # If cloning fails, return a simple reference
        print(f"Warning: Failed to clone object, returning reference: {e}")
        return obj


def safe_queue_put(queue, item, timeout=1.0):
    """Safely put item in queue with timeout and error handling"""
    try:
        if hasattr(queue, 'put'):
            queue.put(item, timeout=timeout)
            return True
        else:
            return False
    except Exception as e:
        print(f"Warning: Failed to put item in queue: {e}")
        return False


def safe_queue_get(queue, timeout=1.0):
    """Safely get item from queue with timeout and error handling"""
    try:
        if hasattr(queue, 'get'):
            return queue.get(timeout=timeout)
        else:
            return None
    except Exception as e:
        print(f"Warning: Failed to get item from queue: {e}")
        return None
