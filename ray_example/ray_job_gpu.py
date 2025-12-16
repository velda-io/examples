import ray
import time

# Initialize Ray
ray.init()

@ray.remote(num_gpus=1)
def gpu_task():
    import torch
    return torch.cuda.get_device_name(0)

if __name__ == "__main__":
    # Launch tasks in parallel (this will use multiple workers)
    futures = [gpu_task.remote()]

    # Gather results
    results = ray.get(futures)

    print("Results:", results)
