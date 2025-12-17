import ray
import time

# Initialize Ray
ray.init()

# Define a remote function (a task that will run in parallel)
@ray.remote
def worker_task(x):
    time.sleep(1)  # Simulate some work
    return x * x

if __name__ == "__main__":
    # Launch tasks in parallel (this will use multiple workers)
    futures = [worker_task.remote(i) for i in range(10)]

    # Gather results
    results = ray.get(futures)

    print("Results:", results)
