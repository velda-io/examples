# Welcome to Velda Ray tutorial

This environment shows how to quickly setup distributed job processing with Ray, without managing clusters or dependencies.

## Quick start with auto-scaling:
Run `ray up ray_config.yaml` to start the Ray cluster, then submit job with

```
RAY_ADDRESS=ray://ray-velda:10001 python ray_job.py
```
Or use `ray_job_gpu.py` to try with GPU tasks.

Make sure you tear down the cluster with `ray down ray_config.yaml` after use, or the head node will continue bill your usage.

To view dashboard: https://8265-ray-velda-[instanceid].i.velda.cloud
The instance ID should be in your URL if you connect with VSCode-web, or in field instance in `/run/velda/velda.yaml'.

## Setup cluster manually
Install ray & velda autoscaler add-on:
```
pip install ray[default] ray-velda
```

Follow (ray_config.yaml)[ray_config.yaml] to create a new ray config file, then run `ray up ray_config.yaml` to start the cluster.

## Start without auto-scaling
You can also start the cluster manually without auto-scaling addon:

Start head node:
```
vrun -s ray --keep-alive --tty=no ray start --head
```

Start worker nodes:
```
RAY_JOB_ID=$(vbatch -N 2 ray start --block --address=ray:6279)
```

Submit jobs:
```
RAY_ADDRESS=ray://ray:10001 python ray_job.py
```

Delete workers:
```
velda task cancel ${RAY_JOB_ID}
```