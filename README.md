# Welcome to Velda playground

This is a playground where you can explore the power of Velda. 
You can try the examples, check-out repos or run applications here. Please only use public data in this playground.

You can also make changes like setting up Python virtual-env, install new packages, etc.

## Quick start

To run workload with GPU:

Example: 
```vrun -P gpu-t4-1 ./example_train.py```

Note currently there's about 30-seconds start-up time to allocate the GPU instance.

More instance types are customizable with self-hosted / enterprise edition.

### Docker
To run docker, you need to start the docker daemon: In a separate console, run `sudo dockerd` to start.

## More reading
[See Velda document here](https://velda.io/run.html) 