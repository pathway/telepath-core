# Telepath on Docker

## nvidia runtime

- Telepath on Docker requires the NVIDIA docker runtime
- See:
    - [https://developer.nvidia.com/blog/gpu-containers-runtime/](https://developer.nvidia.com/blog/gpu-containers-runtime/)
    - [https://github.com/NVIDIA/nvidia-docker](https://github.com/NVIDIA/nvidia-docker)

## nvidia runtime test

- This should work as a quick test:
```
docker container run --gpus all -it telepath:TELEPATH_VERSION  nvidia-smi
```

