# triton-inference-memory-allocation

Git repo to demonstrate triton's buffer behavior.

## How to reproduce

### 1 - Build docker images

`docker compose -f docker/docker-compose.yaml up --build`

### 2 - Trigger the requests

This will trigger the client to send 200 requests to triton
`curl http://127.0.0.1:8888/`

Note that the model is set to wait 2s for its queue to fill-up

### 3 - Watch the VRAM not being freed

You can see that even after the request is done, the buffer is not freed.

## Model overview

![image](https://github.com/gladiaio/triton-inference-memory-allocation/assets/43698357/7a86ef5f-f7bd-4d3d-aceb-91fa92577eff)
