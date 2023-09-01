# triton-inference-memory-allocation

Git repo to demonstrate triton's buffer behavior.

## How to reproduce

### 1 - Build docker images

`docker compose -f docker/docker-compose.yaml up --build`

### 2 - Trigger the requests

This will trigger the client to send 200 requests to triton
`curl http://0.0.0.0:8888/`

Note that the model is set to wait 2s for its queue to fill-up

### 3 - Watch the VRAM not being freed

You can see that even after the request is done, the buffer is not freed.
