# Triton inference memory allocation

All tests were launched on an Nvidia Tesla V100 32GB.

```bash
bash scripts/run.sh
```

## The dummy model

In this example, we experience a behaviour similar to a memory leak. When under a lot of calls, the model grows in VRAM and the memory is never freed after the effort.
By reading [this doc](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/model_management.html#model-control-mode-explicit) we know it could be the memory allocator strategy. Is there something we could do for switching strategy of the memory allocator in the case of VRAM ?

```bash
# LD_PRELOAD="/usr/lib/$(uname -m)-linux-gnu/libtcmalloc.so.4:${LD_PRELOAD}"
LD_PRELOAD="/usr/lib/$(uname -m)-linux-gnu/libjemalloc.so:${LD_PRELOAD}"
exec env LD_PRELOAD=$LD_PRELOAD "$@"
```

## The pyannote model

Under stress and for a specific config, some pyannote processes disapear on version 22.12. The same config but with version 23.08, the processes do not disappear. Was this really fixed ? or is it just luck that our configs are behaving ok ?

Also, models are now loaded concurrently which generates a race for writing files here. The flag `--model-load-thread-count=1` is supposed to serialize the loading, but this is not the case. We have forced the serialization in [model.py](models/pyannote_diarization/1/model.py) by spleeping proportionally to the instance id.

## The MFCC computations

TBC

