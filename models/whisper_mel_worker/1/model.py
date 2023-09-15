"""
whisper mel worker
"""
import time

import numpy as np
import torch
import triton_python_backend_utils as pb_utils

from .mel_module import MelCompute

AUDIO_INPUT = "audio_input"
CONTENT_FRAMES_OUTPUT = "content_frames_output"
MEL_OUTPUT = "mel_output"
MODEL_INSTANCE_KIND = "model_instance_kind"
MODEL_INSTANCE_DEVICE_ID = "model_instance_device_id"
MODEL_INSTANCE_NAME = "model_instance_name"



class TritonPythonModel:
    def initialize(self, args):
        # You must parse model_config. JSON string is not parsed here
        # self.model_config = json.loads(args['model_config'])
        model_instance_kind = args[MODEL_INSTANCE_KIND]
        model_instance_device_id = args[MODEL_INSTANCE_DEVICE_ID]
        self.instance = args[MODEL_INSTANCE_NAME]
        self.device = torch.device(
            f"cuda:{model_instance_device_id}"
            if model_instance_kind == "GPU"
            else "cpu"
        )

        self.mel_compute = MelCompute(
            mel_filters_path="/models/whisper_mel_worker/1/hyperparams/mel_filters.npz",
            device=self.device,
        )
        self.logger = pb_utils.Logger
        self.logger.log_verbose(f"{self.instance} initialized")

    # You must add the Python 'async' keyword to the beginning of `execute`
    # function if you want to use `async_exec` function.
    async def execute(self, requests):
        responses = []
        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        init_time = time.time()
        for request in requests:
            # Get audio
            audio_input = (
                pb_utils.get_input_tensor_by_name(request, AUDIO_INPUT)
                .as_numpy()
                .squeeze(0)
            )
            audio_input = torch.from_numpy(audio_input).to(self.device)
            audio_input, content_frames = self.mel_compute(audio_input)

            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        MEL_OUTPUT,
                        np.array(
                            audio_input.unsqueeze(0).cpu().numpy(),
                            dtype=np.float32,
                        ),
                    ),
                    pb_utils.Tensor(
                        CONTENT_FRAMES_OUTPUT,
                        np.array([[content_frames]], dtype=np.int32),
                    ),
                ]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        end_time = time.time()
        self.logger.log_verbose(
            f"{self.instance} : end : {end_time - init_time}"
        )
        return responses

    def finalize(self):
        self.logger.log_verbose(f"{self.instance} finalized")
