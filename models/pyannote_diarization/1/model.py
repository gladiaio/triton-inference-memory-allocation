"""
pyannote diarization
"""

import gc
import os
import time

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from pyannote.audio import Pipeline

from .pyannote_utils import get_outputs

UTF_8 = "utf-8"
AUDIO_INPUT = "audio_input"
DIARIZATION_OUTPUT = "diarization_output"
MAX_SPEAKERS_INPUT = "max_speakers_input"
MIN_SPEAKERS_INPUT = "min_speakers_input"
MODEL_INSTANCE_DEVICE_ID = "model_instance_device_id"
MODEL_INSTANCE_KIND = "model_instance_kind"
MODEL_INSTANCE_NAME = "model_instance_name"
NO_REQUEST_ID = "none"
NUM_SPEAKERS_INPUT = "num_speakers_input"
REQUEST_ID_INPUT = "request_id_input"
RETURN_EMBEDDINGS_INPUT = "return_embeddings_input"
DEFAULT_SAMPLE_RATE = 16_000


class TritonPythonModel:
    def initialize(self, args):
        instance_kind = args[MODEL_INSTANCE_KIND]
        self.device_id = args[MODEL_INSTANCE_DEVICE_ID]
        self.instance = args[MODEL_INSTANCE_NAME]
        self.device = torch.device(
            f"cuda:{self.device_id}" if instance_kind == "GPU" else "cpu"
        )
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
        instance_id = self.instance.split("_")[-1]
        time.sleep(int(instance_id))
        self.diarization_pipeline = Pipeline.from_pretrained(
            config_path, use_auth_token="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"
        )

        self.logger = pb_utils.Logger
        self.diarization_pipeline._inferences[
            "_segmentation"
        ].model.state_dict = None
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.log_verbose(f"{self.instance} initialized")

    async def execute(self, requests):
        init = time.time()
        self.logger.log_verbose(f"{self.instance} : init execution")
        responses = []
        self.logger.log_verbose(
            f"{self.instance}:len(requests):{len(requests)}"
        )
        for request in requests:
            # retrieve detect language boolean (optional)
            request_id_input = pb_utils.get_input_tensor_by_name(
                request, REQUEST_ID_INPUT
            )
            if request_id_input:
                request_id_input = request_id_input.as_numpy()
                request_id_input = request_id_input[0][0]
                request_id_input = request_id_input.decode(UTF_8)
            else:
                request_id_input = NO_REQUEST_ID
            self.logger.log_verbose(
                f"{self.instance}:{request_id_input} init request"
            )

            # get audio input (mandatory)
            audio_input = pb_utils.get_input_tensor_by_name(
                request, AUDIO_INPUT
            )
            audio_input = torch.from_numpy(audio_input.as_numpy()).to(
                self.device
            )

            # get num_speakers input (optional)
            num_speakers_input = pb_utils.get_input_tensor_by_name(
                request, NUM_SPEAKERS_INPUT
            )
            if num_speakers_input:
                num_speakers_input = num_speakers_input.as_numpy()[0][0]
            self.logger.log_verbose(
                f"{self.instance}:{request_id_input}"
                + f":num_speakers_input:{num_speakers_input=}"
            )

            # get min_speakers input (optional)
            min_speakers_input = pb_utils.get_input_tensor_by_name(
                request, MIN_SPEAKERS_INPUT
            )
            if min_speakers_input:
                min_speakers_input = min_speakers_input.as_numpy()[0][0]
            self.logger.log_verbose(
                f"{self.instance}:{request_id_input}"
                + f":min_speakers_input:{min_speakers_input=}"
            )

            # get num_speakers input (optional)
            max_speakers_input = pb_utils.get_input_tensor_by_name(
                request, MAX_SPEAKERS_INPUT
            )
            if max_speakers_input:
                max_speakers_input = max_speakers_input.as_numpy()[0][0]
            self.logger.log_verbose(
                f"{self.instance}:{request_id_input}"
                + f":max_speakers_input:{max_speakers_input=}"
            )

            # get return_embeddings input (optional)
            return_embeddings_input = pb_utils.get_input_tensor_by_name(
                request, RETURN_EMBEDDINGS_INPUT
            )
            if return_embeddings_input:
                return_embeddings_input = return_embeddings_input.as_numpy()
                return_embeddings_input = return_embeddings_input[0][0]
                return_embeddings_input = bool(return_embeddings_input)
            else:
                return_embeddings_input = False
            self.logger.log_verbose(
                f"{self.instance}:{request_id_input}"
                + f":embeddings:{return_embeddings_input}"
            )

            # compute diarization and return a list
            audio_file = {
                "waveform": audio_input,
                "sample_rate": DEFAULT_SAMPLE_RATE,
                "uri": "torch_pipe_fake",
            }
            diarization_output = self.diarization_pipeline(
                file=audio_file,
                num_speakers=num_speakers_input,
                min_speakers=min_speakers_input,
                max_speakers=max_speakers_input,
            )

            self.logger.log_verbose(
                f"{self.instance}:{request_id_input}"
                + f":doutput: {diarization_output}"
            )
            diarization_output_as_list = get_outputs(
                diarization_output, self, request_id_input
            )

            response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        DIARIZATION_OUTPUT,
                        np.array(diarization_output_as_list, dtype=object),
                    )
                ],
            )
            self.logger.log_verbose(
                f"{self.instance}:{request_id_input} end request"
            )
            responses.append(response)
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.log_verbose(
            f"{self.instance}:{request_id_input}"
            + f" end: {time.time() - init}"
        )
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is OPTIONAL. This function allows
        the model to perform any necessary clean ups before exit.
        """
        self.logger.log_verbose(f"{self.instance} finalized")
