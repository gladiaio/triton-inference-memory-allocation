import argparse
import asyncio
import random

import numpy as np
import torchaudio
import torchaudio.transforms as T
import tritonclient.http.aio as httpclient

NUMBER_OF_CALLS_DUMMY = 256
NUMBER_OF_CALLS_DIARI = 30

client = httpclient.InferenceServerClient(url="localhost:8000")

audio_for_diari, sr = torchaudio.load("clean-code.mp3")
resample_rate = 16000
resampler = T.Resample(sr, resample_rate, dtype=audio_for_diari.dtype)
audio_for_diari = resampler(audio_for_diari)


async def dummy_infer():
    input_data_numpy = np.random.rand(1, 80, 3000).astype(np.single)
    dummy_input = httpclient.InferInput(
        "dummy_input", input_data_numpy.shape, "FP32"
    )
    dummy_input.set_data_from_numpy(input_data_numpy, binary_data=True)
    dummy_output = httpclient.InferRequestedOutput(
        "dummy_output", binary_data=True
    )
    result = await client.infer(
        model_name="dummy",
        model_version="1",
        inputs=[dummy_input],
        outputs=[dummy_output],
    )
    return result


async def mel_infer():
    a = random.randint(1, 55)
    b = a + 4
    audio_input_numpy = audio_for_diari[
        0:1, a * 60 * 16000 : b * 60 * 16000
    ].numpy()
    mel_input = httpclient.InferInput(
        "audio_input", audio_input_numpy.shape, "FP32"
    )
    mel_input.set_data_from_numpy(audio_input_numpy, binary_data=True)
    mel_output = httpclient.InferRequestedOutput(
        "mel_output", binary_data=True
    )
    content_frame_output = httpclient.InferRequestedOutput(
        "content_frames_output", binary_data=True
    )
    result = await client.infer(
        model_name="whisper_mel_worker",
        model_version="1",
        inputs=[mel_input],
        outputs=[mel_output, content_frame_output],
    )
    return result


async def pyannote_infer():
    audio_input_numpy = audio_for_diari[
        0:1, 3 * 60 * 16000 : 7 * 60 * 16000
    ].numpy()

    audio_input_tensor = httpclient.InferInput(
        "audio_input", audio_input_numpy.shape, "FP32"
    )
    audio_input_tensor.set_data_from_numpy(audio_input_numpy, binary_data=True)
    diarization_output = httpclient.InferRequestedOutput(
        "diarization_output", binary_data=True
    )

    result = await client.infer(
        model_name="pyannote_diarization",
        model_version="1",
        inputs=[audio_input_tensor],
        outputs=[diarization_output],
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test",
        type=str,
        default="mel",
    )
    args = parser.parse_args()

    if args.test == "pyannote":
        # OOM VRAM and some pyannote processes go down
        results = [pyannote_infer() for _ in range(NUMBER_OF_CALLS_DIARI)]
        loop = asyncio.get_event_loop()
        results, unfinished = loop.run_until_complete(asyncio.wait(results))

        for res in results:
            diari_output = res.result().as_numpy("diarization_output")
            assert len(diari_output.shape) > 3

    if args.test == "mel":
        # test mel endpoint
        results = [mel_infer() for _ in range(NUMBER_OF_CALLS_DUMMY)]
        loop = asyncio.get_event_loop()
        results, unfinished = loop.run_until_complete(asyncio.wait(results))

        for res in results:
            mel_output = res.result().as_numpy("mel_output")
            content_frame_output = res.result().as_numpy(
                "content_frames_output"
            )
            assert len(mel_output.shape) == 3

    if args.test == "dummy":
        # grow dummy model
        results = [dummy_infer() for _ in range(NUMBER_OF_CALLS_DUMMY)]
        loop = asyncio.get_event_loop()
        results, unfinished = loop.run_until_complete(asyncio.wait(results))
