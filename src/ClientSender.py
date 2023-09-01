import os
import asyncio
import numpy as np
import tritonclient.http.aio as httpclient

from typing import Any


class ClientSender:

    def __init__(self) -> None:

        if not (
            os.environ.get("TRITON_SERVER_URL")
            and os.environ.get("TRITON_SERVER_PORT")
        ):
            raise RuntimeError("TRITON_SERVER_URL and TRITON_SERVER_PORT must be defined")

        # init variables
        triton_server_url = (
            os.environ["TRITON_SERVER_URL"] + ":" + os.environ["TRITON_SERVER_PORT"]
        )
        triton_client_timeout = int(
            os.getenv("TRITON_CLIENT_TIMEOUT", 300)
        )
        triton_client_verbosity = (
            os.getenv("TRITON_CLIENT_VERBOSITY", "0"),
        )

        # init triton client
        self.triton_client = httpclient.InferenceServerClient(
            url=triton_server_url,
            verbose=triton_client_verbosity == "1",
            conn_timeout=triton_client_timeout,
        )

    async def __call__(self, *_args: Any, **kwds: Any) -> Any:

        sender_input_data_numpy = np.random.rand(1, 80, 3000).astype(np.single)
        sender_input_data_tensor = httpclient.InferInput(
            "sender_input_data.1", sender_input_data_numpy.shape, "FP32"
        )
        sender_input_data_tensor.set_data_from_numpy(
            sender_input_data_numpy, binary_data=True
        )

        # set output
        sender_output_data = httpclient.InferRequestedOutput(
            "sender_output_data", binary_data=True
        )

        # infer
        for j in range(1):
            req = []
            for i in range(200):
                results = self.triton_client.infer(
                    model_name="sender",
                    model_version="1",
                    inputs=[sender_input_data_tensor],
                    outputs=[sender_output_data],
                )
                req.append(results)

            print(f"[*] {len(req)=}")


            req = await asyncio.gather(*req)

        req[0].get_response()

        print(f'[*] RECEIVED: {req[0].as_numpy("sender_output_data")}')

        return "Done"

    async def close(self) -> None:
        await self.triton_client.close()
