import onnx
import torch
from time import time

class Sender(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.device = kwargs.get("device", "cpu")

    def forward(self, sender_input_data, *args, **kwargs):

        # before = time()
        # for i in range(100):
        #     torch.rand(
        #         sender_input_data.size(0), sender_input_data.size(1), 1280,
        #         device=sender_input_data.device
        #     ) + torch.rand(
        #         sender_input_data.size(0), sender_input_data.size(1), 1280,
        #         device=sender_input_data.device
        #     ) * torch.rand(
        #         sender_input_data.size(0), sender_input_data.size(1), 1280,
        #         device=sender_input_data.device
        #     ) - torch.rand(
        #         sender_input_data.size(0), sender_input_data.size(1), 1280,
        #         device=sender_input_data.device
        #     )
        # print(time() - before)
        return torch.rand(
            # sender_input_data.size(0), 10, 1280, # small output
            sender_input_data.size(0), 1000, 1280, # LARGE output
            device=sender_input_data.device
        ) + sender_input_data[0][0][0]

        # return torch.rand_like(
        #     sender_input_data,
        #     device=self.device
        # )


def export_sender(path):

    model = Sender("cuda:0").to("cuda:0")
    dummy_input = torch.rand(1, 80, 3000, device="cuda:0")

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        do_constant_folding=True,
        opset_version=17,
        input_names=["sender_input_data.1"],
        output_names=["sender_output_data"],
        dynamic_axes={
            'sender_input_data.1' : {0 : 'batch_size'},
            'sender_output_data' : {0 : 'batch_size'}
        }
    )

if __name__ == "__main__":
    export_sender("models/sender/1/model.onnx")
