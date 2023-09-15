import torch


class Dummy(torch.nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()

        self.device = device

    def forward(self, dummy_input):
        return (
            torch.rand(
                dummy_input.size(0),
                1000,
                1280,
                device=dummy_input.device,
            )
            + dummy_input[0][0][0]
        )


def export_dummy(path):
    model = Dummy("cuda")
    dummy_input = torch.rand(1, 80, 3000, device=model.device)

    torch.onnx.export(
        model,
        dummy_input,
        path,
        verbose=True,
        do_constant_folding=True,
        opset_version=17,
        input_names=["dummy_input"],
        output_names=["dummy_output"],
        dynamic_axes={
            "dummy_input": {0: "batch_size"},
            "dummy_output": {0: "batch_size"},
        },
    )


if __name__ == "__main__":
    export_dummy("models/dummy/1/model.onnx")
