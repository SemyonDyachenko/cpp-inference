import shutil

import onnx
import torch

from train import MnistNet


def main():
    model = MnistNet()
    model.load_state_dict(torch.load("mnist.pth", map_location="cpu"))
    model.eval()

    dummy_input = torch.randn(1, 1, 28, 28)

    torch.onnx.export(
        model,
        dummy_input,
        "mnist.onnx",
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=13,
    )

    onnx_model = onnx.load("mnist.onnx")
    onnx.checker.check_model(onnx_model)
    print("mnist.onnx is valid")

    dest = "../inference_server/models/mnist.onnx"
    shutil.copy("mnist.onnx", dest)
    print(f"copied mnist.onnx to {dest}")


if __name__ == "__main__":
    main()
