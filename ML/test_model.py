import numpy as np
import onnxruntime
from torchvision import datasets, transforms


def main():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    session = onnxruntime.InferenceSession("mnist.onnx")

    num_samples = 10
    correct = 0

    for i in range(num_samples):
        image, label = test_dataset[i]
        input_data = image.unsqueeze(0).numpy().astype(np.float32)

        outputs = session.run(["output"], {"input": input_data})
        logits = outputs[0][0]
        predicted = int(np.argmax(logits))

        correct += int(predicted == label)
        print(f"sample {i}: predicted={predicted}  actual={label}")

    accuracy = correct / num_samples
    print(f"accuracy on {num_samples} samples: {accuracy:.4f}")


if __name__ == "__main__":
    main()
