import onnxruntime as rt
import numpy as np

sess = rt.InferenceSession("model.onnx")

print("Входы:")
for inp in sess.get_inputs():
    print(f"  {inp.name}: {inp.shape} {inp.type}")

print("Выходы:")
for out in sess.get_outputs():
    print(f"  {out.name}: {out.shape} {out.type}")

# тестовый запрос
x = np.array([[5.1, 3.5, 1.4, 0.2]], dtype=np.float32)
result = sess.run(None, {"input": x})
print("\nРезультат:", result)