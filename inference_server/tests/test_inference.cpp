#include <gtest/gtest.h>

#include "inference/inference.hpp"

TEST(Inference, LoadsModel) {
  EXPECT_NO_THROW(Inference("models/mnist.onnx"));
}

TEST(Inference, PredictReturnsValidDigit) {
  Inference inference("models/mnist.onnx");
  std::vector<float> image(28 * 28, 0.0f);

  auto result = inference.predict(image);

  EXPECT_GE(result.digit, 0);
  EXPECT_LE(result.digit, 9);
  EXPECT_GE(result.confidence, 0.0f);
  EXPECT_LE(result.confidence, 1.0f);
}
