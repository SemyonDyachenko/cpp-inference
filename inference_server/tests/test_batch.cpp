#include <gtest/gtest.h>

#include "batch/batch_collector.hpp"
#include "inference/inference.hpp"

TEST(BatchCollector, ProcessesFiveRequests) {
  Inference inference("models/mnist.onnx");
  BatchCollector batch_collector(inference, 5, 50);

  std::vector<std::future<Inference::Result>> futures;
  for (int i = 0; i < 5; ++i) {
    std::vector<float> image(28 * 28, static_cast<float>(i) / 10.0f);
    futures.push_back(batch_collector.add(std::move(image)));
  }

  for (auto& future : futures) {
    auto result = future.get();
    EXPECT_GE(result.digit, 0);
    EXPECT_LE(result.digit, 9);
  }
}

TEST(BatchCollector, ReturnsResultsInRequestOrder) {
  Inference inference("models/mnist.onnx");
  BatchCollector batch_collector(inference, 5, 50);

  std::vector<std::vector<float>> images;
  for (int i = 0; i < 5; ++i) {
    images.emplace_back(28 * 28, static_cast<float>(i) / 10.0f);
  }

  std::vector<std::future<Inference::Result>> futures;
  for (auto& image : images) {
    futures.push_back(batch_collector.add(image));
  }

  auto expected = inference.predictBatch(images);
  for (size_t i = 0; i < futures.size(); ++i) {
    auto result = futures[i].get();
    EXPECT_EQ(result.digit, expected[i].digit);
  }
}
