#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>   // For std::shared_ptr
#include <numeric>  // For std::accumulate
#include <random>
#include <unordered_set>
#include <vector>

#include "array.h"

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  std::shared_ptr<Array> data;
  std::shared_ptr<Array> grad;
  std::vector<std::shared_ptr<Tensor>> children;
  std::function<void()> backprop;

  Tensor(const std::shared_ptr<Array>& data);
  Tensor(const std::shared_ptr<Array>& data, const std::vector<std::shared_ptr<Tensor>>& children, std::function<void()> backprop);

  int nelements();
  std::shared_ptr<Tensor> operator[](int index);
  std::shared_ptr<Tensor> slice(const std::vector<Slice>& slices);
  void print(const std::string& indent = "");
  void init_grad();
  void backward();

 private:
  void topological_sort(std::vector<std::shared_ptr<Tensor>>& sorted);
  void backward_step();
};

std::shared_ptr<Tensor> from_vector(const std::vector<float>& data, const std::vector<int>& shape);
std::shared_ptr<Tensor> from_array(const std::shared_ptr<Array>& data);
std::shared_ptr<Tensor> tanhf(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> expf(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> powf(const std::shared_ptr<Tensor>& a, float b);
std::shared_ptr<Tensor> one_hot(const std::shared_ptr<Tensor>& x, int num_classes = -1);
std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims = {});
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, float b);
std::shared_ptr<Tensor> operator*(float a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, float b);
std::shared_ptr<Tensor> operator/(float a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, float b);
std::shared_ptr<Tensor> operator+(float a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, float b);
std::shared_ptr<Tensor> operator-(float a, const std::shared_ptr<Tensor>& b);
std::shared_ptr<Tensor> operator%(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b);

template <typename Engine>
std::shared_ptr<Tensor> randn(const std::vector<int>& shape, Engine& engine) {
  std::normal_distribution<float> distribution(0.0, 1.0);
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  for (auto& val : data) {
    val = distribution(engine);
  }
  return from_vector(data, shape);
}

std::shared_ptr<Tensor> zeros(const std::vector<int>& shape);
std::shared_ptr<Tensor> ones(const std::vector<int>& shape);

class Layer {
 public:
  template <typename Engine>
  Layer(int numInputs, int numNeurons, Engine& engine) {
    W = randn({numInputs, numNeurons}, engine);
    b = randn({1, numNeurons}, engine);
  }

  std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& inputs);

  std::shared_ptr<Tensor> W;
  std::shared_ptr<Tensor> b;
};

class MLP {
 public:
  template <typename Engine>
  MLP(int numInputs, std::vector<int> numOutputs, Engine& engine) {
    layers = std::vector<Layer>(numOutputs.size(), Layer(numInputs, numOutputs[0], engine));
    for (int i = 1; i < numOutputs.size(); ++i) {
      layers[i] = Layer(numOutputs[i - 1], numOutputs[i], engine);
    }
  }

  std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& inputs);

  std::vector<Layer> layers;
};
