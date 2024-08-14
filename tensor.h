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

bool no_grad();

class NoGrad {
 public:
  NoGrad();
  ~NoGrad();
 private:
  bool previous;
};

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  std::shared_ptr<Array> data;
  std::shared_ptr<Array> grad;
  std::vector<std::shared_ptr<Tensor>> children;
  std::function<void()> backprop;

  Tensor(const std::shared_ptr<Array>& data);
  Tensor(const std::shared_ptr<Array>& data, const std::vector<std::shared_ptr<Tensor>>& children, std::function<void()> backprop);

  int nelement();
  std::shared_ptr<Tensor> view(const std::vector<int>& shape);
  std::shared_ptr<Tensor> operator[](int index);
  std::shared_ptr<Tensor> index(const std::vector<std::shared_ptr<Tensor>>& indices);
  std::shared_ptr<Tensor> slice(const std::vector<Slice>& slices);
  void print(const std::string& indent = "");
  void init_grad();
  void backward();

 private:
  void topological_sort(std::vector<std::shared_ptr<Tensor>>& sorted);
  void backward_step();
};

std::shared_ptr<Tensor> arange(float start, float stop, float step = 1);
std::shared_ptr<Tensor> from_vector(const std::vector<float>& data, const std::vector<int>& shape);
std::shared_ptr<Tensor> from_array(const std::shared_ptr<Array>& data);
std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, float b);
std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor>& a);
std::shared_ptr<Tensor> one_hot(const std::shared_ptr<Tensor>& x, int num_classes = -1);
std::shared_ptr<Tensor> max(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims = {});
std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims = {});
std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims = {});
std::shared_ptr<Tensor> variance(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims = {});
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
std::shared_ptr<Tensor> squeeze(const std::shared_ptr<Tensor>& x);
std::shared_ptr<Tensor> cross_entropy(const std::shared_ptr<Tensor>& logits, const std::shared_ptr<Tensor>& target);
std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& logits, const std::vector<int>& dims);

template <typename Engine>
std::shared_ptr<Tensor> multinomial(const std::shared_ptr<Tensor>& probs, Engine& engine) {
  auto p = squeeze(probs);
  if (p->data->shape.size() != 1) {
    throw std::runtime_error("probabilities must be one-dimensional");
  }
  auto u = std::uniform_real_distribution<float>(0, 1)(engine);
  for (float i = 0; i < p->nelement(); i += 1) {
    u -= p->index({from_vector({i}, {1})})->data->data[0];
    if (u <= 0) {
      return from_vector({i}, {1});
    }
  }
  return from_vector({static_cast<float>(p->nelement() - 1)}, {1});
}

template <typename Engine>
std::shared_ptr<Tensor> randn(const std::vector<int>& shape, Engine& engine) {
  std::normal_distribution<float> distribution(0.0, 1.0);
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  for (auto& val : data) {
    val = distribution(engine);
  }
  return from_vector(data, shape);
}

template <typename Engine>
std::shared_ptr<Tensor> randint(int low, int high, const std::vector<int>& shape, Engine& engine) {
  auto dist = std::uniform_int_distribution<int>(low, high - 1);
  auto data = std::vector<float>(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  for (int i = 0; i < data.size(); i += 1) {
    data[i] = dist(engine);
  }
  return from_vector(data, shape);
}

std::shared_ptr<Tensor> zeros(const std::vector<int>& shape);
std::shared_ptr<Tensor> ones(const std::vector<int>& shape);
