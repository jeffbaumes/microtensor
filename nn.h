#pragma once

#include "tensor.h"

#include <memory>

class Module {
 public:
  virtual std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& x) = 0;
  std::vector<std::shared_ptr<Tensor>> parameters;
  std::shared_ptr<Tensor> out;
  bool training = true;
 private:
};

class Linear : public Module {
 public:
  template <typename Engine>
  Linear(int fan_in, int fan_out, Engine& engine, bool bias = true);

  virtual std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& x) override;

  std::shared_ptr<Tensor> W;
  std::shared_ptr<Tensor> b;
};

template <typename Engine>
Linear::Linear(int fan_in, int fan_out, Engine& engine, bool bias) {
  W = randn({fan_in, fan_out}, engine) / sqrtf(fan_in);
  parameters.push_back(W);
  if (bias) {
    b = zeros({fan_out});
    parameters.push_back(b);
  }
}


class BatchNorm1d : public Module {
public:
  BatchNorm1d(int dim, float momentum = 0.1f, float epsilon = 1.0e-5f);
  std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& x) override;

  std::shared_ptr<Tensor> gamma;
  std::shared_ptr<Tensor> beta;
  std::shared_ptr<Tensor> running_mean;
  std::shared_ptr<Tensor> running_var;
  float momentum;
  float epsilon;
};


class Tanh : public Module {
 public:
  std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& x) override;
};


class MLP : public Module {
 public:
  template <typename Engine>
  MLP(int numInputs, std::vector<int> numOutputs, Engine& engine);

  std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& x) override;

  std::vector<std::shared_ptr<Module>> layers;
};

template <typename Engine>
MLP::MLP(int numInputs, std::vector<int> numOutputs, Engine& engine) {
  layers = std::vector<std::shared_ptr<Module>>(2 * numOutputs.size(), std::make_shared<Linear>(numInputs, numOutputs[0], engine));
  layers[1] = std::make_shared<Tanh>();
  for (int i = 1; i < numOutputs.size(); i += 1) {
    layers[2 * i] = std::make_shared<Linear>(numOutputs[i - 1], numOutputs[i], engine);
    layers[2 * i + 1] = std::make_shared<Tanh>();
  }
}
