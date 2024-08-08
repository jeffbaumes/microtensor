#include "nn.h"

std::shared_ptr<Tensor> Linear::operator()(const std::shared_ptr<Tensor>& inputs) {
  if (b) {
    out = inputs % W + b;
  } else {
    out = inputs % W;
  }
  return out;
}

BatchNorm1d::BatchNorm1d(int dim, float momentum, float epsilon) : momentum(momentum), epsilon(epsilon) {
  gamma = ones({dim});
  beta = zeros({dim});
  parameters.push_back(gamma);
  parameters.push_back(beta);
  running_mean = zeros({dim});
  running_var = ones({dim});
  training = true;
}

std::shared_ptr<Tensor> BatchNorm1d::operator()(const std::shared_ptr<Tensor>& x) {
  std::shared_ptr<Tensor> x_mean, x_var;
  if (training) {
    x_mean = mean(x, {0});
    x_var = mean(pow(x - x_mean, 2.0f), {0});
  } else {
    x_mean = running_mean;
    x_var = running_var;
  }
  // xhat = (x - xmean) / torch.sqrt(xvar + self.eps)
  auto x_hat = (x - x_mean) / sqrt(x_var + epsilon);
  out = gamma * x_hat + beta;
  if (training) {
    NoGrad _;
    running_mean = running_mean * (1.0f - momentum) + x_mean * momentum;
    running_var = running_var * (1.0f - momentum) + x_var * momentum;
  }
  return out;
}

std::shared_ptr<Tensor> Tanh::operator()(const std::shared_ptr<Tensor>& x) {
  out = tanh(x);
  return out;
}

std::shared_ptr<Tensor> MLP::operator()(const std::shared_ptr<Tensor>& x) {
  auto outputs = x;
  for (int i = 0; i < layers.size(); ++i) {
    outputs = layers[i](outputs);
  }
  return outputs;
}
