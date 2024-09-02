#include "nn.h"

std::shared_ptr<Tensor> Linear::operator()(const std::shared_ptr<Tensor>& inputs) {
  if (b) {
    out = inputs % W + b;
  } else {
    out = inputs % W;
  }
  return out;
}

BatchNorm1dUnoptimized::BatchNorm1dUnoptimized(int dim, float momentum, float epsilon) : momentum(momentum), epsilon(epsilon) {
  gamma = ones({dim});
  beta = zeros({dim});
  parameters.push_back(gamma);
  parameters.push_back(beta);
  running_mean = zeros({dim});
  running_var = ones({dim});
}

std::shared_ptr<Tensor> BatchNorm1dUnoptimized::operator()(const std::shared_ptr<Tensor>& x) {
  std::shared_ptr<Tensor> x_mean, x_var;
  if (training) {
    x_mean = mean(x, {0});
    x_var = variance_biased(x, {0});
  } else {
    x_mean = running_mean;
    x_var = running_var;
  }
  auto x_hat = (x - x_mean) / sqrt(x_var + epsilon);
  out = gamma * x_hat + beta;
  if (training) {
    NoGrad _;
    running_mean = running_mean * (1.0f - momentum) + x_mean * momentum;
    running_var = running_var * (1.0f - momentum) + x_var * momentum;
  }
  return out;
}

BatchNorm1d::BatchNorm1d(int dim, float momentum, float epsilon) : momentum(momentum), epsilon(epsilon) {
  gamma = ones({1, dim});
  beta = zeros({1, dim});
  parameters.push_back(gamma);
  parameters.push_back(beta);
  running_mean = zeros({1, dim});
  running_var = ones({1, dim});
}

std::shared_ptr<Tensor> BatchNorm1d::operator()(const std::shared_ptr<Tensor>& x) {
  std::shared_ptr<Array> x_mean, x_var;
  if (training) {
    x_mean = mean(x->data, {0});
    x_var = variance_biased(x->data, {0});
  } else {
    x_mean = running_mean->data;
    x_var = running_var->data;
  }
  auto bnvar_inv = 1.0f / sqrt(x_var + epsilon);
  auto bnraw = (x->data - x_mean) * bnvar_inv;
  auto out_data = gamma->data * bnraw + beta->data;
  out = std::make_shared<Tensor>(out_data);
  if (training) {
    running_mean->data = running_mean->data * (1.0f - momentum) + x_mean * momentum;
    running_var->data = running_var->data * (1.0f - momentum) + x_var * momentum;
  }

  if (no_grad()) {
    return out;
  }

  out->children = {x, gamma, beta};
  std::weak_ptr<Tensor> x_weak = x;
  std::weak_ptr<Tensor> gamma_weak = gamma;
  std::weak_ptr<Tensor> beta_weak = beta;
  out->backprop = [x_weak, gamma_weak, beta_weak, bnraw, bnvar_inv, this]() {
    auto x = x_weak.lock();
    auto gamma = gamma_weak.lock();
    auto beta = beta_weak.lock();
    if (!x || !gamma || !beta) {
      throw std::runtime_error("one of the tensors is null");
    }

    // Translations from python code in nn-zero-to-hero series:
    // out->grad is "dhpreact"
    // x->grad is "dhprebn"
    // gamma->data is "bngain"
    // gamma->grad is "dbngain"
    // beta->data is "bnbias"
    // beta->grad is "dbnbias"

    // ```python
    // dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    // dbnbias = dhpreact.sum(0, keepdim=True)
    // ```
    gamma->grad = gamma->grad + sum(bnraw * out->grad, {0});
    beta->grad = beta->grad + sum(out->grad, {0});

    float n = x->data->shape[0];

    // ```python
    //   dhprebn = bngain * bnvar_inv/n * (n * dhpreact - dhpreact.sum(0) - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0))
    // ```
    // Note: the `n/(n-1)` factor is removed because I'm using biased variance as described in the (possibly erroneous?) original paper
    // I'm matching pytorch outputs here.
    auto dhprebn = gamma->data * bnvar_inv/n * (n * out->grad - sum(out->grad, {0}) - bnraw * sum(out->grad * bnraw, {0}));

    x->grad = x->grad + dhprebn;
  };

  return out;
}

std::shared_ptr<Tensor> Tanh::operator()(const std::shared_ptr<Tensor>& x) {
  out = tanh(x);
  return out;
}

std::shared_ptr<Tensor> MLP::operator()(const std::shared_ptr<Tensor>& x) {
  auto outputs = x;
  for (int i = 0; i < layers.size(); ++i) {
    outputs = (*layers[i])(outputs);
  }
  return outputs;
}
