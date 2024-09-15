#include "nn.h"

std::shared_ptr<Tensor> Linear::operator()(const std::shared_ptr<Tensor>& inputs) {
  if (b) {
    out = inputs % W + b;
  } else {
    out = inputs % W;
  }
  return out;
}

std::shared_ptr<Tensor> Embedding::operator()(const std::shared_ptr<Tensor>& x) {
  return W->index({x});
}

std::shared_ptr<Tensor> Flatten::operator()(const std::shared_ptr<Tensor>& x) {
  return x->view({x->data->shape[0], -1});
}

std::shared_ptr<Tensor> FlattenConsecutive::operator()(const std::shared_ptr<Tensor>& x) {
  int batch1 = x->data->shape[0];
  int batch2 = x->data->shape[1] / n;
  int dim = x->data->shape[2] * n;
  if (batch2 == 1) {
    return x->view({batch1, dim});
  }
  return x->view({batch1, batch2, dim});
}

Sequential::Sequential(const std::vector<std::shared_ptr<Module>>& layers) : layers(layers) {
  for (auto& layer : layers) {
    for (auto& parameter : layer->parameters) {
      parameters.push_back(parameter);
    }
  }
}

std::shared_ptr<Tensor> Sequential::operator()(const std::shared_ptr<Tensor>& x) {
  auto outputs = x;
  for (int i = 0; i < layers.size(); ++i) {
    outputs = (*layers[i])(outputs);
  }
  return outputs;
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
    int n = x->data->shape.size() - 1;
    std::vector<int> dims(n);
    for (int i = 0; i < n; ++i) {
      dims[i] = i;
    }
    x_mean = mean(x, dims);
    x_var = variance_biased(x, dims);
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
  // TODO: check for contiguous memory since it's assumed

  // TODO: optimize forward pass using loops

  std::shared_ptr<Array> x_mean, x_var;
  if (training) {
    int n = x->data->shape.size() - 1;
    std::vector<int> dims(n);
    for (int i = 0; i < n; ++i) {
      dims[i] = i;
    }
    x_mean = mean(x->data, dims);
    x_var = variance_biased(x->data, dims);
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
    // n is "m"

    // ```python
    // dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
    // dbnbias = dhpreact.sum(0, keepdim=True)
    // ```
    // gamma->grad = gamma->grad sum(out->grad, {0});

    // ```python
    //   dhprebn = bngain * bnvar_inv/n * (n * dhpreact - dhpreact.sum(0) - n/(n-1) * bnraw * (dhpreact * bnraw).sum(0))
    // ```
    // Note: the `n/(n-1)` factor is removed because I'm using biased variance as described in the (possibly erroneous?) original paper
    // I'm matching pytorch outputs here.

    // To handle more batch dimensions, m is the product of all dimensions except the last one.
    // It's like we are calling view(-1, n) on the input tensor x.
    int dimensions = x->data->shape.size();
    int m = 1;
    for (int i = 0; i < dimensions - 1; ++i) {
      m *= x->data->shape[i];
    }
    int n = x->data->shape[dimensions - 1];

    auto& x_grad = x->grad->data;
    auto& beta_grad = beta->grad->data;
    auto& gamma_grad = gamma->grad->data;
    auto& gamma_data = gamma->data->data;
    auto& out_grad = out->grad->data;
    auto& bnraw_data = bnraw->data;
    auto& bnvar_inv_data = bnvar_inv->data;

    for (int j = 0; j < n; ++j) {
      // Compute the sums
      float sum1 = 0.0f;
      float sum2 = 0.0f;
      for (int i = 0, ind = j; i < m; ++i, ind += n) {
        sum1 += out_grad[ind];
        sum2 += out_grad[ind] * bnraw_data[ind];
      }

      // Compute dhprebn
      float factor = gamma_data[j] * bnvar_inv_data[j] / m;
      for (int i = 0, ind = j; i < m; ++i, ind += n) {
        float val = factor * (m * out_grad[ind] - sum1 - bnraw_data[ind] * sum2);
        x_grad[ind] += val;
      }
      beta_grad[j] += sum1;
      gamma_grad[j] += sum2;
    }
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
