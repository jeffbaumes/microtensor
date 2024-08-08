#include "tensor.h"

bool NO_GRAD = false;

bool no_grad() {
  return NO_GRAD;
}

NoGrad::NoGrad() : previous(NO_GRAD) {
  NO_GRAD = true;
}

NoGrad::~NoGrad() {
  NO_GRAD = previous;
}


Tensor::Tensor(const std::shared_ptr<Array>& data)
    : data(data), grad() {}

Tensor::Tensor(const std::shared_ptr<Array>& data, const std::vector<std::shared_ptr<Tensor>>& children, std::function<void()> backprop)
    : data(data), grad(), children(children), backprop(backprop) {}

int Tensor::nelement() {
  return data->nelement();
}

std::shared_ptr<Tensor> Tensor::view(const std::vector<int>& shape) {
  auto result = from_array(data->view(shape));
  if (NO_GRAD) {
    return result;
  }
  result->children = {shared_from_this()};
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [this, result_weak]() {
    auto result = result_weak.lock();
    if (!result) {
      throw std::runtime_error("one of the tensors is null");
    }
    grad = grad + result->grad->view(data->shape);
  };
  return result;
}

// TODO: Needs backprop function
std::shared_ptr<Tensor> Tensor::operator[](int index) {
  return from_array((*data)[index]);
}

std::shared_ptr<Tensor> Tensor::index(const std::vector<std::shared_ptr<Tensor>>& indices) {
  std::vector<std::shared_ptr<Array>> array_indices;
  for (auto& index : indices) {
    array_indices.push_back(index->data);
  }
  auto result = from_array(data->index(array_indices));
  if (NO_GRAD) {
    return result;
  }
  result->children = {shared_from_this()};

  std::weak_ptr<Tensor> result_weak = result;

  // TODO: backprop assumes indices are N one-dimensional arrays, where N is the dim of the tensor.
  // Should generalize to M arbitrary dimensional index arrays (all of same shape), where M <= N.
  // Forward pass already implements this.
  result->backprop = [this, array_indices, result_weak]() {
    auto result = result_weak.lock();
    if (!result) {
      throw std::runtime_error("one of the tensors is null");
    }
    for (int i = 0; i < array_indices[0]->shape[0]; ++i) {
      size_t linearIndex = 0;
      for (size_t dim = 0; dim < array_indices.size(); ++dim) {
        assert(i*array_indices[dim]->strides[0] < array_indices[dim]->data.size());
        assert(dim < grad->strides.size());
        linearIndex += array_indices[dim]->data[i*array_indices[dim]->strides[0]] * grad->strides[dim];
      }
      assert(linearIndex < grad->data.size());
      assert(i < result->grad->data.size());
      grad->data[linearIndex] += result->grad->data[i];
    }
  };
  return result;
}

// TODO: Needs backprop function
std::shared_ptr<Tensor> Tensor::slice(const std::vector<Slice>& slices) {
  return from_array(data->slice(slices));
}

void Tensor::print(const std::string& indent) {
  data->print(indent);
}

void Tensor::init_grad() {
  int num = nelement();
  if (!grad) {
    grad = std::make_shared<Array>(std::vector<float>(num), data->shape);
  } else if (grad->shape != data->shape) {
    grad->data.resize(num);
    grad->shape = data->shape;
  }
}

void Tensor::backward() {
  init_grad();
  for (auto& val : grad->data) {
    val = 1.0f;
  }
  auto sorted = std::vector<std::shared_ptr<Tensor>>();
  topological_sort(sorted);
  for (auto& node : sorted) {
    node->backward_step();
  }
}

void Tensor::topological_sort(std::vector<std::shared_ptr<Tensor>>& sorted) {
  std::unordered_set<std::shared_ptr<Tensor>> visited;
  std::function<void(const std::shared_ptr<Tensor>&)> dfs = [&](const std::shared_ptr<Tensor>& node) {
    if (visited.count(node)) {
      return;
    }
    visited.insert(node);
    for (auto& child : node->children) {
      dfs(child);
    }
    sorted.push_back(node);
  };
  dfs(shared_from_this());
  std::reverse(sorted.begin(), sorted.end());
}

void Tensor::backward_step() {
  for (auto& child : children) {
    child->init_grad();
  }
  if (backprop) {
    backprop();
  }
}

std::shared_ptr<Tensor> arange(float start, float stop, float step) {
  return from_array(array_arange(start, stop, step));
}

std::shared_ptr<Tensor> from_vector(const std::vector<float>& data, const std::vector<int>& shape) {
  return from_array(array_from_vector(data, shape));
}

std::shared_ptr<Tensor> from_array(const std::shared_ptr<Array>& data) {
  return std::make_shared<Tensor>(data);
}

std::shared_ptr<Tensor> tanh(const std::shared_ptr<Tensor>& a) {
  auto result = from_array(tanh(a->data));
  if (NO_GRAD) {
    return result;
  }
  result->children = {a};
  std::weak_ptr<Tensor> a_weak = a;
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [a_weak, result_weak]() {
    auto a = a_weak.lock();
    auto result = result_weak.lock();
    if (!a || !result) {
      throw std::runtime_error("one of the tensors is null");
    }
    a->grad = a->grad + result->grad * (1.0f - result->data * result->data);
  };
  return result;
}

std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a) {
  auto result = from_array(exp(a->data));
  if (NO_GRAD) {
    return result;
  }
  result->children = {a};
  std::weak_ptr<Tensor> a_weak = a;
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [a_weak, result_weak]() {
    auto a = a_weak.lock();
    auto result = result_weak.lock();
    if (!a || !result) {
      throw std::runtime_error("one of the tensors is null");
    }
    a->grad = a->grad + result->grad * result->data;
  };
  return result;
}

std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& a) {
  auto result = from_array(log(a->data));
  if (NO_GRAD) {
    return result;
  }
  result->children = {a};
  std::weak_ptr<Tensor> a_weak = a;
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [a_weak, result_weak]() {
    auto a = a_weak.lock();
    auto result = result_weak.lock();
    if (!a || !result) {
      throw std::runtime_error("one of the tensors is null");
    }
    // Assumes a->data is positive
    a->grad = a->grad + result->grad * (1.0f / a->data);
  };
  return result;
}

std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, float b) {
  auto result = from_array(pow(a->data, b));
  if (NO_GRAD) {
    return result;
  }
  result->children = {a};
  std::weak_ptr<Tensor> a_weak = a;
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [a_weak, result_weak, b]() {
    auto a = a_weak.lock();
    auto result = result_weak.lock();
    if (!a || !result) {
      throw std::runtime_error("one of the tensors is null");
    }
    a->grad = a->grad + result->grad * b * pow(a->data, b - 1.0f);
  };
  return result;
}

std::shared_ptr<Tensor> sqrt(const std::shared_ptr<Tensor>& a) {
  return pow(a, 0.5f);
}

std::shared_ptr<Tensor> one_hot(const std::shared_ptr<Tensor>& x, int num_classes) {
  // Note: no backprop for one_hot
  return from_array(one_hot(x->data, num_classes));
}

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims) {
  auto result = from_array(sum(a->data, dims));
  if (NO_GRAD) {
    return result;
  }
  result->children = {a};
  std::weak_ptr<Tensor> a_weak = a;
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [a_weak, result_weak]() {
    auto a = a_weak.lock();
    auto result = result_weak.lock();
    if (!a || !result) {
      throw std::runtime_error("one of the tensors is null");
    }
    broadcast_op(a->grad, result->grad, true, std::plus<float>());
  };
  return result;
}

std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims) {
  if (dims.empty()) {
    return sum(a) / a->nelement();
  }
  float divisor = 1.0f;
  for (int i = 0; i < dims.size(); ++i) {
    divisor *= a->data->shape[dims[i]];
  }
  return sum(a, dims) / divisor;
}

std::shared_ptr<Tensor> variance(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims) {
  auto x_mean = mean(a, dims);
  float divisor = 1.0f;
  for (int i = 0; i < dims.size(); ++i) {
    divisor *= a->data->shape[dims[i]];
  }
  return sum(pow(a - x_mean, 2.0f), dims) / (divisor - 1.0f);
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  auto result = from_array(a->data * b->data);
  if (NO_GRAD) {
    return result;
  }
  result->children = {a, b};
  std::weak_ptr<Tensor> a_weak = a;
  std::weak_ptr<Tensor> b_weak = b;
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [a_weak, b_weak, result_weak]() {
    auto a = a_weak.lock();
    auto b = b_weak.lock();
    auto result = result_weak.lock();
    if (!a || !b || !result) {
      throw std::runtime_error("one of the tensors is null");
    }
    // a->grad = a->grad + result->grad * b->data;
    // b->grad = b->grad + result->grad * a->data;
    auto b_mult = broadcast_op(result->grad, b->data, false, std::multiplies<float>());
    auto a_mult = broadcast_op(result->grad, a->data, false, std::multiplies<float>());
    broadcast_op(a->grad, b_mult, true, std::plus<float>());
    broadcast_op(b->grad, a_mult, true, std::plus<float>());
  };
  return result;
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, float b) {
  return a * from_vector({b}, {1});
}

std::shared_ptr<Tensor> operator*(float a, const std::shared_ptr<Tensor>& b) {
  return from_vector({a}, {1}) * b;
}

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  return a * pow(b, -1.0f);
}

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, float b) {
  return a * std::pow(b, -1.0f);
}

std::shared_ptr<Tensor> operator/(float a, const std::shared_ptr<Tensor>& b) {
  return a * pow(b, -1.0f);
}

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  auto result = from_array(a->data + b->data);
  if (NO_GRAD) {
    return result;
  }
  result->children = {a, b};
  std::weak_ptr<Tensor> a_weak = a;
  std::weak_ptr<Tensor> b_weak = b;
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [a_weak, b_weak, result_weak]() {
    auto a = a_weak.lock();
    auto b = b_weak.lock();
    auto result = result_weak.lock();
    if (!a || !b || !result) {
      throw std::runtime_error("one of the tensors is null");
    }
    broadcast_op(a->grad, result->grad, true, std::plus<float>());
    broadcast_op(b->grad, result->grad, true, std::plus<float>());
  };
  return result;
}

std::shared_ptr<Tensor> operator+(const std::shared_ptr<Tensor>& a, float b) {
  return a + from_vector({b}, {1});
}

std::shared_ptr<Tensor> operator+(float a, const std::shared_ptr<Tensor>& b) {
  return from_vector({a}, {1}) + b;
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a) {
  return from_vector({-1.0f}, {1}) * a;
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  return a + (-b);
}

std::shared_ptr<Tensor> operator-(const std::shared_ptr<Tensor>& a, float b) {
  return a + (-b);
}

std::shared_ptr<Tensor> operator-(float a, const std::shared_ptr<Tensor>& b) {
  return a + (-b);
}

// Matrix multiplication operator
std::shared_ptr<Tensor> operator%(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  auto result = from_array(a->data % b->data);
  if (NO_GRAD) {
    return result;
  }
  result->children = {a, b};
  std::weak_ptr<Tensor> a_weak = a;
  std::weak_ptr<Tensor> b_weak = b;
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [a_weak, b_weak, result_weak]() {
    auto a = a_weak.lock();
    auto b = b_weak.lock();
    auto result = result_weak.lock();
    if (!a || !b || !result) {
      throw std::runtime_error("one of the tensors is null");
    }
    a->grad = a->grad + multiply_transpose(result->grad, false, b->data, true);
    b->grad = b->grad + multiply_transpose(a->data, true, result->grad, false);
  };
  return result;
}

std::shared_ptr<Tensor> squeeze(const std::shared_ptr<Tensor>& x) {
  return from_array(squeeze(x->data));
}

std::shared_ptr<Tensor> cross_entropy(const std::shared_ptr<Tensor>& logits, const std::shared_ptr<Tensor>& target) {
  auto counts = exp(logits);
  auto probs = counts / sum(counts, {1});
  auto loss = -mean(log(probs->index({arange(0, logits->data->shape[0]), target})));
  return loss;
}

std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& logits) {
  auto counts = exp(logits);
  return counts / sum(counts, {1});
}

std::shared_ptr<Tensor> zeros(const std::vector<int>& shape) {
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  return from_vector(data, shape);
}

std::shared_ptr<Tensor> ones(const std::vector<int>& shape) {
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()), 1.0f);
  return from_vector(data, shape);
}
