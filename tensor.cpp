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
  // assert(indices.size() == data->shape.size());
  // for (auto& index : indices) {
  //   assert(index->data->shape.size() == 1);
  // }
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
    // Iterate over all indices add the gradient of the output to the gradient of the input
    int index_elements = std::accumulate(array_indices[0]->shape.begin(), array_indices[0]->shape.end(), 1, std::multiplies<int>());
    int lookup_elements = std::accumulate(data->shape.begin() + array_indices.size(), data->shape.end(), 1, std::multiplies<int>());
    for (int i = 0; i < index_elements; i += 1) {
      size_t this_data_index = 0;
      for (size_t indices_index = 0; indices_index < array_indices.size(); ++indices_index) {
        auto index = array_indices[indices_index];
        size_t index_data_index = 0;
        size_t remainder = i;
        for (size_t dim = 0; dim < index->shape.size(); ++dim) {
          size_t dim_index = remainder / std::accumulate(index->shape.begin() + dim + 1, index->shape.end(), 1, std::multiplies<int>());
          remainder %= std::accumulate(index->shape.begin() + dim + 1, index->shape.end(), 1, std::multiplies<int>());
          index_data_index += dim_index * index->strides[dim];
        }
        this_data_index += index->data[index_data_index] * data->strides[indices_index];
      }
      for (int j = 0; j < lookup_elements; j += 1) {
        size_t this_data_offset = 0;
        size_t remainder = j;
        for (size_t dim = array_indices.size(); dim < data->shape.size(); ++dim) {
          size_t dim_index = remainder / data->strides[dim];
          remainder %= data->strides[dim];
          this_data_offset += dim_index * data->strides[dim];
        }
        grad->data[this_data_index + this_data_offset] += result->grad->data[i*lookup_elements + j];
      }
    }
    // for (int i = 0; i < array_indices[0]->shape[0]; ++i) {
    //   size_t linearIndex = 0;
    //   for (size_t dim = 0; dim < array_indices.size(); ++dim) {
    //     assert(i*array_indices[dim]->strides[0] < array_indices[dim]->data.size());
    //     assert(dim < grad->strides.size());
    //     linearIndex += array_indices[dim]->data[i*array_indices[dim]->strides[0]] * grad->strides[dim];
    //   }
    //   assert(linearIndex < grad->data.size());
    //   assert(i < result->grad->data.size());
    //   grad->data[linearIndex] += result->grad->data[i];
    // }
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
    // broadcast_add(a->grad, result->grad, true);
  };
  return result;
}

std::shared_ptr<Tensor> max(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims) {
  auto result = from_array(max(a->data, dims));
  // TODO: I'm going to assume no gradient calculation for max, is this right?
  return result;
}

std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims) {
  float n = a->nelement();
  if (!dims.empty()) {
    n = 1.0f;
    for (int i = 0; i < dims.size(); ++i) {
      n *= a->data->shape[dims[i]];
    }
  }
  return sum(a, dims) / n;
}

std::shared_ptr<Tensor> variance(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims) {
  float n = a->nelement();
  if (!dims.empty()) {
    n = 1.0f;
    for (int i = 0; i < dims.size(); ++i) {
      n *= a->data->shape[dims[i]];
    }
  }
  return sum(pow(a - mean(a, dims), 2.0f), dims) / (n - 1.0f);
}

std::shared_ptr<Tensor> variance_biased(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims) {
  float n = a->nelement();
  if (!dims.empty()) {
    n = 1.0f;
    for (int i = 0; i < dims.size(); ++i) {
      n *= a->data->shape[dims[i]];
    }
  }
  return sum(pow(a - mean(a, dims), 2.0f), dims) / n;
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

    // auto b_mult = broadcast_mult(result->grad, b->data, false);
    // auto a_mult = broadcast_mult(result->grad, a->data, false);
    // broadcast_add(a->grad, b_mult, true);
    // broadcast_add(b->grad, a_mult, true);

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

    // broadcast_add(a->grad, result->grad, true);
    // broadcast_add(b->grad, result->grad, true);
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

std::shared_ptr<Tensor> cross_entropy_unoptimized(const std::shared_ptr<Tensor>& logits, const std::shared_ptr<Tensor>& target) {
  auto counts = exp(logits - max(logits, {1}));
  auto probs = counts / sum(counts, {1});
  auto loss = -mean(log(probs->index({arange(0, logits->data->shape[0]), target})));
  return loss;
}

std::shared_ptr<Tensor> cross_entropy(const std::shared_ptr<Tensor>& logits, const std::shared_ptr<Tensor>& target) {
  if (logits->data->shape.size() != 2) {
    throw std::runtime_error("logits must be two-dimensional");
  }
  if (target->data->shape.size() != 1) {
    throw std::runtime_error("target must be one-dimensional");
  }

  // Expanding this operation for speed
  // auto counts = exp(logits->data - max(logits->data, {1}));
  // auto probs = counts / sum(counts, {1});
  // auto result = from_array(-mean(log(probs->index({array_arange(0, logits->data->shape[0]), target->data}))));

  int n = logits->data->shape[0];
  int m = logits->data->shape[1];
  int n_stride = logits->data->strides[0];
  int m_stride = logits->data->strides[1];
  auto& logits_data = logits->data->data;
  auto& target_data = target->data->data;
  std::vector<float> probs(n*m);
  int probs_n_stride = m;
  int probs_m_stride = 1;

  float ysum = 0.0f;
  for (int i = 0; i < n; ++i) {
    float mx = std::numeric_limits<float>::lowest();
    int ind = i*n_stride;
    for (int j = 0; j < m; ++j, ind += m_stride) {
      float val = logits_data[ind];
      if (val > mx) {
        mx = val;
      }
    }
    float sum = 0.0f;
    float yval = 0.0f;
    ind = i*n_stride;
    int probs_ind = i*probs_n_stride;
    for (int j = 0; j < m; ++j, ind += m_stride, probs_ind += probs_m_stride) {
      float val = std::exp(logits_data[ind] - mx);
      probs[probs_ind] = val;
      sum += val;
      if (j == target_data[i]) {
        yval = val;
      }
    }
    probs_ind = i*probs_n_stride;
    for (int j = 0; j < m; ++j, probs_ind += probs_m_stride) {
      probs[probs_ind] /= sum;
    }
    ysum += std::log(yval / sum);
  }
  auto result = from_vector(std::vector<float>({-ysum / n}), {1});

  if (NO_GRAD) {
    return result;
  }

  result->children = {logits};
  std::weak_ptr<Tensor> logits_weak = logits;
  std::weak_ptr<Tensor> target_weak = target;
  std::weak_ptr<Tensor> result_weak = result;
  result->backprop = [logits_weak, target_weak, result_weak, probs]() {
    auto logits = logits_weak.lock();
    auto target = target_weak.lock();
    auto result = result_weak.lock();
    if (!logits || !target || !result) {
      throw std::runtime_error("one of the tensors is null");
    }

    // Expanding this operation for speed
    // logits->grad = (softmax(logits->data, {1}) - one_hot(target->data, logits->data->shape[1])) / logits->data->shape[0];

    int n = logits->data->shape[0];
    float n_inv = 1.0f / n;
    int m = logits->data->shape[1];
    int grad_n_stride = logits->grad->strides[0];
    int grad_m_stride = logits->grad->strides[1];
    int target_stride = target->data->strides[0];
    int probs_n_stride = m;
    int probs_m_stride = 1;
    auto& logits_grad = logits->grad->data;
    auto& target_data = target->data->data;
    float result_grad = result->grad->data[0];
    for (int i = 0; i < n; ++i) {
      int grad_ind = i*grad_n_stride;
      int probs_ind = i*probs_n_stride;
      int target_val = target_data[i*target_stride];
      for (int j = 0; j < m; ++j, grad_ind += grad_m_stride, probs_ind += probs_m_stride) {
        float val = probs[probs_ind];
        if (j == target_val) {
          val -= 1.0f;
        }
        logits_grad[grad_ind] = val * n_inv * result_grad;
      }
    }
  };
  return result;
}

std::shared_ptr<Tensor> softmax(const std::shared_ptr<Tensor>& logits, const std::vector<int>& dims) {
  auto counts = exp(logits - max(logits, dims));
  return counts / sum(counts, dims);
}

std::shared_ptr<Tensor> zeros(const std::vector<int>& shape) {
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  return from_vector(data, shape);
}

std::shared_ptr<Tensor> ones(const std::vector<int>& shape) {
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()), 1.0f);
  return from_vector(data, shape);
}
