#include "tensor.h"

Tensor::Tensor(const std::shared_ptr<Array>& data)
    : data(data), grad() {}

Tensor::Tensor(const std::shared_ptr<Array>& data, const std::vector<std::shared_ptr<Tensor>>& children, std::function<void()> backprop)
    : data(data), grad(), children(children), backprop(backprop) {}

int Tensor::nelements() {
  return data->nelements();
}

// TODO: Needs backprop function
std::shared_ptr<Tensor> Tensor::operator[](int index) {
  return from_array((*data)[index]);
}

std::shared_ptr<Tensor> Tensor::index(const std::vector<std::shared_ptr<Tensor>>& indices) {
  std::vector<std::shared_ptr<Array>> dataIndices;
  for (auto& index : indices) {
    dataIndices.push_back(index->data);
  }
  auto result = from_array(data->index(dataIndices));
  result->children = {shared_from_this()};
  result->backprop = [this, dataIndices, result]() {
    for (int i = 0; i < dataIndices[0]->shape[0]; ++i) {
      size_t linearIndex = 0;
      for (size_t dim = 0; dim < dataIndices.size(); ++dim) {
        assert(i*dataIndices[dim]->strides[0] < dataIndices[dim]->data.size());
        assert(dim < grad->strides.size());
        linearIndex += dataIndices[dim]->data[i*dataIndices[dim]->strides[0]] * grad->strides[dim];
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
  int num = nelements();
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
  result->children = {a};
  result->backprop = [a, result]() {
    a->grad = a->grad + result->grad * (1.0f - result->data * result->data);
  };
  return result;
}

std::shared_ptr<Tensor> exp(const std::shared_ptr<Tensor>& a) {
  auto result = from_array(exp(a->data));
  result->children = {a};
  result->backprop = [a, result]() {
    a->grad = a->grad + result->grad * result->data;
  };
  return result;
}

std::shared_ptr<Tensor> log(const std::shared_ptr<Tensor>& a) {
  auto result = from_array(log(a->data));
  result->children = {a};
  result->backprop = [a, result]() {
    // Assumes a->data is positive
    a->grad = a->grad + result->grad * (1.0f / a->data);
  };
  return result;
}

std::shared_ptr<Tensor> pow(const std::shared_ptr<Tensor>& a, float b) {
  auto result = from_array(pow(a->data, b));
  result->children = {a};
  result->backprop = [a, result, b]() {
    a->grad = a->grad + result->grad * b * pow(a->data, b - 1.0f);
  };
  return result;
}

std::shared_ptr<Tensor> one_hot(const std::shared_ptr<Tensor>& x, int num_classes) {
  // Note: no backprop for one_hot
  return from_array(one_hot(x->data, num_classes));
}

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims) {
  auto result = from_array(sum(a->data, dims));
  result->children = {a};
  result->backprop = [a, result]() {
    broadcast_op(a->grad, result->grad, true, std::plus<float>());
  };
  return result;
}

std::shared_ptr<Tensor> mean(const std::shared_ptr<Tensor>& a, const std::vector<int>& dims) {
  if (dims.empty()) {
    return sum(a) / a->nelements();
  }
  float divisor = 1.0f;
  for (int i = 0; i < dims.size(); ++i) {
    divisor *= a->data->shape[i];
  }
  return sum(a, dims) / divisor;
}

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  auto result = from_array(a->data * b->data);
  result->children = {a, b};
  result->backprop = [a, b, result]() {
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
  result->children = {a, b};
  result->backprop = [a, b, result]() {
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
  result->children = {a, b};
  result->backprop = [a, b, result]() {
    a->grad = a->grad + multiply_transpose(result->grad, false, b->data, true);
    b->grad = b->grad + multiply_transpose(a->data, true, result->grad, false);
  };
  return result;
}

std::shared_ptr<Tensor> zeros(const std::vector<int>& shape) {
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  return from_vector(data, shape);
}

std::shared_ptr<Tensor> ones(const std::vector<int>& shape) {
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()), 1.0f);
  return from_vector(data, shape);
}

std::shared_ptr<Tensor> Layer::operator()(const std::shared_ptr<Tensor>& inputs) {
  return tanh(inputs % W + b);
}

std::shared_ptr<Tensor> MLP::operator()(const std::shared_ptr<Tensor>& inputs) {
  auto outputs = inputs;
  for (int i = 0; i < layers.size(); ++i) {
    outputs = layers[i](outputs);
  }
  return outputs;
}
