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
  return std::make_shared<Tensor>((*data)[index]);
}

// TODO: Needs backprop function
std::shared_ptr<Tensor> Tensor::slice(const std::vector<Slice>& slices) {
  return std::make_shared<Tensor>(data->slice(slices));
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

std::shared_ptr<Tensor> from_vector(const std::vector<float>& data, const std::vector<int>& shape) {
  return std::make_shared<Tensor>(std::make_shared<Array>(data, shape));
}

std::shared_ptr<Tensor> from_array(const std::shared_ptr<Array>& data) {
  return std::make_shared<Tensor>(data);
}

std::shared_ptr<Tensor> tanhf(const std::shared_ptr<Tensor>& a) {
  auto result = from_array(tanhf(a->data));
  result->children = {a};
  result->backprop = [a, result]() {
    a->grad = a->grad + result->grad * (1.0f - result->data * result->data);
  };
  return result;
}

std::shared_ptr<Tensor> expf(const std::shared_ptr<Tensor>& a) {
  auto result = from_array(expf(a->data));
  result->children = {a};
  result->backprop = [a, result]() {
    a->grad = a->grad + result->grad * result->data;
  };
  return result;
}

std::shared_ptr<Tensor> powf(const std::shared_ptr<Tensor>& a, float b) {
  auto result = from_array(powf(a->data, b));
  result->children = {a};
  result->backprop = [a, result, b]() {
    a->grad = a->grad + result->grad * b * powf(a->data, b - 1.0f);
  };
  return result;
}

std::shared_ptr<Tensor> one_hot(const std::shared_ptr<Tensor>& x, int num_classes) {
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

std::shared_ptr<Tensor> operator*(const std::shared_ptr<Tensor>& a, const std::shared_ptr<Tensor>& b) {
  auto result = from_array(a->data * b->data);
  result->children = {a, b};
  result->backprop = [a, b, result]() {
    a->grad = a->grad + result->grad * b->data;
    b->grad = b->grad + result->grad * a->data;
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
  return a * powf(b, -1.0f);
}

std::shared_ptr<Tensor> operator/(const std::shared_ptr<Tensor>& a, float b) {
  return a * powf(b, -1.0f);
}

std::shared_ptr<Tensor> operator/(float a, const std::shared_ptr<Tensor>& b) {
  return a * powf(b, -1.0f);
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
  return tanhf(inputs % W + b);
}

std::shared_ptr<Tensor> MLP::operator()(const std::shared_ptr<Tensor>& inputs) {
  auto outputs = inputs;
  for (int i = 0; i < layers.size(); ++i) {
    outputs = layers[i](outputs);
  }
  return outputs;
}
