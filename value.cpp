
#include "value.h"

float randomMinusOneToOne() {
  static std::random_device rd;
  static std::mt19937 eng(rd());
  static std::uniform_real_distribution<float> distr(-1.0f, 1.0f);

  return distr(eng);
}

Value::Value(float data) : data(data), grad(0) {}

void Value::backward() {
  grad = 1.0f;
  auto sorted = std::vector<std::shared_ptr<Value>>();
  topological_sort(sorted);
  for (auto& node : sorted) {
    if (node->backprop) {
      node->backprop();
    }
  }
}

void Value::print_tree(int indent = 0) {
  for (int i = 0; i < indent; ++i) {
    std::cout << " ";
  }
  std::cout << "data=" << data << "|grad=" << grad << std::endl;
  for (auto& child: children) {
    child->print_tree(indent + 2);
  }
}

void Value::print() {
  std::cout << "data=" << data << "|grad=" << grad << std::endl;
}

void Value::topological_sort(std::vector<std::shared_ptr<Value>>& sorted) {
  std::unordered_set<std::shared_ptr<Value>> visited;
  std::function<void(const std::shared_ptr<Value>&)> dfs = [&](const std::shared_ptr<Value>& node) {
    if (visited.count(node)) {
      return;
    }
    visited.insert(node);
    for (auto& child: node->children) {
      dfs(child);
    }
    sorted.push_back(node);
  };
  dfs(shared_from_this());
  std::reverse(sorted.begin(), sorted.end());
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  auto result = std::make_shared<Value>(a->data + b->data);
  result->children = {a, b};
  result->backprop = [a, b, result]() {
    a->grad += result->grad;
    b->grad += result->grad;
  };
  return result;
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, float b) {
  return a + std::make_shared<Value>(b);
}

std::shared_ptr<Value> operator+(float a, const std::shared_ptr<Value>& b) {
  return std::make_shared<Value>(a) + b;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a) {
  return a * -1.0f;
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  return a + (-b);
}

std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, float b) {
  return a - std::make_shared<Value>(b);
}

std::shared_ptr<Value> operator-(float a, const std::shared_ptr<Value>& b) {
  return std::make_shared<Value>(a) - b;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  auto result = std::make_shared<Value>(a->data * b->data);
  result->children = {a, b};
  result->backprop = [a, b, result]() {
    a->grad += result->grad * b->data;
    b->grad += result->grad * a->data;
  };
  return result;
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, float b) {
  return a * std::make_shared<Value>(b);
}

std::shared_ptr<Value> operator*(float a, const std::shared_ptr<Value>& b) {
  return std::make_shared<Value>(a) * b;
}

std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& a) {
  auto result = std::make_shared<Value>(std::tanh(a->data));
  result->children = {a};
  result->backprop = [a, result]() {
    a->grad += result->grad * (1.0f - result->data * result->data);
  };
  return result;
}

std::shared_ptr<Value> exp(const std::shared_ptr<Value>& a) {
  auto result = std::make_shared<Value>(std::exp(a->data));
  result->children = {a};
  result->backprop = [a, result]() {
    a->grad += result->grad * result->data;
  };
  return result;
}

std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, float b) {
  auto result = std::make_shared<Value>(std::pow(a->data, b));
  result->children = {a};
  result->backprop = [a, b, result]() {
    a->grad += result->grad * (b * std::pow(a->data, b - 1.0f));
  };
  return result;
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  return a * pow(b, -1.0f);
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, float b) {
  return a / std::make_shared<Value>(b);
}

std::shared_ptr<Value> operator/(float a, const std::shared_ptr<Value>& b) {
  return std::make_shared<Value>(a) / b;
}

Neuron::Neuron(int numInputs) {
  weights = std::vector<std::shared_ptr<Value>>(numInputs);
  for (int i = 0; i < numInputs; ++i) {
    weights[i] = std::make_shared<Value>(randomMinusOneToOne());
  }
  bias = std::make_shared<Value>(randomMinusOneToOne());
}

void Neuron::parameters(std::vector<std::shared_ptr<Value>>& params) {
  for (auto &weight : weights) {
    params.push_back(weight);
  }
  params.push_back(bias);
}

std::shared_ptr<Value> Neuron::operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
  auto sum = bias;
  for (int i = 0; i < inputs.size(); ++i) {
    sum = sum + inputs[i] * weights[i];
  }
  return tanh(sum);
}

Layer::Layer(int numInputs, int numNeurons) {
  neurons = std::vector<Neuron>(numNeurons, Neuron(numInputs));
}

void Layer::parameters(std::vector<std::shared_ptr<Value>>& params) {
  for (auto& neuron : neurons) {
    neuron.parameters(params);
  }
}

std::vector<std::shared_ptr<Value>> Layer::operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
  auto outputs = std::vector<std::shared_ptr<Value>>(neurons.size());
  for (int i = 0; i < neurons.size(); ++i) {
    outputs[i] = neurons[i](inputs);
  }
  return outputs;
}

MLP::MLP(int numInputs, std::vector<int> numOutputs) {
  layers = std::vector<Layer>(numOutputs.size(), Layer(numInputs, numOutputs[0]));
  for (int i = 1; i < numOutputs.size(); ++i) {
    layers[i] = Layer(numOutputs[i - 1], numOutputs[i]);
  }
}

void MLP::parameters(std::vector<std::shared_ptr<Value>>& params) {
  for (auto& layer : layers) {
    layer.parameters(params);
  }
}

std::vector<std::shared_ptr<Value>> MLP::operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
  auto outputs = inputs;
  for (int i = 0; i < layers.size(); ++i) {
    outputs = layers[i](outputs);
  }
  return outputs;
}
