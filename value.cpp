
#include "value.h"

float randomMinusOneToOne() {
  static std::random_device rd;
  static std::mt19937 eng(rd());
  static std::uniform_real_distribution<float> distr(-1.0f, 1.0f);

  return distr(eng);
}

Value::Value(float data) : data(data), grad(0), op('\0'), a_(nullptr), b_(nullptr) {}
Value::Value(float data, char op, std::shared_ptr<Value> a)
      : data(data), grad(0), op(op), a_(a), b_(nullptr) {}
Value::Value(float data, char op, std::shared_ptr<Value> a, std::shared_ptr<Value> b)
      : data(data), grad(0), op(op), a_(a), b_(b) {}

void Value::backward() {
  grad = 1.0f;
  auto sorted = std::vector<std::shared_ptr<Value>>();
  topological_sort(sorted);
  for (auto& node : sorted) {
    node->backward_step();
  }
}

void Value::print_tree(int indent = 0) {
  for (int i = 0; i < indent; ++i) {
    std::cout << " ";
  }
  std::cout << "data=" << std::fixed << std::setprecision(4) << data << "|grad=" << grad << "|op=" << op << std::endl;
  if (a_) {
    a_->print_tree(indent + 2);
  }
  if (b_) {
    b_->print_tree(indent + 2);
  }
}

void Value::print() {
  std::cout << "data=" << std::fixed << std::setprecision(4) << data << "|grad=" << grad << "|op=" << op << std::endl;
}

void Value::backward_step() {
  switch (op) {
    case '\0':
      return;
    case '+':
      a_->grad += grad;
      b_->grad += grad;
      break;
    case '*':
      a_->grad += grad * b_->data;
      b_->grad += grad * a_->data;
      break;
    case 't':
      a_->grad += grad * (1.0f - data * data);
      break;
    case 'e':
      a_->grad += grad * data;
      break;
    case 'p':
      a_->grad += grad * (b_->data * powf(a_->data, b_->data - 1.0f));
      break;
  }
}

void Value::topological_sort(std::vector<std::shared_ptr<Value>>& sorted) {
  std::unordered_set<std::shared_ptr<Value>> visited;
  std::function<void(const std::shared_ptr<Value>&)> dfs = [&](const std::shared_ptr<Value>& node) {
    if (visited.count(node)) {
      return;
    }
    visited.insert(node);
    if (node->a_) {
      dfs(node->a_);
    }
    if (node->b_) {
      dfs(node->b_);
    }
    sorted.push_back(node);
  };
  dfs(shared_from_this());
  std::reverse(sorted.begin(), sorted.end());
}

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  return std::make_shared<Value>(a->data + b->data, '+', a, b);
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
  return std::make_shared<Value>(a->data * b->data, '*', a, b);
}

std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, float b) {
  return a * std::make_shared<Value>(b);
}

std::shared_ptr<Value> operator*(float a, const std::shared_ptr<Value>& b) {
  return std::make_shared<Value>(a) * b;
}

std::shared_ptr<Value> tanhf(const std::shared_ptr<Value>& a) {
  return std::make_shared<Value>(tanhf(a->data), 't', a);
}

std::shared_ptr<Value> expf(const std::shared_ptr<Value>& a) {
  return std::make_shared<Value>(expf(a->data), 'e', a);
}

std::shared_ptr<Value> powf(const std::shared_ptr<Value>& a, float b) {
  return std::make_shared<Value>(powf(a->data, b), 'p', a, std::make_shared<Value>(b));
}

std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
  return a * powf(b, -1.0f);
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
  return tanhf(sum);
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
