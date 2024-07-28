
#include <iostream>
#include <cmath>
#include <memory>
#include <vector>
#include <unordered_set>
#include <functional>
#include <algorithm>
#include <random>
#include <iomanip>
#include <span>

float randomMinusOneToOne() {
  static std::random_device rd;
  static std::mt19937 eng(rd());
  static std::uniform_real_distribution<float> distr(-1.0f, 1.0f);

  return distr(eng);
}

class Value : public std::enable_shared_from_this<Value> {
 public:
  float data;
  float grad;
  char op;
  std::shared_ptr<Value> a_;
  std::shared_ptr<Value> b_;

  Value(float data) : data(data), grad(0), op('\0'), a_(nullptr), b_(nullptr) {}
  Value(float data, char op, std::shared_ptr<Value> a)
      : data(data), grad(0), op(op), a_(a), b_(nullptr) {}
  Value(float data, char op, std::shared_ptr<Value> a, std::shared_ptr<Value> b)
      : data(data), grad(0), op(op), a_(a), b_(b) {}

  void Backward() {
    grad = 1.0f;
    auto sorted = std::vector<std::shared_ptr<Value>>();
    TopologicalSort(sorted);
    for (auto& node : sorted) {
      node->BackwardStep();
    }
  }

  void PrintTree(int indent = 0) {
    for (int i = 0; i < indent; ++i) {
      std::cout << " ";
    }
    std::cout << "data=" << std::fixed << std::setprecision(4) << data << "|grad=" << grad << "|op=" << op << std::endl;
    if (a_) {
      a_->PrintTree(indent + 2);
    }
    if (b_) {
      b_->PrintTree(indent + 2);
    }
  }

  void Print() {
    std::cout << "data=" << std::fixed << std::setprecision(4) << data << "|grad=" << grad << "|op=" << op << std::endl;
  }

  friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return std::make_shared<Value>(a->data + b->data, '+', a, b);
  }

  friend std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, float b) {
    return a + std::make_shared<Value>(b);
  }

  friend std::shared_ptr<Value> operator+(float a, const std::shared_ptr<Value>& b) {
    return std::make_shared<Value>(a) + b;
  }

  friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a) {
    return a * -1.0f;
  }

  friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a + (-b);
  }

  friend std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, float b) {
    return a - std::make_shared<Value>(b);
  }

  friend std::shared_ptr<Value> operator-(float a, const std::shared_ptr<Value>& b) {
    return std::make_shared<Value>(a) - b;
  }

  friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return std::make_shared<Value>(a->data * b->data, '*', a, b);
  }

  friend std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, float b) {
    return a * std::make_shared<Value>(b);
  }

  friend std::shared_ptr<Value> operator*(float a, const std::shared_ptr<Value>& b) {
    return std::make_shared<Value>(a) * b;
  }

  friend std::shared_ptr<Value> tanhf(const std::shared_ptr<Value>& a) {
    return std::make_shared<Value>(tanhf(a->data), 't', a);
  }

  friend std::shared_ptr<Value> expf(const std::shared_ptr<Value>& a) {
    return std::make_shared<Value>(expf(a->data), 'e', a);
  }

  friend std::shared_ptr<Value> powf(const std::shared_ptr<Value>& a, float b) {
    return std::make_shared<Value>(powf(a->data, b), 'p', a, std::make_shared<Value>(b));
  }

  friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b) {
    return a * powf(b, -1.0f);
  }

  friend std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, float b) {
    return a / std::make_shared<Value>(b);
  }

  friend std::shared_ptr<Value> operator/(float a, const std::shared_ptr<Value>& b) {
    return std::make_shared<Value>(a) / b;
  }

private:
  void BackwardStep() {
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

  void TopologicalSort(std::vector<std::shared_ptr<Value>>& sorted) {
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
};

class Neuron {
public:
  Neuron(int numInputs) {
    weights = std::vector<std::shared_ptr<Value>>(numInputs);
    for (int i = 0; i < numInputs; ++i) {
      weights[i] = std::make_shared<Value>(randomMinusOneToOne());
    }
    bias = std::make_shared<Value>(randomMinusOneToOne());
  }

  void Parameters(std::vector<std::shared_ptr<Value>>& params) {
    for (auto &weight : weights) {
      params.push_back(weight);
    }
    params.push_back(bias);
  }

  std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
    auto sum = bias;
    for (int i = 0; i < inputs.size(); ++i) {
      sum = sum + inputs[i] * weights[i];
    }
    return tanhf(sum);
  }

  std::vector<std::shared_ptr<Value>> weights;
  std::shared_ptr<Value> bias;
};

class Layer {
public:
  Layer(int numInputs, int numNeurons) {
    neurons = std::vector<Neuron>(numNeurons, Neuron(numInputs));
  }

  void Parameters(std::vector<std::shared_ptr<Value>>& params) {
    for (auto& neuron : neurons) {
      neuron.Parameters(params);
    }
  }

  std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
    auto outputs = std::vector<std::shared_ptr<Value>>(neurons.size());
    for (int i = 0; i < neurons.size(); ++i) {
      outputs[i] = neurons[i](inputs);
    }
    return outputs;
  }

  std::vector<Neuron> neurons;
};

class MLP {
public:
  MLP(int numInputs, std::vector<int> numOutputs) {
    layers = std::vector<Layer>(numOutputs.size(), Layer(numInputs, numOutputs[0]));
    for (int i = 1; i < numOutputs.size(); ++i) {
      layers[i] = Layer(numOutputs[i - 1], numOutputs[i]);
    }
  }

  void Parameters(std::vector<std::shared_ptr<Value>>& params) {
    for (auto& layer : layers) {
      layer.Parameters(params);
    }
  }

  std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs) {
    auto outputs = inputs;
    for (int i = 0; i < layers.size(); ++i) {
      outputs = layers[i](outputs);
    }
    return outputs;
  }

  std::vector<Layer> layers;
};

int main() {
  {
    auto a = std::make_shared<Value>(1.0f);
    auto b = std::make_shared<Value>(2.0f);
    auto c = std::make_shared<Value>(3.0f);
    auto d = std::make_shared<Value>(4.0f);

    auto e = a + b;
    auto f = c * d;
    auto g = tanhf(e + f);

    g->Backward();

    std::cout << "a|data=" << a->data << "|grad=" << a->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
    std::cout << "c|data=" << c->data << "|grad=" << c->grad << std::endl;
    std::cout << "d|data=" << d->data << "|grad=" << d->grad << std::endl;
    std::cout << "e|data=" << e->data << "|grad=" << e->grad << std::endl;
    std::cout << "f|data=" << f->data << "|grad=" << f->grad << std::endl;
    std::cout << "g|data=" << g->data << "|grad=" << g->grad << std::endl;
  }

  {
    auto x1 = std::make_shared<Value>(2.0f);
    auto x2 = std::make_shared<Value>(0.0f);
    auto w1 = std::make_shared<Value>(-3.0f);
    auto w2 = std::make_shared<Value>(1.0f);
    auto b = std::make_shared<Value>(6.881373587f);
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = x1w1 + x2w2;
    auto n = x1w1x2w2 + b;
    auto o = tanhf(n);

    o->Backward();

    std::cout << "x1|data=" << x1->data << "|grad=" << x1->grad << std::endl;
    std::cout << "x2|data=" << x2->data << "|grad=" << x2->grad << std::endl;
    std::cout << "w1|data=" << w1->data << "|grad=" << w1->grad << std::endl;
    std::cout << "w2|data=" << w2->data << "|grad=" << w2->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
    std::cout << "x1w1|data=" << x1w1->data << "|grad=" << x1w1->grad << std::endl;
    std::cout << "x2w2|data=" << x2w2->data << "|grad=" << x2w2->grad << std::endl;
    std::cout << "x1w1x2w2|data=" << x1w1x2w2->data << "|grad=" << x1w1x2w2->grad << std::endl;
    std::cout << "n|data=" << n->data << "|grad=" << n->grad << std::endl;
    std::cout << "o|data=" << o->data << "|grad=" << o->grad << std::endl;
  }

  {
    auto x1 = std::make_shared<Value>(2.0f);
    auto x2 = std::make_shared<Value>(0.0f);
    auto w1 = std::make_shared<Value>(-3.0f);
    auto w2 = std::make_shared<Value>(1.0f);
    auto b = std::make_shared<Value>(6.881373587f);
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = x1w1 + x2w2;
    auto n = x1w1x2w2 + b;
    auto o = (expf(2*n) - 1) / (expf(2*n) + 1);

    o->Backward();

    std::cout << "x1|data=" << x1->data << "|grad=" << x1->grad << std::endl;
    std::cout << "x2|data=" << x2->data << "|grad=" << x2->grad << std::endl;
    std::cout << "w1|data=" << w1->data << "|grad=" << w1->grad << std::endl;
    std::cout << "w2|data=" << w2->data << "|grad=" << w2->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
    std::cout << "x1w1|data=" << x1w1->data << "|grad=" << x1w1->grad << std::endl;
    std::cout << "x2w2|data=" << x2w2->data << "|grad=" << x2w2->grad << std::endl;
    std::cout << "x1w1x2w2|data=" << x1w1x2w2->data << "|grad=" << x1w1x2w2->grad << std::endl;
    std::cout << "n|data=" << n->data << "|grad=" << n->grad << std::endl;
    std::cout << "o|data=" << o->data << "|grad=" << o->grad << std::endl;
  }

  {
    auto a = std::make_shared<Value>(3.0f);
    auto b = a + a;
    b->Backward();
    std::cout << "a|data=" << a->data << "|grad=" << a->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
  }

  {
    auto a = std::make_shared<Value>(3.0f);
    auto b = a + 5;
    b->Backward();
    std::cout << "a|data=" << a->data << "|grad=" << a->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
    b->Print();
  }

  {
    auto x = std::vector<std::shared_ptr<Value>> {
      std::make_shared<Value>(2.0f),
      std::make_shared<Value>(2.0f),
      std::make_shared<Value>(-1.0f),
    };
    auto n = MLP(3, {4, 4, 1});
    auto out = n(x);
    std::cout << n(x)[0]->data << std::endl;
  }

  {
    auto n = MLP(3, {4, 4, 1});
    auto xs = std::vector<std::vector<std::shared_ptr<Value>>> {
      {
        std::make_shared<Value>(2.0f),
        std::make_shared<Value>(3.0f),
        std::make_shared<Value>(-1.0f),
      },
      {
        std::make_shared<Value>(3.0f),
        std::make_shared<Value>(-1.0f),
        std::make_shared<Value>(0.5f),
      },
      {
        std::make_shared<Value>(5.0f),
        std::make_shared<Value>(1.0f),
        std::make_shared<Value>(1.0f),
      },
      {
        std::make_shared<Value>(1.0f),
        std::make_shared<Value>(1.0f),
        std::make_shared<Value>(-1.0f),
      },
    };
    auto ys = std::vector<std::shared_ptr<Value>> {
      std::make_shared<Value>(1.0f),
      std::make_shared<Value>(-1.0f),
      std::make_shared<Value>(-1.0f),
      std::make_shared<Value>(1.0f),
    };

    auto ypred = std::vector<std::shared_ptr<Value>>(4);
    auto err = std::vector<std::shared_ptr<Value>>(4);
    auto loss = std::make_shared<Value>(0.0f);
    std::vector<std::shared_ptr<Value>> params;

    for (int k = 0; k < 2000; k += 1) {
      // Forward pass
      ypred = std::vector<std::shared_ptr<Value>> {
        n(xs[0])[0],
        n(xs[1])[0],
        n(xs[2])[0],
        n(xs[3])[0],
      };
      err = std::vector<std::shared_ptr<Value>> {
        powf(ypred[0] - ys[0], 2.0f),
        powf(ypred[1] - ys[1], 2.0f),
        powf(ypred[2] - ys[2], 2.0f),
        powf(ypred[3] - ys[3], 2.0f),
      };
      loss = err[0] + err[1] + err[2] + err[3];

      // Backward pass
      params.clear();
      n.Parameters(params);
      for (auto& p : params) {
        p->grad = 0.0f;
      }
      loss->Backward();

      // Update
      for (auto& p : params) {
        p->data += -0.02f * p->grad;
      }

      std::cout << k << " " << loss->data << std::endl;
    }

    ypred[0]->Print();
    ypred[1]->Print();
    ypred[2]->Print();
    ypred[3]->Print();
  }

  return 0;
}
