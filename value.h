#pragma once

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

class Value : public std::enable_shared_from_this<Value> {
 public:
  float data;
  float grad;
  std::vector<std::shared_ptr<Value>> children;
  std::function<void()> backprop;

  Value(float data);

  void backward();
  void print_tree(int indent);
  void print();

private:
  void topological_sort(std::vector<std::shared_ptr<Value>>& sorted);
};

std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator+(const std::shared_ptr<Value>& a, float b);
std::shared_ptr<Value> operator+(float a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator-(const std::shared_ptr<Value>& a, float b);
std::shared_ptr<Value> operator-(float a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator*(const std::shared_ptr<Value>& a, float b);
std::shared_ptr<Value> operator*(float a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> operator/(const std::shared_ptr<Value>& a, float b);
std::shared_ptr<Value> operator/(float a, const std::shared_ptr<Value>& b);
std::shared_ptr<Value> tanh(const std::shared_ptr<Value>& a);
std::shared_ptr<Value> exp(const std::shared_ptr<Value>& a);
std::shared_ptr<Value> pow(const std::shared_ptr<Value>& a, float b);

class Neuron {
public:
  Neuron(int numInputs);
  void parameters(std::vector<std::shared_ptr<Value>>& params);
  std::shared_ptr<Value> operator()(const std::vector<std::shared_ptr<Value>>& inputs);

  std::vector<std::shared_ptr<Value>> weights;
  std::shared_ptr<Value> bias;
};

class Layer {
public:
  Layer(int numInputs, int numNeurons);
  void parameters(std::vector<std::shared_ptr<Value>>& params);
  std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs);

  std::vector<Neuron> neurons;
};

class MLP {
public:
  MLP(int numInputs, std::vector<int> numOutputs);
  void parameters(std::vector<std::shared_ptr<Value>>& params);
  std::vector<std::shared_ptr<Value>> operator()(const std::vector<std::shared_ptr<Value>>& inputs);

  std::vector<Layer> layers;
};
