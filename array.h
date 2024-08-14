#pragma once

#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <memory>   // For std::shared_ptr
#include <numeric>  // For std::accumulate
#include <random>
#include <unordered_set>
#include <vector>

class Slice {
 public:
  Slice(int start, int stop);
  Slice(int idx);
  int start, stop;
  bool direct;
};

class Array : public std::enable_shared_from_this<Array> {
 public:
  std::vector<float> data;
  std::vector<int> shape;
  std::vector<int> strides;

  Array();
  Array(
    const std::vector<float>& data,
    const std::vector<int>& shape
  );
  Array(
    std::shared_ptr<Array> parent,
    int offset,
    const std::vector<int>& shape,
    const std::vector<int>& strides
  );

  int nelement();
  std::shared_ptr<Array> view(const std::vector<int>& shape);
  std::shared_ptr<Array> operator[](int index);
  std::shared_ptr<Array> slice(const std::vector<Slice>& slices);
  std::shared_ptr<Array> index(const std::vector<std::shared_ptr<Array>>& indices);
  void print(const std::string& indent = "");
};

void calculate_strides(const std::vector<int>& shape, std::vector<int>& strides);

std::shared_ptr<Array> array_arange(float start, float stop, float step = 1);
std::shared_ptr<Array> array_from_vector(const std::vector<float>& data, const std::vector<int>& shape);
std::shared_ptr<Array> broadcast_op(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b, bool assign, std::function<float(float, float)> op);
std::shared_ptr<Array> broadcast_add(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b, bool assign);
std::shared_ptr<Array> broadcast_mult(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b, bool assign);
std::shared_ptr<Array> operator*(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b);
std::shared_ptr<Array> operator*(const std::shared_ptr<Array>& a, float b);
std::shared_ptr<Array> operator*(float a, const std::shared_ptr<Array>& b);
std::shared_ptr<Array> operator/(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b);
std::shared_ptr<Array> operator/(const std::shared_ptr<Array>& a, float b);
std::shared_ptr<Array> operator/(float a, const std::shared_ptr<Array>& b);
std::shared_ptr<Array> operator+(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b);
std::shared_ptr<Array> operator+(const std::shared_ptr<Array>& a, float b);
std::shared_ptr<Array> operator+(float a, const std::shared_ptr<Array>& b);
std::shared_ptr<Array> operator-(const std::shared_ptr<Array>& a);
std::shared_ptr<Array> operator-(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b);
std::shared_ptr<Array> operator-(const std::shared_ptr<Array>& a, float b);
std::shared_ptr<Array> operator-(float a, const std::shared_ptr<Array>& b);
std::shared_ptr<Array> map_function(const std::shared_ptr<Array>& a, std::function<float(const std::vector<int>&, float)> op);
std::shared_ptr<Array> tanh(const std::shared_ptr<Array>& a);
std::shared_ptr<Array> exp(const std::shared_ptr<Array>& a);
std::shared_ptr<Array> log(const std::shared_ptr<Array>& a);
std::shared_ptr<Array> pow(const std::shared_ptr<Array>& a, float b);
std::shared_ptr<Array> sqrt(const std::shared_ptr<Array>& a);
std::shared_ptr<Array> one_hot(const std::shared_ptr<Array>& x, int num_classes = -1);
std::shared_ptr<Array> max(const std::shared_ptr<Array>& a, const std::vector<int>& dims = {});
std::shared_ptr<Array> sum(const std::shared_ptr<Array>& a, const std::vector<int>& dims = {});
std::shared_ptr<Array> mean(const std::shared_ptr<Array>& a, const std::vector<int>& dims = {});
std::shared_ptr<Array> variance(const std::shared_ptr<Array>& a, const std::vector<int>& dims = {});
std::shared_ptr<Array> multiply_transpose(const std::shared_ptr<Array>& a, bool a_transpose, const std::shared_ptr<Array>& b, bool b_transpose);
std::shared_ptr<Array> operator%(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b);
std::shared_ptr<Array> squeeze(const std::shared_ptr<Array>& x);
