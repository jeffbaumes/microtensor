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
  int start, stop;
  bool direct;
  Slice(int start, int stop) : start(start), stop(stop), direct(false) {}
  // Special constructor to handle single indices (for direct indexing)
  Slice(int idx) : start(idx), stop(idx + 1), direct(true) {}
};

class Array;

std::shared_ptr<Array> array_from_vector(const std::vector<float>& data, const std::vector<int>& shape) {
  return std::make_shared<Array>(data, shape);
}

class Array : public std::enable_shared_from_this<Array> {
 public:
  std::vector<float> data;
  std::vector<int> shape;
  std::vector<int> strides;

  Array() {}

  Array(
    const std::vector<float>& data,
    const std::vector<int>& shape
  ) : data(data), shape(shape) {
    calculate_strides(shape, strides);
  }

  // Constructor for a sub-tensor
  Array(
    std::shared_ptr<Array> parent,
    int offset,
    const std::vector<int>& shape,
    const std::vector<int>& strides
  ) : data(std::vector<float>(parent->data.begin() + offset, parent->data.end())),
      shape(shape),
      strides(strides) {}

  std::shared_ptr<Array> operator[](int index) {
    int offset = index * strides[0];
    if (shape.size() == 1) {
      std::vector<float> scalarData = {data[offset]};
      std::vector<int> scalarShape = {1};
      return std::make_shared<Array>(scalarData, scalarShape);
    } else {
      std::vector<int> newShape(shape.begin() + 1, shape.end());
      std::vector<int> newStrides(strides.begin() + 1, strides.end());
      return std::make_shared<Array>(shared_from_this(), offset, newShape, newStrides);
    }
  }

  std::shared_ptr<Array> slice(const std::vector<Slice>& slices) {
    if (slices.size() > shape.size()) {
      throw std::invalid_argument("More slices provided than tensor dimensions.");
    }

    std::vector<int> newShape, newStrides;
    int offset = 0;
    for (size_t i = 0; i < shape.size(); ++i) {
      Slice currentSlice = i < slices.size() ? slices[i] : Slice(0, shape[i]);
      if (currentSlice.direct) {
        offset += currentSlice.start * strides[i];
        continue;
      }
      if (currentSlice.stop == -1) {
        currentSlice.stop = shape[i];
      }
      newShape.push_back(currentSlice.stop - currentSlice.start);
      newStrides.push_back(strides[i]);
      offset += currentSlice.start * strides[i];
    }
    return std::make_shared<Array>(shared_from_this(), offset, newShape, newStrides);
  }

  void calculate_strides(const std::vector<int>& shape, std::vector<int>& strides) {
    strides.resize(shape.size());
    int stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides[i] = stride;
      stride *= shape[i];
    }
  }

  void print(const std::string& indent = "") {
    if (shape.size() == 1) {
      // Base case: 1D tensor
      std::cout << "[";
      for (int i = 0; i < shape[0]; ++i) {
        std::cout << data[i * strides[0]];
        if (i < shape[0] - 1) std::cout << ", ";
      }
      std::cout << "]";
    } else {
      // Recursive case: N-D tensor
      std::cout << "[\n";
      int subTensorSize = shape[0];
      for (int i = 0; i < subTensorSize; ++i) {
        std::cout << indent << "  ";
        auto subTensor = (*this)[i];
        subTensor->print(indent + "  ");
        if (i < subTensorSize - 1) std::cout << ",\n";
      }
      std::cout << "\n"
                << indent << "]";
    }
    if (indent.length() == 0) {
      std::cout << "\n";
    }
  }
};

std::shared_ptr<Array> map_function(const std::shared_ptr<Array>& a, std::function<float(float)> op) {
  auto a_shape = a->shape;
  auto a_data = a->data;
  auto a_strides = a->strides;

  size_t nelements = std::accumulate(a_shape.begin(), a_shape.end(), 1, std::multiplies<int>());

  std::vector<float> result(nelements);

  for (size_t i = 0; i < nelements; ++i) {
    size_t a_index = 0;
    size_t remainder = i;
    for (size_t dim = 0; dim < a_shape.size(); ++dim) {
      size_t a_stride = a_strides[dim];
      size_t dim_index = remainder / std::accumulate(a_shape.begin() + dim + 1, a_shape.end(), 1, std::multiplies<int>());
      remainder %= std::accumulate(a_shape.begin() + dim + 1, a_shape.end(), 1, std::multiplies<int>());
      a_index += dim_index * a_stride;
    }
    result[i] = op(a_data[a_index]);
  }

  return std::make_shared<Array>(result, a_shape);
}

std::shared_ptr<Array> tanhf(const std::shared_ptr<Array>& a) {
  return map_function(a, [](float x) { return std::tanh(x); });
}

std::shared_ptr<Array> expf(const std::shared_ptr<Array>& a) {
  return map_function(a, [](float x) { return std::exp(x); });
}

std::shared_ptr<Array> powf(const std::shared_ptr<Array>& a, float b) {
  return map_function(a, [b](float x) { return std::pow(x, b); });
}

std::shared_ptr<Array> broadcast_op(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b, bool assign, std::function<float(float, float)> op) {
  // Determine the result shape
  size_t maxDims = std::max(a->shape.size(), b->shape.size());
  std::vector<int> out_shape(maxDims);
  for (auto& val : out_shape) {
    val = 1;
  }
  for (size_t i = 0; i < maxDims; ++i) {
    int a_dim = i < a->shape.size() ? a->shape[a->shape.size() - 1 - i] : 1;
    int b_dim = i < b->shape.size() ? b->shape[b->shape.size() - 1 - i] : 1;
    if (a_dim != b_dim && a_dim != 1 && b_dim != 1) {
      throw std::invalid_argument("Shapes are not broadcast compatible.");
    }
    out_shape[maxDims - 1 - i] = std::max(a_dim, b_dim);
  }

  // Calculate the total number of elements based on the result shape
  std::vector<float> out_data;
  size_t totalElements = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>());
  if (!assign) {
    out_data.resize(totalElements);
  }

  // Adjust strides for broadcasting
  std::vector<size_t> a_broadcast_strides(maxDims, 0), b_broadcast_strides(maxDims, 0);
  for (size_t i = 0; i < maxDims; ++i) {
    if (i < a->shape.size() && a->shape[a->shape.size() - 1 - i] == out_shape[maxDims - 1 - i]) {
      a_broadcast_strides[maxDims - 1 - i] = a->strides[a->shape.size() - 1 - i];
    }
    if (i < b->shape.size() && b->shape[b->shape.size() - 1 - i] == out_shape[maxDims - 1 - i]) {
      b_broadcast_strides[maxDims - 1 - i] = b->strides[b->shape.size() - 1 - i];
    }
  }

  for (size_t i = 0; i < totalElements; ++i) {
    size_t indexA = 0, indexB = 0, remainder = i;
    for (size_t dim = 0; dim < maxDims; ++dim) {
      size_t dimIndex = remainder / std::accumulate(out_shape.begin() + dim + 1, out_shape.end(), 1, std::multiplies<int>());
      remainder %= std::accumulate(out_shape.begin() + dim + 1, out_shape.end(), 1, std::multiplies<int>());

      if (dim < a->shape.size()) {
        indexA += dimIndex * a_broadcast_strides[dim];
      }
      if (dim < b->shape.size()) {
        indexB += dimIndex * b_broadcast_strides[dim];
      }
    }
    if (assign) {
      a->data[indexA] = op(a->data[indexA], b->data[indexB]);
    } else {
      out_data[i] = op(a->data[indexA], b->data[indexB]);
    }
  }

  if (assign) {
    return a;
  }
  return std::make_shared<Array>(out_data, out_shape);
}

std::shared_ptr<Array> operator*(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b) {
  return broadcast_op(a, b, false, std::multiplies<float>());
}

std::shared_ptr<Array> operator*(const std::shared_ptr<Array>& a, float b) {
  return a * array_from_vector({b}, {1});
}

std::shared_ptr<Array> operator*(float a, const std::shared_ptr<Array>& b) {
  return array_from_vector({a}, {1}) * b;
}

std::shared_ptr<Array> operator/(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b) {
  return a * powf(b, -1.0f);
}

std::shared_ptr<Array> operator/(const std::shared_ptr<Array>& a, float b) {
  return a * powf(b, -1.0f);
}

std::shared_ptr<Array> operator/(float a, const std::shared_ptr<Array>& b) {
  return a * powf(b, -1.0f);
}

std::shared_ptr<Array> operator+(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b) {
  return broadcast_op(a, b, false, std::plus<float>());
}

std::shared_ptr<Array> operator+(const std::shared_ptr<Array>& a, float b) {
  return a + array_from_vector({b}, {1});
}

std::shared_ptr<Array> operator+(float a, const std::shared_ptr<Array>& b) {
  return array_from_vector({a}, {1}) + b;
}

std::shared_ptr<Array> operator-(const std::shared_ptr<Array>& a) {
  return (-1.0f) * a;
}

std::shared_ptr<Array> operator-(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b) {
  return a + (-b);
}

std::shared_ptr<Array> operator-(const std::shared_ptr<Array>& a, float b) {
  return a + (-b);
}

std::shared_ptr<Array> operator-(float a, const std::shared_ptr<Array>& b) {
  return a + (-b);
}

std::shared_ptr<Array> sum(const std::shared_ptr<Array>& a) {
  float sum = 0.0f;
  size_t totalElements = std::accumulate(a->shape.begin(), a->shape.end(), 1, std::multiplies<size_t>());

  for (size_t linearIndex = 0; linearIndex < totalElements; ++linearIndex) {
    size_t flatIndex = 0;
    size_t remainder = linearIndex;
    for (size_t dim = 0; dim < a->shape.size(); ++dim) {
      size_t dimIndex = remainder % a->shape[dim];
      remainder /= a->shape[dim];
      flatIndex += dimIndex * a->strides[dim];
    }
    sum += a->data[flatIndex];
  }

  return array_from_vector({sum}, {1});
}

std::shared_ptr<Array> multiply_transpose(const std::shared_ptr<Array>& a, bool a_transpose, const std::shared_ptr<Array>& b, bool b_transpose) {
  if (a->shape.size() != 2 || b->shape.size() != 2) {
    throw std::invalid_argument("Matrix multiplication requires two 2D tensors.");
  }
  if (a->shape[a_transpose ? 0 : 1] != b->shape[b_transpose ? 1 : 0]) {
    throw std::invalid_argument("Tensor shapes are not compatible for matrix multiplication.");
  }
  std::vector<float> result;
  int m = a->shape[a_transpose ? 1 : 0];
  int n = a->shape[a_transpose ? 0 : 1];
  int p = b->shape[b_transpose ? 0 : 1];
  int a_stride0 = a->strides[0], a_stride1 = a->strides[1];
  int b_stride0 = b->strides[0], b_stride1 = b->strides[1];
  auto a_data = a->data;
  auto b_data = b->data;
  result.resize(m * p);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < p; ++j) {
      float dotProduct = 0;
      for (int k = 0; k < n; ++k) {
        float a_val = a_transpose ? a_data[k * a_stride0 + i * a_stride1] : a_data[i * a_stride0 + k * a_stride1];
        float b_val = b_transpose ? b_data[j * b_stride0 + k * b_stride1] : b_data[k * b_stride0 + j * b_stride1];
        dotProduct += a_val * b_val;
      }
      result[i * p + j] = dotProduct;
    }
  }
  return array_from_vector(result, {m, p});
}

std::shared_ptr<Array> operator%(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b) {
  return multiply_transpose(a, false, b, false);
}

class Tensor;

std::shared_ptr<Tensor> from_vector(const std::vector<float>& data, const std::vector<int>& shape) {
  return std::make_shared<Tensor>(std::make_shared<Array>(data, shape));
}

std::shared_ptr<Tensor> from_array(const std::shared_ptr<Array>& data) {
  return std::make_shared<Tensor>(data);
}

class Tensor : public std::enable_shared_from_this<Tensor> {
 public:
  std::shared_ptr<Array> data;
  std::shared_ptr<Array> grad;
  std::vector<std::shared_ptr<Tensor>> children;
  std::function<void()> backprop;

  Tensor(const std::shared_ptr<Array>& data)
      : data(data), grad() {}

  Tensor(const std::shared_ptr<Array>& data, const std::vector<std::shared_ptr<Tensor>>& children, std::function<void()> backprop)
      : data(data), grad(), children(children), backprop(backprop) {}

  // TODO: Needs backprop function
  std::shared_ptr<Tensor> operator[](int index) {
    return std::make_shared<Tensor>((*data)[index]);
  }

  // TODO: Needs backprop function
  std::shared_ptr<Tensor> slice(const std::vector<Slice>& slices) {
    return std::make_shared<Tensor>(data->slice(slices));
  }

  void print(const std::string& indent = "") {
    data->print(indent);
  }

  void init_grad() {
    int nelements = std::accumulate(data->shape.begin(), data->shape.end(), 1, std::multiplies<int>());
    if (!grad) {
      grad = std::make_shared<Array>(std::vector<float>(nelements), data->shape);
    } else if (grad->shape != data->shape) {
      grad->data.resize(nelements);
      grad->shape = data->shape;
    }
  }

  void backward() {
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

 private:
  void topological_sort(std::vector<std::shared_ptr<Tensor>>& sorted) {
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

  void backward_step() {
    for (auto& child : children) {
      child->init_grad();
    }
    if (backprop) {
      backprop();
    }
  }
};

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

std::shared_ptr<Tensor> sum(const std::shared_ptr<Tensor>& a) {
  auto result = from_array(sum(a->data));
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

template <typename Engine>
std::shared_ptr<Tensor> randn(const std::vector<int>& shape, Engine& engine) {
  std::normal_distribution<float> distribution(0.0, 1.0);
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  for (auto& val : data) {
    val = distribution(engine);
  }
  return from_vector(data, shape);
}

std::shared_ptr<Tensor> zeros(const std::vector<int>& shape) {
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()));
  return from_vector(data, shape);
}

std::shared_ptr<Tensor> ones(const std::vector<int>& shape) {
  std::vector<float> data(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()), 1.0f);
  return from_vector(data, shape);
}

class Layer {
 public:
  template <typename Engine>
  Layer(int numInputs, int numNeurons, Engine& engine) {
    W = randn({numInputs, numNeurons}, engine);
    b = randn({1, numNeurons}, engine);
  }

  std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& inputs) {
    return tanhf(inputs % W + b);
  }

  std::shared_ptr<Tensor> W;
  std::shared_ptr<Tensor> b;
};

class MLP {
 public:
  template <typename Engine>
  MLP(int numInputs, std::vector<int> numOutputs, Engine& engine) {
    layers = std::vector<Layer>(numOutputs.size(), Layer(numInputs, numOutputs[0], engine));
    for (int i = 1; i < numOutputs.size(); ++i) {
      layers[i] = Layer(numOutputs[i - 1], numOutputs[i], engine);
    }
  }

  std::shared_ptr<Tensor> operator()(const std::shared_ptr<Tensor>& inputs) {
    auto outputs = inputs;
    for (int i = 0; i < layers.size(); ++i) {
      outputs = layers[i](outputs);
    }
    return outputs;
  }

  std::vector<Layer> layers;
};

int main() {
  // {
  //   auto m = std::make_shared<Tensor>(Tensor({0, 1, 2, 2, 1, 0}, {1, 2, 3, 1}));
  //   m->print();
  // }

  // {
  //   auto m1 = std::make_shared<Tensor>(Tensor({0, 1, 2, 2, 1, 0}, {2, 3}));
  //   auto m2 = std::make_shared<Tensor>(Tensor({1, 1, 1, 2, 1, 3}, {3, 2}));
  //   m2 = m2->slice({{0, -1}, {0, 1}});
  //   auto result = m1 % m2;
  //   std::cout << "m1:" << std::endl;
  //   m1->print();
  //   std::cout << "m2:" << std::endl;
  //   m2->print();
  //   std::cout << "result:" << std::endl;
  //   result->print();
  //   assert(result->data == std::vector<float>({3, 3}));
  //   assert(result->shape == std::vector<int>({2, 1}));
  // }

  // {
  //   auto tensor = Tensor({1, 2, 3, 4, 5, 6}, {3, 2});

  //   std::cout << "Original Tensor:" << std::endl;
  //   for (int i = 0; i < tensor.shape[0]; ++i) {
  //     for (int j = 0; j < tensor.shape[1]; ++j) {
  //       std::cout << tensor.slice({{i}, {j}})->data[0] << " ";
  //     }
  //     std::cout << std::endl;
  //   }

  //   auto subTensor = tensor.slice({{1, -1}, {0, 1}});
  //   std::cout << "Sub-Tensor:" << std::endl;
  //   // std::cout << subTensor.shape.size() << std::endl;
  //   // std::cout << subTensor.shape[0] << " " << subTensor.shape[1] << std::endl;
  //   for (int i = 0; i < subTensor->shape[0]; ++i) {
  //     for (int j = 0; j < subTensor->shape[1]; ++j) {
  //       std::cout << subTensor->slice({{i}, {j}})->data[0] << " ";
  //     }
  //     std::cout << std::endl;
  //   }
  // }

  // {
  //   auto t1 = std::make_shared<Tensor>(Tensor({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, {2, 3, 2}));
  //   auto t2 = std::make_shared<Tensor>(Tensor({1, 2, 3, 4, 5, 6}, {3, 2}));
  //   std::cout << "t1:" << std::endl;
  //   t1->print();
  //   t1 = (*t1)[0];
  //   std::cout << "t1:" << std::endl;
  //   t1->print();
  //   auto sum = t1 + t2;
  //   std::cout << "sum:" << std::endl;
  //   sum->print();
  //   auto prod = t1 * t2;
  //   std::cout << "prod:" << std::endl;
  //   prod->print();
  // }

  // {
  //   auto t1 = from_vector({1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6}, {3, 2, 2});
  //   auto t2 = from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
  //   std::cout << "t1:" << std::endl;
  //   t1->print();
  //   t1 = t1->slice({{0, -1}, {0, -1}, {0}});
  //   std::cout << "t1:" << std::endl;
  //   t1->print();
  //   std::cout << "t2:" << std::endl;
  //   t2->print();
  //   auto sum = t1 + t2;
  //   std::cout << "sum:" << std::endl;
  //   sum->print();
  //   assert(sum->data->data == std::vector<float>({2, 4, 6, 8, 10, 12}));
  //   auto prod = t1 * t2;
  //   std::cout << "prod:" << std::endl;
  //   prod->print();
  //   assert(prod->data->data == std::vector<float>({1, 4, 9, 16, 25, 36}));
  // }

  // {
  //   auto x = from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
  //   std::cout << "x:" << std::endl;
  //   x->print();
  //   auto W = from_vector({1, 2, 3, 4}, {2, 2});
  //   std::cout << "W:" << std::endl;
  //   W->print();
  //   auto b = from_vector({1, 2}, {1, 2});
  //   std::cout << "b:" << std::endl;
  //   b->print();
  //   auto prod = x % W;
  //   std::cout << "x % W:" << std::endl;
  //   prod->print();
  //   auto out = prod + b;
  //   std::cout << "x % W + b:" << std::endl;
  //   out->print();
  //   out->Backward();
  //   std::cout << "x->grad:" << std::endl;
  //   x->grad->print();
  //   std::cout << "W->grad:" << std::endl;
  //   W->grad->print();
  //   std::cout << "b->grad:" << std::endl;
  //   b->grad->print();
  //   std::cout << "prod->grad:" << std::endl;
  //   prod->grad->print();
  //   std::cout << "out->grad:" << std::endl;
  //   out->grad->print();
  // }

  {
    // video 1: micrograd example

    std::default_random_engine engine(std::random_device{}());
    auto n = MLP(3, {4, 4, 1}, engine);
    auto xs = from_vector({2.0f, 3.0f, -1.0f, 3.0f, -1.0f, 0.5f, 5.0f, 1.0f, 1.0f, 1.0f, 1.0f, -1.0f}, {4, 3});
    auto ys = from_vector({1.0f, -1.0f, -1.0f, 1.0f}, {4, 1});

    std::shared_ptr<Tensor> ypred;

    for (int k = 0; k < 2000; k += 1) {
      // Forward pass
      ypred = n(xs);
      auto err = powf(ypred - ys, 2.0f);
      auto loss = sum(err);

      // Backward pass
      for (auto& layer : n.layers) {
        if (layer.W->grad) {
          layer.W->grad = nullptr;
        }
        if (layer.b->grad) {
          layer.b->grad = nullptr;
        }
      }
      loss->backward();

      // Update
      for (auto& layer : n.layers) {
        layer.W->data = layer.W->data - 0.02f * layer.W->grad;
        layer.b->data = layer.b->data - 0.02f * layer.b->grad;
      }

      std::cout << k << " " << loss->data->data[0] << std::endl;
    }

    ypred->print();
  }

  return 0;
}
