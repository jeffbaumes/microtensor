#include "array.h"

#include <map>
#include <immintrin.h>
// #include <omp.h>
#include <mkl.h>

Slice::Slice(int start, int stop) : start(start), stop(stop), direct(false) {}

Slice::Slice(int idx) : start(idx), stop(idx + 1), direct(true) {}

Array::Array() {}

Array::Array(
  const std::vector<float>& data,
  const std::vector<int>& shape
) : data(data), shape(shape) {
  calculate_strides(shape, strides);
}

Array::Array(
  std::shared_ptr<Array> parent,
  int offset,
  const std::vector<int>& shape,
  const std::vector<int>& strides
) : data(std::vector<float>(parent->data.begin() + offset, parent->data.end())),
    shape(shape),
    strides(strides) {}

int Array::nelement() {
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>());
}

std::shared_ptr<Array> Array::view(const std::vector<int>& desired_shape) {
  auto view_shape = desired_shape;
  bool filled = false;
  for (int i = 0; i < view_shape.size(); ++i) {
    if (view_shape[i] == -1) {
      if (filled) {
        throw std::invalid_argument("Only one dimension can be inferred.");
      }
      int inferred = nelement();
      for (int j = 0; j < view_shape.size(); ++j) {
        if (j != i) {
          inferred /= view_shape[j];
        }
      }
      view_shape[i] = inferred;
      filled = true;
    }
  }
  if (std::accumulate(view_shape.begin(), view_shape.end(), 1, std::multiplies<int>()) != nelement()) {
    throw std::invalid_argument("Shape must have the same number of elements as the original array.");
  }
  std::vector<int> view_strides(view_shape.size());
  int stride = strides[strides.size() - 1];
  for (int i = view_shape.size() - 1; i >= 0; --i) {
    view_strides[i] = stride;
    stride *= view_shape[i];
  }
  return std::make_shared<Array>(shared_from_this(), 0, view_shape, view_strides);
}

std::shared_ptr<Array> Array::operator[](int index) {
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

std::shared_ptr<Array> Array::slice(const std::vector<Slice>& slices) {
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

std::shared_ptr<Array> Array::index(const std::vector<std::shared_ptr<Array>>& indices) {
  if (indices.empty()) {
    throw std::invalid_argument("Index list must not be empty.");
  }
  if (indices.size() > shape.size()) {
    throw std::invalid_argument("Index list size must be at most the number of dimensions.");
  }
  for (auto& index : indices) {
    if (index->shape != indices[0]->shape) {
      throw std::invalid_argument("All indices must have the same size.");
    }
  }

  // result_shape = [...indices[*]->shape, ...shape[indices.size():]]
  std::vector<int> result_shape(indices[0]->shape);
  for (size_t i = indices.size(); i < shape.size(); ++i) {
    result_shape.push_back(shape[i]);
  }

  // Iterate over all indices and insert a copy of the sub-data at each location
  std::vector<float> result_data;
  int index_elements = std::accumulate(indices[0]->shape.begin(), indices[0]->shape.end(), 1, std::multiplies<int>());
  int lookup_elements = std::accumulate(shape.begin() + indices.size(), shape.end(), 1, std::multiplies<int>());
  for (int i = 0; i < index_elements; i += 1) {
    size_t this_data_index = 0;
    for (size_t indices_index = 0; indices_index < indices.size(); ++indices_index) {
      auto index = indices[indices_index];
      size_t index_data_index = 0;
      size_t remainder = i;
      for (size_t dim = 0; dim < index->shape.size(); ++dim) {
        size_t dim_index = remainder / std::accumulate(index->shape.begin() + dim + 1, index->shape.end(), 1, std::multiplies<int>());
        remainder %= std::accumulate(index->shape.begin() + dim + 1, index->shape.end(), 1, std::multiplies<int>());
        index_data_index += dim_index * index->strides[dim];
      }
      this_data_index += index->data[index_data_index] * strides[indices_index];
    }
    for (int j = 0; j < lookup_elements; j += 1) {
      size_t this_data_offset = 0;
      size_t remainder = j;
      for (size_t dim = indices.size(); dim < shape.size(); ++dim) {
        size_t dim_index = remainder / strides[dim];
        remainder %= strides[dim];
        this_data_offset += dim_index * strides[dim];
      }
      result_data.push_back(data[this_data_index + this_data_offset]);
    }
  }
  return array_from_vector(result_data, result_shape);
}

void calculate_strides(const std::vector<int>& shape, std::vector<int>& strides) {
  strides.resize(shape.size());
  int stride = 1;
  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
}

void Array::print(const std::string& indent) {
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

std::shared_ptr<Array> array_arange(float start, float stop, float step) {
  std::vector<float> data;
  for (float i = start; i < stop; i += step) {
    data.push_back(i);
  }
  return array_from_vector(data, {static_cast<int>(data.size())});
}

std::shared_ptr<Array> array_from_vector(const std::vector<float>& data, const std::vector<int>& shape) {
  return std::make_shared<Array>(data, shape);
}

std::shared_ptr<Array> map_function(const std::shared_ptr<Array>& a, std::function<float(const std::vector<int>&,float)> op) {
  auto a_shape = a->shape;
  auto a_data = a->data;
  auto a_strides = a->strides;

  size_t nelement = a->nelement();

  std::vector<int> index(a_shape.size(), 0);

  // Precompute the products of dimensions for each dimension
  std::vector<size_t> dim_products(a_shape.size());
  for (size_t dim = 0; dim < a_shape.size(); ++dim) {
    dim_products[dim] = std::accumulate(a_shape.begin() + dim + 1, a_shape.end(), 1, std::multiplies<int>());
  }

  std::vector<float> result(nelement);

  for (size_t i = 0; i < nelement; ++i) {
    size_t a_index = 0;
    size_t remainder = i;
    for (size_t dim = 0; dim < a_shape.size(); ++dim) {
      size_t a_stride = a_strides[dim];
      index[dim] = remainder / dim_products[dim];
      remainder %= dim_products[dim];
      a_index += index[dim] * a_stride;
    }
    result[i] = op(index, a_data[a_index]);
  }

  return std::make_shared<Array>(result, a_shape);
}

std::shared_ptr<Array> tanh(const std::shared_ptr<Array>& a) {
  return map_function(a, [](const std::vector<int>&, float x) { return std::tanh(x); });
}

std::shared_ptr<Array> exp(const std::shared_ptr<Array>& a) {
  return map_function(a, [](const std::vector<int>&, float x) { return std::exp(x); });
}

std::shared_ptr<Array> log(const std::shared_ptr<Array>& a) {
  return map_function(a, [](const std::vector<int>&, float x) { return std::log(x); });
}

std::shared_ptr<Array> pow(const std::shared_ptr<Array>& a, float b) {
  return map_function(a, [b](const std::vector<int>&, float x) { return std::pow(x, b); });
}

std::shared_ptr<Array> sqrt(const std::shared_ptr<Array>& a) {
  return map_function(a, [](const std::vector<int>&, float x) { return std::sqrt(x); });
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
  std::vector<int> out_strides;
  calculate_strides(out_shape, out_strides);

  // Calculate the total number of elements based on the result shape
  std::vector<float> out_data;
  size_t totalElements = std::accumulate(out_shape.begin(), out_shape.end(), 1, std::multiplies<int>());
  if (!assign) {
    out_data.resize(totalElements);
  }

  // Adjust strides for broadcasting
  std::vector<int> a_broadcast_strides(maxDims, 0), b_broadcast_strides(maxDims, 0);
  for (int i = 0; i < maxDims; ++i) {
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
      size_t dimIndex = remainder / out_strides[dim];
      remainder %= out_strides[dim];
      indexA += dimIndex * a_broadcast_strides[dim];
      indexB += dimIndex * b_broadcast_strides[dim];
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
  return a * pow(b, -1.0f);
}

std::shared_ptr<Array> operator/(const std::shared_ptr<Array>& a, float b) {
  return a * std::pow(b, -1.0f);
}

std::shared_ptr<Array> operator/(float a, const std::shared_ptr<Array>& b) {
  return a * pow(b, -1.0f);
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

std::shared_ptr<Array> one_hot(const std::shared_ptr<Array>& x, int num_classes) {
  if (num_classes == -1) {
    int nelement = x->nelement();
    int maximum = 0;
    for (int i = 0; i < nelement; i += 1) {
      size_t remainder = i;
      std::vector<int> inputIndices(x->shape.size(), 0);
      for (size_t dim = 0; dim < x->shape.size(); ++dim) {
        inputIndices[dim] = remainder % x->shape[dim];
        remainder /= x->shape[dim];
      }
      size_t inputFlatIndex = 0;
      for (size_t dim = 0; dim < inputIndices.size(); ++dim) {
        inputFlatIndex += inputIndices[dim] * x->strides[dim];
      }
      maximum = std::max(maximum, static_cast<int>(x->data[inputFlatIndex]));
    }
    num_classes = maximum + 1;
  }
  auto shape = x->shape;
  shape.push_back(num_classes);
  auto result_data = std::vector<float>(std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()), 0.0f);
  int nelement = x->nelement();
  for (int i = 0; i < nelement; i += 1) {
    int value = static_cast<int>(x->data[i]);
    if (value > num_classes - 1) {
      throw std::runtime_error("Maximum value in x exceeds num_classes - 1");
    }
    result_data[i * num_classes + value] = 1;
  }
  return std::make_shared<Array>(result_data, shape);
}

std::shared_ptr<Array> sum(const std::shared_ptr<Array>& a, const std::vector<int>& d) {
  std::vector<int> dims;
  if (d.size() == 0) {
    for (int i = 0; i < a->shape.size(); ++i) {
      dims.push_back(i);
    }
  } else {
    dims = d;
  }

  std::vector<int> resultShape = a->shape;
  for (int dim : dims) {
    if (dim < resultShape.size()) {
      resultShape[dim] = 1;
    }
  }

  size_t totalResultElements = std::accumulate(resultShape.begin(), resultShape.end(), 1, std::multiplies<size_t>());
  std::vector<float> resultData(totalResultElements, 0.0f);

  std::vector<int> resultStrides(a->shape.size(), 0);
  int stride = 1;
  for (int i = resultShape.size() - 1; i >= 0; --i) {
    resultStrides[i] = stride;
    stride *= resultShape[i];
  }

  for (size_t linearIndex = 0; linearIndex < a->data.size(); ++linearIndex) {
    size_t remainder = linearIndex;
    std::vector<int> inputIndices(a->shape.size(), 0);
    for (size_t dim = 0; dim < a->shape.size(); ++dim) {
      inputIndices[dim] = remainder % a->shape[dim];
      remainder /= a->shape[dim];
    }

    size_t inputFlatIndex = 0;
    for (size_t dim = 0; dim < inputIndices.size(); ++dim) {
      inputFlatIndex += inputIndices[dim] * a->strides[dim];
    }

    std::vector<int> outputIndices = inputIndices;
    for (int dim : dims) {
      outputIndices[dim] = 0; // Set dimensions being summed over to 0
    }

    size_t resultFlatIndex = 0;
    for (size_t dim = 0; dim < outputIndices.size(); ++dim) {
      resultFlatIndex += outputIndices[dim] * resultStrides[dim];
    }

    resultData[resultFlatIndex] += a->data[inputFlatIndex];
  }

  return array_from_vector(resultData, resultShape);
}

std::shared_ptr<Array> max(const std::shared_ptr<Array>& a, const std::vector<int>& d) {
  std::vector<int> dims;
  if (d.size() == 0) {
    for (int i = 0; i < a->shape.size(); ++i) {
      dims.push_back(i);
    }
  } else {
    dims = d;
  }

  std::vector<int> resultShape = a->shape;
  for (int dim : dims) {
    if (dim < resultShape.size()) {
      resultShape[dim] = 1;
    }
  }

  size_t totalResultElements = std::accumulate(resultShape.begin(), resultShape.end(), 1, std::multiplies<size_t>());
  std::vector<float> resultData(totalResultElements, std::numeric_limits<float>::lowest());

  std::vector<int> resultStrides(a->shape.size(), 0);
  int stride = 1;
  for (int i = resultShape.size() - 1; i >= 0; --i) {
    resultStrides[i] = stride;
    stride *= resultShape[i];
  }

  for (size_t linearIndex = 0; linearIndex < a->data.size(); ++linearIndex) {
    size_t remainder = linearIndex;
    std::vector<int> inputIndices(a->shape.size(), 0);
    for (size_t dim = 0; dim < a->shape.size(); ++dim) {
      inputIndices[dim] = remainder % a->shape[dim];
      remainder /= a->shape[dim];
    }

    size_t inputFlatIndex = 0;
    for (size_t dim = 0; dim < inputIndices.size(); ++dim) {
      inputFlatIndex += inputIndices[dim] * a->strides[dim];
    }

    std::vector<int> outputIndices = inputIndices;
    for (int dim : dims) {
      outputIndices[dim] = 0; // Set dimensions being summed over to 0
    }

    size_t resultFlatIndex = 0;
    for (size_t dim = 0; dim < outputIndices.size(); ++dim) {
      resultFlatIndex += outputIndices[dim] * resultStrides[dim];
    }

    resultData[resultFlatIndex] = std::max(resultData[resultFlatIndex], a->data[inputFlatIndex]);
  }

  return array_from_vector(resultData, resultShape);
}

std::shared_ptr<Array> mean(const std::shared_ptr<Array>& a, const std::vector<int>& dims) {
  float n = a->nelement();
  if (!dims.empty()) {
    n = 1.0f;
    for (int i = 0; i < dims.size(); ++i) {
      n *= a->shape[dims[i]];
    }
  }
  return sum(a, dims) / n;
}

std::shared_ptr<Array> variance(const std::shared_ptr<Array>& a, const std::vector<int>& dims) {
  float n = a->nelement();
  if (!dims.empty()) {
    n = 1.0f;
    for (int i = 0; i < dims.size(); ++i) {
      n *= a->shape[dims[i]];
    }
  }
  return sum(pow(a - mean(a, dims), 2.0f), dims) / (n - 1.0f);
}

std::shared_ptr<Array> variance_biased(const std::shared_ptr<Array>& a, const std::vector<int>& dims) {
  float n = a->nelement();
  if (!dims.empty()) {
    n = 1.0f;
    for (int i = 0; i < dims.size(); ++i) {
      n *= a->shape[dims[i]];
    }
  }
  return sum(pow(a - mean(a, dims), 2.0f), dims) / n;
}

// std::shared_ptr<Array> multiply_transpose(const std::shared_ptr<Array>& a, bool a_transpose, const std::shared_ptr<Array>& b, bool b_transpose) {
//   if (a->shape.size() != 2 || b->shape.size() != 2) {
//     throw std::invalid_argument("Matrix multiplication requires two 2D tensors.");
//   }
//   if (a->shape[a_transpose ? 0 : 1] != b->shape[b_transpose ? 1 : 0]) {
//     throw std::invalid_argument("Tensor shapes are not compatible for matrix multiplication.");
//   }
//   std::vector<float> result;
//   int m = a->shape[a_transpose ? 1 : 0];
//   int n = a->shape[a_transpose ? 0 : 1];
//   int p = b->shape[b_transpose ? 0 : 1];

//   int a_stride0 = a->strides[a_transpose ? 1 : 0], a_stride1 = a->strides[a_transpose ? 0 : 1];
//   int b_stride0 = b->strides[b_transpose ? 1 : 0], b_stride1 = b->strides[b_transpose ? 0 : 1];

//   auto a_data = a->data;
//   auto b_data = b->data;
//   result.resize(m * p);

//   // for (int i = 0; i < m; ++i) {
//   //   for (int j = 0; j < p; ++j) {
//   //     float dotProduct = 0;
//   //     for (int k = 0; k < n; ++k) {
//   //       float a_val = a_data[i * a_stride0 + k * a_stride1];
//   //       float b_val = b_data[k * b_stride0 + j * b_stride1];
//   //       dotProduct += a_val * b_val;
//   //     }
//   //     result[i * p + j] = dotProduct;
//   //   }
//   // }

//   // #pragma omp parallel for collapse(2) num_threads(2)
//   for (int i = 0; i < m; ++i) {
//     for (int j = 0; j < p; ++j) {
//       float dotProduct = 0;
//       int k = 0;
//       int ia = i * a_stride0;
//       int jb = j * b_stride1;
//       __m256 sum = _mm256_setzero_ps(); // Initialize sum vector to 0

//       // Use SIMD for the bulk of the operations
//       for (; k <= n - 8; k += 8) {
//         // Load elements one by one due to non-contiguous memory
//         __m256 a_vec = _mm256_set_ps(
//           a_data[ia + (k+7) * a_stride1],
//           a_data[ia + (k+6) * a_stride1],
//           a_data[ia + (k+5) * a_stride1],
//           a_data[ia + (k+4) * a_stride1],
//           a_data[ia + (k+3) * a_stride1],
//           a_data[ia + (k+2) * a_stride1],
//           a_data[ia + (k+1) * a_stride1],
//           a_data[ia + k * a_stride1]
//         );

//         __m256 b_vec = _mm256_set_ps(
//           b_data[(k+7) * b_stride0 + jb],
//           b_data[(k+6) * b_stride0 + jb],
//           b_data[(k+5) * b_stride0 + jb],
//           b_data[(k+4) * b_stride0 + jb],
//           b_data[(k+3) * b_stride0 + jb],
//           b_data[(k+2) * b_stride0 + jb],
//           b_data[(k+1) * b_stride0 + jb],
//           b_data[k * b_stride0 + jb]
//         );


//         // __m256 a_vec = _mm256_loadu_ps(&a_data[i * a_stride0 + k * a_stride1]);
//         // __m256 b_vec = _mm256_loadu_ps(&b_data[k * b_stride0 + j * b_stride1]);
//         __m256 prod = _mm256_mul_ps(a_vec, b_vec);
//         sum = _mm256_add_ps(sum, prod);
//       }

//       // Reduce the sum vector and add to dotProduct
//       float temp[8];
//       _mm256_storeu_ps(temp, sum);
//       for (int x = 0; x < 8; ++x) dotProduct += temp[x];

//       // Handle any remaining elements
//       for (; k < n; ++k) {
//         float a_val = a_data[i * a_stride0 + k * a_stride1];
//         float b_val = b_data[k * b_stride0 + j * b_stride1];
//         dotProduct += a_val * b_val;
//       }

//       result[i * p + j] = dotProduct;
//     }
//   }

//   return array_from_vector(result, {m, p});
// }

std::shared_ptr<Array> multiply_transpose(const std::shared_ptr<Array>& a, bool a_transpose, const std::shared_ptr<Array>& b, bool b_transpose) {
  if (a->shape.size() != 2 || b->shape.size() != 2) {
    throw std::invalid_argument("Matrix multiplication requires two 2D tensors.");
  }
  if (a->shape[a_transpose ? 0 : 1] != b->shape[b_transpose ? 1 : 0]) {
    throw std::invalid_argument("Tensor shapes are not compatible for matrix multiplication.");
  }
  int m = a->shape[a_transpose ? 1 : 0];
  int k = a->shape[a_transpose ? 0 : 1];
  int n = b->shape[b_transpose ? 0 : 1];

  float alpha = 1.0;
  float beta = 0.0;

  // A: m x k matrix
  // B: k x n matrix
  // C: m x n matrix (result)

  // Leading dimensions
  int lda = a_transpose ? m : k;
  int ldb = b_transpose ? k : n;
  int ldc = n;

  // Perform matrix multiplication: C = alpha*A*B + beta*C
  std::vector<float> c(m * n);
  cblas_sgemm(CblasRowMajor, a_transpose ? CblasTrans : CblasNoTrans, b_transpose ? CblasTrans : CblasNoTrans,
              m, n, k, alpha, a->data.data(), lda, b->data.data(), ldb, beta, c.data(), ldc);

  return array_from_vector(c, {m, n});
}

std::shared_ptr<Array> operator%(const std::shared_ptr<Array>& a, const std::shared_ptr<Array>& b) {
  return multiply_transpose(a, false, b, false);
}

std::shared_ptr<Array> squeeze(const std::shared_ptr<Array>& x) {
  auto shape = x->shape;
  std::vector<int> new_shape;
  for (auto& s : shape) {
    if (s != 1) {
      new_shape.push_back(s);
    }
  }
  return array_from_vector(x->data, new_shape);
}

std::shared_ptr<Array> softmax(const std::shared_ptr<Array>& logits, const std::vector<int>& dims) {
  auto counts = exp(logits - max(logits, dims));
  return counts / sum(counts, dims);
}
