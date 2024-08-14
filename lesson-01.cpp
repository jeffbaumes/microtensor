#include "nn.h"
#include "tensor.h"

int main() {
  {
    auto m = from_vector({0, 1, 2, 2, 1, 0}, {1, 2, 3, 1});
    m->print();
  }

  {
    auto m1 = from_vector({0, 1, 2, 2, 1, 0}, {2, 3});
    auto m2 = from_vector({1, 1, 1, 2, 1, 3}, {3, 2});
    m2 = m2->slice({{0, -1}, {0, 1}});
    auto result = m1 % m2;
    std::cout << "m1:" << std::endl;
    m1->print();
    std::cout << "m2:" << std::endl;
    m2->print();
    std::cout << "result:" << std::endl;
    result->print();
    assert(result->data->data == std::vector<float>({3, 3}));
    assert(result->data->shape == std::vector<int>({2, 1}));
  }

  {
    auto tensor = from_vector({1, 2, 3, 4, 5, 6}, {3, 2});

    std::cout << "Original Tensor:" << std::endl;
    tensor->print();

    auto subTensor = tensor->slice({{1, -1}, {0, 1}});
    std::cout << "Sub-Tensor:" << std::endl;
    subTensor->print();
  }

  {
    auto t1 = from_vector({1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6}, {2, 3, 2});
    auto t2 = from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    std::cout << "t1:" << std::endl;
    t1->print();
    t1 = (*t1)[0];
    std::cout << "t1:" << std::endl;
    t1->print();
    auto sum = t1 + t2;
    std::cout << "sum:" << std::endl;
    sum->print();
    auto prod = t1 * t2;
    std::cout << "prod:" << std::endl;
    prod->print();
  }

  {
    auto t1 = from_vector({1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6}, {3, 2, 2});
    auto t2 = from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    std::cout << "t1:" << std::endl;
    t1->print();
    t1 = t1->slice({{0, -1}, {0, -1}, {0}});
    std::cout << "t1:" << std::endl;
    t1->print();
    std::cout << "t2:" << std::endl;
    t2->print();
    auto sum = t1 + t2;
    std::cout << "sum:" << std::endl;
    sum->print();
    assert(sum->data->data == std::vector<float>({2, 4, 6, 8, 10, 12}));
    auto prod = t1 * t2;
    std::cout << "prod:" << std::endl;
    prod->print();
    assert(prod->data->data == std::vector<float>({1, 4, 9, 16, 25, 36}));
  }

  {
    auto x = from_vector({1, 2, 3, 4, 5, 6}, {3, 2});
    std::cout << "x:" << std::endl;
    x->print();
    auto W = from_vector({1, 2, 3, 4}, {2, 2});
    std::cout << "W:" << std::endl;
    W->print();
    auto b = from_vector({1, 2}, {1, 2});
    std::cout << "b:" << std::endl;
    b->print();
    auto prod = x % W;
    std::cout << "x % W:" << std::endl;
    prod->print();
    auto out = prod + b;
    std::cout << "x % W + b:" << std::endl;
    out->print();
    out->backward();
    std::cout << "x->grad:" << std::endl;
    x->grad->print();
    std::cout << "W->grad:" << std::endl;
    W->grad->print();
    std::cout << "b->grad:" << std::endl;
    b->grad->print();
    std::cout << "prod->grad:" << std::endl;
    prod->grad->print();
    std::cout << "out->grad:" << std::endl;
    out->grad->print();
  }

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
      auto err = pow(ypred - ys, 2.0f);
      auto loss = sum(err);

      // Backward pass
      for (auto& layer : n.layers) {
        for (auto& parameter : layer->parameters) {
          if (parameter->grad) {
            parameter->grad = nullptr;
          }
        }
      }
      loss->backward();

      // Update
      for (auto& layer : n.layers) {
        for (auto& parameter : layer->parameters) {
          parameter->data = parameter->data - 0.02f * parameter->grad;
        }
      }

      std::cout << k << " " << loss->data->data[0] << std::endl;
    }

    ypred->print();
  }

  return 0;
}
