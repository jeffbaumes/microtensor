#include <gtest/gtest.h>
#include <vector>

#include "nn.h"

TEST(NNTest, BatchNorm1d) {
/* pytorch equivalent

import torch
import torch.nn as nn
torch.set_printoptions(precision=10)
x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float32)
x.requires_grad = True
bn = nn.BatchNorm1d(2, dtype=torch.float32)
bn.weight.retain_grad()
bn.bias.retain_grad()
result = bn(x)
print(result)
loss = result.sum()
loss.backward()
print(bn.weight.grad)
print(bn.bias.grad)
*/

  auto bn = BatchNorm1d(2);
  auto bn_input = from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  auto bn_output = bn(bn_input);
  auto s = sum(bn_output);
  s->backward();

  float val = 1.2247426510f;
  EXPECT_EQ(bn_output->data->data, std::vector<float>({-val, -val, 0.0f, 0.0f, val, val}));
  EXPECT_EQ(bn.beta->grad->data, std::vector<float>({3.0f, 3.0f}));
  EXPECT_EQ(bn.gamma->grad->data, std::vector<float>({0.0f, 0.0f}));
}

TEST(NNTest, BatchNorm1dInternals) {
  auto gamma = ones({2});
  auto beta = zeros({2});
  auto x = from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  auto x_mean = mean(x, {0});
  auto x_var = variance_biased(x, {0});
  auto x_hat = (x - x_mean) / sqrt(x_var + 1.0e-5f);
  auto out = gamma * x_hat + beta;
  auto s = sum(out);
  s->backward();
  gamma->print();
  x_hat->print();
  beta->print();
  gamma->grad->print();
  x_hat->grad->print();
  beta->grad->print();
  EXPECT_EQ(beta->grad->data, std::vector<float>({3.0f, 3.0f}));
  EXPECT_EQ(gamma->grad->data, std::vector<float>({0.0f, 0.0f}));
}

TEST(NNTest, Addition) {
  auto beta = ones({2});
  auto x_hat = ones({3, 2});
  auto out = x_hat + beta;
  auto s = sum(out);
  s->print();
  s->backward();
  out->grad->print();
  x_hat->grad->print();
  beta->grad->print();
  EXPECT_EQ(beta->grad->data, std::vector<float>({3.0f, 3.0f}));
}

TEST(NNTest, BatchNorm1dOptimized) {
  std::vector<float> x_data = {1.0, 2.0, 3.0, 4.0, -5.0, 6.0, 20.0, 10.0, -30.0, -3.0, 2.0, 31.0};

  auto x1 = from_vector(x_data, {4, 3});
  auto bn1 = std::make_shared<BatchNorm1dUnoptimized>(3);
  auto result1 = (*bn1)(x1);
  auto loss1 = sum(tanh(result1));
  loss1->backward();

  auto x2 = from_vector(x_data, {4, 3});
  auto bn2 = std::make_shared<BatchNorm1d>(3);
  auto result2 = (*bn2)(x2);
  auto loss2 = sum(tanh(result2));
  loss2->backward();

/* pytorch equivalent

import torch
import torch.nn as nn
torch.set_printoptions(precision=10)
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, -5.0, 6.0], [20.0, 10.0, -30.0], [-3.0, 2.0, 31.0]], dtype=torch.float32)
x.requires_grad = True
bn = nn.BatchNorm1d(3, dtype=torch.float32)
result = bn(x)
t = result.tanh()
loss = t.sum()
print(loss)
loss.backward()
print(x.grad)
*/

  EXPECT_EQ(loss1->data->shape, loss2->data->shape);
  for (int i = 0; i < loss1->data->nelement(); i++) {
    EXPECT_FLOAT_EQ(loss1->data->data[i], loss2->data->data[i]);
  }
  EXPECT_FLOAT_EQ(loss1->data->data[0], -0.3947991729f);
  EXPECT_FLOAT_EQ(loss2->data->data[0], -0.3947991729f);

  EXPECT_EQ(x1->grad->shape, x2->grad->shape);
  for (int i = 0; i < x1->grad->nelement(); i++) {
    EXPECT_FLOAT_EQ(x1->grad->data[i], x2->grad->data[i]);
  }
  EXPECT_FLOAT_EQ(x1->grad->data[0], 0.0110761188f);
  EXPECT_FLOAT_EQ(x2->grad->data[0], 0.0110761188f);
}
