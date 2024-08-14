#include <gtest/gtest.h>
#include <vector>

#include "nn.h"

TEST(NNTest, BatchNorm1d) {
  auto bn = BatchNorm1d(2);
  auto bn_input = from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  auto bn_output = bn(bn_input);
  auto s = sum(bn_output);
  s->backward();

  float val = 0.9999988079071045f;
  EXPECT_EQ(bn_output->data->data, std::vector<float>({-val, -val, 0.0f, 0.0f, val, val}));
  EXPECT_EQ(bn.beta->grad->data, std::vector<float>({3.0f, 3.0f}));
  EXPECT_EQ(bn.gamma->grad->data, std::vector<float>({0.0f, 0.0f}));
}

TEST(NNTest, BatchNorm2dInternals) {
  auto gamma = ones({2});
  auto beta = zeros({2});
  auto x = from_vector({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, {3, 2});
  auto x_mean = mean(x, {0});
  auto x_var = variance(x, {0});
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
