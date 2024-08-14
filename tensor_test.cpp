#include <gtest/gtest.h>
#include <vector>

#include "tensor.h"

TEST(TensorTest, VarianceAndSumBackward) {
  auto tt = from_vector({0, 1, 2, 4}, {2, 2});
  auto vv = variance(tt, {0});
  auto ss = sum(vv);
  ss->backward();

  EXPECT_EQ(vv->data->data, std::vector<float>({2.0f, 4.5f}));
  EXPECT_EQ(ss->data->data, std::vector<float>({6.5f}));
  EXPECT_EQ(vv->grad->data, std::vector<float>({1.0f, 1.0f}));
  EXPECT_EQ(tt->grad->data, std::vector<float>({-2.0f, -3.0f, 2.0f, 3.0f}));
}
