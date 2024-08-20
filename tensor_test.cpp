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

TEST(TensorTest, CrossEntropyFast) {
  std::vector<float> x_data = {0.1f, -1, 0.2f, -1, 0.3f, -1, 0.4f, -1, 0.5f, -1, 0.6f, -1};
  std::shared_ptr<Tensor> loss1, loss2;
  auto orig = from_vector(x_data, {2, 3, 2});

  auto x1 = orig->slice({Slice{0, -1}, Slice{0, -1}, Slice{0}});
  {
    auto y = from_vector({0, 1}, {2});
    loss1 = cross_entropy_unoptimized(x1, y);
    loss1->print();
    loss1->backward();
    x1->grad->print();
  }

  // Create a test case with a final stride of 2 to test the fast version indexing
  auto x2 = orig->slice({Slice{0, -1}, Slice{0, -1}, Slice{0}});
  {
    auto y = from_vector({0, 1}, {2});
    loss2 = cross_entropy(x2, y);
    loss2->print();
    loss2->backward();
    x2->grad->print();
  }

  EXPECT_EQ(loss1->data->shape, loss2->data->shape);
  for (int i = 0; i < loss1->data->nelement(); i++) {
    EXPECT_FLOAT_EQ(loss1->data->data[i], loss2->data->data[i]);
  }
  EXPECT_EQ(x1->grad->shape, x2->grad->shape);
  for (int i = 0; i < x1->grad->nelement(); i++) {
    EXPECT_FLOAT_EQ(x1->grad->data[i], x2->grad->data[i]);
  }
}
