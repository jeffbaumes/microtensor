#include <gtest/gtest.h>
#include <vector>

#include "array.h"

TEST(ArrayTest, MatrixMultiply) {
  {
    auto a = array_from_vector({0, 1, 2, 3}, {1, 2, 2});
    auto b = array_from_vector({0, 1, 2, 3}, {2, 2, 1});
    auto c = a % b;

    EXPECT_EQ(c->shape, std::vector<int>({1, 2, 2, 1}));
    EXPECT_EQ(c->data, std::vector<float>({2, 3, 6, 11}));
  }

  {
    auto a = array_from_vector({0, 2, 1, 3}, {2, 2, 1});
    auto b = array_from_vector({0, 2, 1, 3}, {1, 2, 2});
    auto c = multiply_transpose(a, true, b, true);

    EXPECT_EQ(c->shape, std::vector<int>({1, 2, 2, 1}));
    EXPECT_EQ(c->data, std::vector<float>({2, 3, 6, 11}));
  }
}
