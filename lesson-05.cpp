#include "tensor.h"
#include "nn.h"

#include <chrono>
#include <iostream>
#include <random>

int main() {
  auto engine = std::default_random_engine(std::random_device{}());

  int iterations = 100;
  auto large = randn({1000, 100}, engine);
  auto target = randint(0, 100, {1000}, engine);

  // Implemented optimized backpropagation for cross_entropy and BatchNorm1d.
  // cross_entropy backprop has ~85x performance boost
  // BatchNorm2d backprop has ~15x performance boost

  {
    auto loss = cross_entropy_unoptimized(large, target);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      cross_entropy_unoptimized(large, target);
    }
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "cross_entropy_unoptimized forward pass: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      loss->backward();
    }
    duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "cross_entropy_unoptimized backward pass: " << duration.count() << " seconds" << std::endl;
  }

  {
    auto loss = cross_entropy(large, target);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      cross_entropy(large, target);
    }
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "cross_entropy forward pass: " << duration.count() << " seconds" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      loss->backward();
    }
    duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "cross_entropy backward pass: " << duration.count() << " seconds" << std::endl;
  }

  {
    BatchNorm1dUnoptimized bn(100);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      bn(large);
    }
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "BatchNorm1dUnoptimized forward pass: " << duration.count() << " seconds" << std::endl;

    auto loss = sum(bn(large));
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      loss->backward();
    }
    duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "BatchNorm1dUnoptimized backward pass: " << duration.count() << " seconds" << std::endl;
  }

  {
    BatchNorm1d bn(100);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      bn(large);
    }
    std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "BatchNorm1d forward pass: " << duration.count() << " seconds" << std::endl;

    auto loss = sum(bn(large));
    start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; i++) {
      loss->backward();
    }
    duration = std::chrono::high_resolution_clock::now() - start;
    std::cout << "BatchNorm1d backward pass: " << duration.count() << " seconds" << std::endl;
  }
}
