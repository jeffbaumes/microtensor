#include "tensor.h"

#include <iostream>
#include <fstream>
#include <map>
#include <vector>

int main() {
  {
    auto t = from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
    t->print();
    auto s0 = sum(t, {0});
    s0->print();
    assert(s0->data->data == std::vector<float>({5, 7, 9}));
    auto s1 = sum(t, {1});
    s1->print();
    assert(s1->data->data == std::vector<float>({6, 15}));
    auto s12 = sum(t, {1, 0});
    s12->print();
    assert(s12->data->data[0] == 21);
    auto s = sum(t);
    s->print();
    assert(s->data->data[0] == 21);
  }

  {
    auto t = from_vector({1, 0, 1, 1}, {4});
    t->print();
    auto h = one_hot(t);
    h->print();
    assert(h->data->data == std::vector<float>({1, 0, 0, 1, 1, 0, 1, 0}));
    auto h2 = one_hot(t, 3);
    h2->print();
    assert(h2->data->data == std::vector<float>({1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0}));
  }

  // Read names.txt into a vector of strings
  std::vector<std::string> names;
  std::ifstream file("names.txt");
  std::string name;
  int num = 0;
  while (std::getline(file, name)) {
    names.push_back(name);
    num += 1;
    if (num == 10) {
      break;
    }
  }

  std::vector<float> xs_vec;
  std::vector<float> ys_vec;
  for (auto& name : names) {
    auto chs = '.' + name + '.';
    for (int i = 1; i < chs.length(); i += 1) {
      xs_vec.push_back(chs[i - 1]);
      ys_vec.push_back(chs[i]);
    }
  }
  std::cout << "Number of examples: " << xs_vec.size() << std::endl;

  // Print first 10 characters in xs and ys
  for (int i = 0; i < 10; i += 1) {
    std::cout << (char)xs_vec[i] << " -> " << (char)ys_vec[i] << std::endl;
  }

  // Convert xs and ys to tensors
  auto xs = from_vector(xs_vec, {(int)xs_vec.size()});
  auto ys = from_vector(ys_vec, {(int)ys_vec.size()});

  auto xenc = one_hot(xs, 27);

  // Print first 10 one-hot encoded values in xenc
  for (int i = 0; i < 10; i += 1) {
    for (int j = 0; j < 27; j += 1) {
      std::cout << xenc->data->data[27 * i + j] << " ";
    }
    std::cout << std::endl;
  }

  std::default_random_engine engine(std::random_device{}());
  auto W = randn({27, 27}, engine);

  for (int k = 0; k < 1; k += 1) {
    // Forward pass
    auto logits = xenc % W;
    auto counts = expf(logits);
    auto probs = counts / sum(counts, {1});
    // auto loss = mean(logf(-probs[arange(num), ys])) + 0.01*mean(powf(W, 2.0f));
    // std::cout << loss->data->data[0] << std::endl;
    // probs->print();
  }

  return 0;
}
