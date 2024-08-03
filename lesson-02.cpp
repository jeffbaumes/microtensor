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
    s0->backward();
    t->grad->print();
    assert(t->grad->data == std::vector<float>({1, 1, 1, 1, 1, 1}));
    auto s1 = sum(t, {1});
    s1->print();
    assert(s1->data->data == std::vector<float>({6, 15}));
    auto s12 = sum(t, {1, 0});
    s12->print();
    assert(s12->data->data[0] == 21);
    auto s = sum(t);
    s->print();
    assert(s->data->data[0] == 21);
    t->grad = {};
    s->backward();
    t->grad->print();
    assert(t->grad->data == std::vector<float>({1, 1, 1, 1, 1, 1}));
  }

  {
    auto t = from_vector({1, 0, 1, 1}, {4});
    t->print();
    auto h = one_hot(t);
    h->print();
    assert(h->data->data == std::vector<float>({0, 1, 1, 0, 0, 1, 0, 1}));
    auto h2 = one_hot(t, 3);
    h2->print();
    assert(h2->data->data == std::vector<float>({0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0}));
  }

  {
    auto t = from_vector({1, 2, 3, 4}, {4});
    auto x = exp(log(t));
    assert(x->data->data == std::vector<float>({1, 2, 3, 4}));
    x->backward();
    assert(t->grad->data == std::vector<float>({1, 1, 1, 1}));
  }

  {
    std::cout << "===test multiply" << std::endl;
    auto t = from_vector({1, 2, 3, 4}, {2, 2});
    auto u = from_vector({2, 3}, {2, 1});
    auto m = t * u;
    auto s = sum(m);
    s->print();
    s->backward();
    t->data->print();
    t->grad->print();
    u->data->print();
    u->grad->print();
    assert(u->grad->data == std::vector<float>({3, 7}));
  }

  {
    std::cout << "===test mean" << std::endl;
    auto t = from_vector({1, 2, 3, 4}, {2, 2});
    auto m = mean(t);
    assert(m->data->data == std::vector<float>({2.5f}));
    m->backward();
    t->grad->print();
    assert(t->grad->data == std::vector<float>({0.25f, 0.25f, 0.25f, 0.25f}));
    auto m0 = mean(t, {0});
    assert(m0->data->data == std::vector<float>({2, 3}));
    auto s = sum(m0);
    t->grad = {};
    s->backward();
    s->grad->print();
    m0->grad->print();
    t->grad->print();
    assert(t->grad->data == std::vector<float>({0.5f, 0.5f, 0.5f, 0.5f}));
  }

  {
    auto t = arange(0, 4);
    t->print();
    assert(t->data->data == std::vector<float>({0, 1, 2, 3}));
    t = arange(0, 4, 3);
    t->print();
    assert(t->data->data == std::vector<float>({0, 3}));
    t = arange(0, 4, 2);
    t->print();
    assert(t->data->data == std::vector<float>({0, 2}));
  }

  // Read names.txt into a vector of strings
  std::vector<std::string> names;
  std::ifstream file("names.txt");
  std::string name;
  int num_names = 0;
  while (std::getline(file, name)) {
    names.push_back(name);
    num_names += 1;
    // if (num_names == 500) {
    //   break;
    // }
  }

  std::map<char, int> stoi;
  std::map<int, char> itos;
  for (char c = 'a'; c <= 'z'; c += 1) {
    stoi[c] = c - 'a' + 1;
    itos[stoi[c]] = c;
  }
  stoi['.'] = 0;
  itos[0] = '.';

  std::vector<float> xs_vec;
  std::vector<float> ys_vec;
  for (auto& name : names) {
    auto chs = '.' + name + '.';
    for (int i = 1; i < chs.length(); i += 1) {
      int ix = stoi[chs[i - 1]];
      int iy = stoi[chs[i]];
      xs_vec.push_back(ix);
      ys_vec.push_back(iy);
    }
  }
  std::cout << "Number of examples: " << xs_vec.size() << std::endl;

  // Print first 10 characters in xs and ys
  for (int i = 0; i < 10; i += 1) {
    std::cout << static_cast<char>(itos[xs_vec[i]]) << " -> " << static_cast<char>(itos[ys_vec[i]]) << std::endl;
  }

  int num = xs_vec.size();

  // Convert xs and ys to tensors
  auto xs = from_vector(xs_vec, {num});
  auto ys = from_vector(ys_vec, {num});

  std::default_random_engine engine(std::random_device{}());
  auto W = randn({27, 27}, engine);

  for (int k = 0; k < 100; k += 1) {
    // Forward pass
    auto xenc = one_hot(xs, 27);
    auto logits = xenc % W;
    auto counts = exp(logits);
    auto counts_sum = sum(counts, {1});
    auto probs = counts / counts_sum;
    auto l1 = -mean(log(probs->index({arange(0, num), ys})));
    auto l2 = 0.01f*mean(pow(W, 2.0f));
    auto loss = l1 + l2;
    std::cout << "===loss " << loss->data->data[0] << std::endl;

    // Backward pass
    W->grad = {};
    loss->backward();

    // Update
    W->data = W->data - 50.0f * W->grad;
  }

  return 0;
}
