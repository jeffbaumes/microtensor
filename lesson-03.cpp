#include "tensor.h"

#include <fstream>
#include <iostream>
#include <map>
#include <random>

void test_index() {
  auto t = from_vector({1, 2, 3, 4, 5, 6}, {2, 3});
  auto x = from_vector({0, 1, 0}, {3});
  auto y = from_vector({0, 0, 2}, {3});
  auto s = t->index({x});
  s->print();
  assert(s->data->data == std::vector<float>({1, 2, 3, 4, 5, 6, 1, 2, 3}));
  assert(s->data->shape == std::vector<int>({3, 3}));
  auto s2 = t->index({x, y});
  s2->print();
  assert(s2->data->data == std::vector<float>({1, 4, 3}));
  assert(s2->data->shape == std::vector<int>({3}));
}

void mlp() {
  // Hyperparameters
  int block_size = 3;
  int embedding_size = 10;
  int hidden_layer_size = 200;
  int minibatch_size = 32;

  // Read names.txt into a vector of strings
  std::vector<std::string> names;
  std::ifstream file("names.txt");
  std::string name;
  int num_names = 0;
  while (std::getline(file, name)) {
    names.push_back(name);
    num_names += 1;
    // if (num_names == 5) {
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

  auto build_dataset = [&stoi, &block_size](const std::vector<std::string>& names) {
    std::vector<float> xs_vec;
    std::vector<float> ys_vec;
    for (auto& name : names) {
      std::string chs = "";
      for (int i = 0; i < block_size; i += 1) {
        chs += '.';
      }
      chs += name + '.';
      for (int i = 0; i < chs.length() - block_size; i += 1) {
        for (int j = 0; j < block_size; j += 1) {
          xs_vec.push_back(stoi[chs[i + j]]);
        }
        ys_vec.push_back(stoi[chs[i + block_size]]);
      }
    }
    std::cout << "Number of examples: " << xs_vec.size() << std::endl;
    auto X = from_vector(xs_vec, {static_cast<int>(ys_vec.size()), block_size});
    auto Y = from_vector(ys_vec, {static_cast<int>(ys_vec.size())});
    return std::make_pair(X, Y);
  };

  auto engine = std::default_random_engine(std::random_device{}());

  std::shuffle(names.begin(), names.end(), engine);
  int n1 = static_cast<int>(names.size() * 0.8);
  int n2 = static_cast<int>(names.size() * 0.9);
  auto train = std::vector<std::string>(names.begin(), names.begin() + n1);
  auto dev = std::vector<std::string>(names.begin() + n1, names.begin() + n2);
  auto test = std::vector<std::string>(names.begin() + n2, names.end());
  auto [Xtr, Ytr] = build_dataset(train);
  auto [Xdev, Ydev] = build_dataset(dev);
  auto [Xte, Yte] = build_dataset(test);

  auto C = randn({27, embedding_size}, engine);
  auto W1 = randn({block_size * embedding_size, hidden_layer_size}, engine);
  auto b1 = randn({hidden_layer_size}, engine);
  auto W2 = randn({hidden_layer_size, 27}, engine);
  auto b2 = randn({27}, engine);
  auto parameters = std::vector<std::shared_ptr<Tensor>>{C, W1, b1, W2, b2};

  int num_params = std::accumulate(parameters.begin(), parameters.end(), 0, [](auto last, auto p) { return last + p->nelement(); });
  std::cout << num_params << std::endl;

  // int iterations = 100000;
  int iterations = 100000;

  for (int k = 0; k < iterations; k += 1) {
    // Minibatch construct
    auto ix = randint(0, Xtr->data->shape[0], {minibatch_size}, engine);

    // Forward pass
    auto emb = C->index({Xtr->index({ix})});
    auto h = tanh(emb->view({minibatch_size, block_size * embedding_size}) % W1 + b1);
    auto logits = h % W2 + b2;
    auto loss = cross_entropy(logits, Ytr->index({ix}));
    if (k % 1000 == 0) {
      std::cerr << k << ": " << loss->data->data[0] << " " << (static_cast<float>(k) / iterations * 100.0f) << "%" << std::endl;
    }

    // Backward pass
    for (auto& p : parameters) {
      p->grad = {};
    }
    loss->backward();

    // Update
    for (auto& p : parameters) {
      p->data = p->data - 0.1f * p->grad;
    }
  }

  {
    auto emb = C->index({Xtr});
    auto h = tanh(emb->view({Ytr->data->shape[0], block_size * embedding_size}) % W1 + b1);
    auto logits = h % W2 + b2;
    auto loss = cross_entropy(logits, Ytr);
    std::cerr << "train loss: " << loss->data->data[0] << std::endl;
  }

  {
    auto emb = C->index({Xdev});
    auto h = tanh(emb->view({Ydev->data->shape[0], block_size * embedding_size}) % W1 + b1);
    auto logits = h % W2 + b2;
    auto loss = cross_entropy(logits, Ydev);
    std::cerr << "dev loss: " << loss->data->data[0] << std::endl;
  }

  for (int i = 0; i < 50; i += 1) {
    std::string out;
    std::vector<float> context(block_size);
    while (true) {
      auto emb = C->index({from_vector(context, {block_size})});
      auto h = tanh(emb->view({1, block_size * embedding_size}) % W1 + b1);
      auto logits = h % W2 + b2;
      auto probs = softmax(logits, {1});
      auto pred = multinomial(probs, engine);
      float next = pred->data->data[0];
      if (next == 0) {
        break;
      }
      context = std::vector<float>(context.begin() + 1, context.end());
      context.push_back(next);
      out += itos[next];
    }
    std::cout << out << std::endl;
  }
}

int main() {
  test_index();
  mlp();
  return 0;
}
