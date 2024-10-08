#include "nn.h"
#include "tensor.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <map>

void mlp() {
  // Hyperparameters
  int block_size = 3;
  int embedding_size = 10;
  int hidden_layer_size = 100;
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
  int vocab_size = stoi.size();

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

  auto C = randn({vocab_size, embedding_size}, engine);
  std::vector<std::shared_ptr<Module>> layers = {
    std::make_shared<Linear>(embedding_size * block_size, hidden_layer_size, engine, false),
    std::make_shared<BatchNorm1d>(hidden_layer_size),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(hidden_layer_size, hidden_layer_size, engine, false),
    std::make_shared<BatchNorm1d>(hidden_layer_size),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(hidden_layer_size, hidden_layer_size, engine, false),
    std::make_shared<BatchNorm1d>(hidden_layer_size),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(hidden_layer_size, hidden_layer_size, engine, false),
    std::make_shared<BatchNorm1d>(hidden_layer_size),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(hidden_layer_size, hidden_layer_size, engine, false),
    std::make_shared<BatchNorm1d>(hidden_layer_size),
    std::make_shared<Tanh>(),
    std::make_shared<Linear>(hidden_layer_size, vocab_size, engine, false),
    std::make_shared<BatchNorm1d>(vocab_size),
  };

  auto final = std::dynamic_pointer_cast<BatchNorm1d>(layers[layers.size() - 1]);
  final->gamma->data = final->gamma->data * 0.1f;

  std::vector<std::shared_ptr<Tensor>> parameters;
  parameters.push_back(C);
  for (auto& layer : layers) {
    for (auto& parameter : layer->parameters) {
      parameters.push_back(parameter);
    }
  }
  int num_parameters = std::accumulate(parameters.begin(), parameters.end(), 0, [](int nelement, std::shared_ptr<Tensor> x) {
    return nelement + x->nelement();
  });
  std::cout << "Number of parameters: " << num_parameters << std::endl;

  auto start = std::chrono::high_resolution_clock::now();

  int iterations = 10000;
  for (int k = 0; k < iterations; k += 1) {
    // Minibatch construct
    auto ix = randint(0, Xtr->data->shape[0], {minibatch_size}, engine);
    auto Xb = Xtr->index({ix});
    auto Yb = Ytr->index({ix});

    // Forward pass
    auto emb = C->index({Xb});
    auto x = emb->view({emb->data->shape[0], -1});
    for (auto& layer : layers) {
      x = (*layer)(x);
    }
    auto loss = cross_entropy(x, Yb);
    if (k % 100 == 0) {
      std::cerr << k << ": " << loss->data->data[0] << " " << (static_cast<float>(k) / iterations * 100.0f) << "%" << std::endl;
    }

    // Backward pass
    for (auto& p : parameters) {
      p->grad = {};
    }
    loss->backward();

    // Update
    for (auto& p : parameters) {
      // p->data = p->data - (k < 5000 ? 0.1f : 0.01f) * p->grad;
      p->data = p->data - 0.1f * p->grad;
    }
  }

  std::chrono::duration<double> duration = std::chrono::high_resolution_clock::now() - start;
  std::cout << "Loop execution time: " << duration.count() << " seconds" << std::endl;

  {
    auto emb = C->index({Xtr});
    auto x = emb->view({emb->data->shape[0], -1});
    for (auto& layer : layers) {
      x = (*layer)(x);
    }
    auto loss = cross_entropy(x, Ytr);
    std::cerr << "train loss: " << loss->data->data[0] << std::endl;
  }

  {
    auto emb = C->index({Xdev});
    auto x = emb->view({emb->data->shape[0], -1});
    for (auto& layer : layers) {
      x = (*layer)(x);
    }
    auto loss = cross_entropy(x, Ydev);
    std::cerr << "dev loss: " << loss->data->data[0] << std::endl;
  }

  for (auto& layer : layers) {
    layer->training = false;
  }

  for (int i = 0; i < 20; i += 1) {
    std::string out;
    auto context = std::vector<float>(block_size);
    while (true) {
      auto emb = C->index({from_vector(context, {block_size})});
      auto x = emb->view({1, -1});
      for (auto& layer : layers) {
        x = (*layer)(x);
      }
      auto logits = x;
      auto probs = softmax(logits, {1});
      auto pred = multinomial(probs, engine);
      auto next = pred->data->data[0];
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

void test_index_backprop() {
  auto engine = std::default_random_engine(std::random_device{}());
  auto x = randn({10, 10}, engine);
  auto y = x->index({from_vector({0, 5, 1, 5}, {2, 2}), from_vector({0, 5, 1, 5}, {2, 2})});
  y->print();
  auto z = sum(y);
  z->backward();
  x->grad->print();
}

void test_view_backprop() {
  auto engine = std::default_random_engine(std::random_device{}());
  auto x = randn({10, 10}, engine);
  auto y = x->view({5, 20});
  y->print();
  auto z = sum(y * y);
  z->backward();
  y->grad->print();
  x->grad->print();
}

int main() {
  // test_view_backprop();
  mlp();
  return 0;
}
