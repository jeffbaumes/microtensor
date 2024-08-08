#include "nn.h"
#include "tensor.h"

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

  {
    NoGrad _;
    auto final = std::dynamic_pointer_cast<BatchNorm1d>(layers[layers.size() - 1]);
    final->gamma->data = final->gamma->data * 0.1f;
    // for (auto& layer : layers) {
    //   auto linear = std::dynamic_pointer_cast<Linear>(layer);
    //   if (linear) {
    //     linear->W->data = linear->W->data * 1.0f;
    //   }
    // }
  }

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
      // if (std::dynamic_pointer_cast<Linear>(layer)) {
      //   std::cout << "Linear" << std::endl;
      // } else if (std::dynamic_pointer_cast<BatchNorm1d>(layer)) {
      //   std::cout << "BatchNorm1d" << std::endl;
      // } else if (std::dynamic_pointer_cast<Tanh>(layer)) {
      //   std::cout << "Tanh" << std::endl;
      // }
      x = (*layer)(x);
      // std::cout << "x = ";
      // x->view({-1})->print();
      // std::cout << std::endl;
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
      p->data = p->data - 0.1f * p->grad;
    }
  }

  for (int i = 0; i < layers.size(); i += 1) {
    std::cout << i << ": " << mean(layers[i]->out->data)->data[0] << "," << sqrt(variance(layers[i]->out->data)->data[0]) << std::endl;
  }
  for (int i = 0; i < layers.size(); i += 1) {
    std::cout << i << ": " << mean(layers[i]->out->grad)->data[0] << "," << sqrt(variance(layers[i]->out->grad)->data[0]) << std::endl;
  }


  // {
  //   NoGrad _;
  //   std::cout << "data = [";
  //   for (int i = 0; i < layers.size(); i += 1) {
  //     // std::cout << mean(tanh->out->data)->data[0] << "," << sqrt(variance(tanh->out->data)->data[0]) << std::endl;
  //     if (i > 0) {
  //       std::cout << ",";
  //     }
  //     layers[i]->out->data->view({-1})->print();
  //   }
  //   std::cout << "]" << std::endl;

  //   std::cout << "grad = [";
  //   for (int i = 0; i < layers.size(); i += 1) {
  //     // std::cout << mean(tanh->out->grad)->data[0] << "," << sqrt(variance(tanh->out->grad)->data[0]) << std::endl;
  //     if (i > 0) {
  //       std::cout << ",";
  //     }
  //     layers[i]->out->grad->view({-1})->print();
  //   }
  //   std::cout << "]" << std::endl;
  // }
}

int main() {
  mlp();
  return 0;
}
