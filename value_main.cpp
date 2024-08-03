#include "value.h"

int main() {
  {
    auto a = std::make_shared<Value>(1.0f);
    auto b = std::make_shared<Value>(2.0f);
    auto c = std::make_shared<Value>(3.0f);
    auto d = std::make_shared<Value>(4.0f);

    auto e = a + b;
    auto f = c * d;
    auto g = tanh(e + f);

    g->backward();

    std::cout << "a|data=" << a->data << "|grad=" << a->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
    std::cout << "c|data=" << c->data << "|grad=" << c->grad << std::endl;
    std::cout << "d|data=" << d->data << "|grad=" << d->grad << std::endl;
    std::cout << "e|data=" << e->data << "|grad=" << e->grad << std::endl;
    std::cout << "f|data=" << f->data << "|grad=" << f->grad << std::endl;
    std::cout << "g|data=" << g->data << "|grad=" << g->grad << std::endl;
  }

  {
    auto x1 = std::make_shared<Value>(2.0f);
    auto x2 = std::make_shared<Value>(0.0f);
    auto w1 = std::make_shared<Value>(-3.0f);
    auto w2 = std::make_shared<Value>(1.0f);
    auto b = std::make_shared<Value>(6.881373587f);
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = x1w1 + x2w2;
    auto n = x1w1x2w2 + b;
    auto o = tanh(n);

    o->backward();

    std::cout << "x1|data=" << x1->data << "|grad=" << x1->grad << std::endl;
    std::cout << "x2|data=" << x2->data << "|grad=" << x2->grad << std::endl;
    std::cout << "w1|data=" << w1->data << "|grad=" << w1->grad << std::endl;
    std::cout << "w2|data=" << w2->data << "|grad=" << w2->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
    std::cout << "x1w1|data=" << x1w1->data << "|grad=" << x1w1->grad << std::endl;
    std::cout << "x2w2|data=" << x2w2->data << "|grad=" << x2w2->grad << std::endl;
    std::cout << "x1w1x2w2|data=" << x1w1x2w2->data << "|grad=" << x1w1x2w2->grad << std::endl;
    std::cout << "n|data=" << n->data << "|grad=" << n->grad << std::endl;
    std::cout << "o|data=" << o->data << "|grad=" << o->grad << std::endl;
  }

  {
    auto x1 = std::make_shared<Value>(2.0f);
    auto x2 = std::make_shared<Value>(0.0f);
    auto w1 = std::make_shared<Value>(-3.0f);
    auto w2 = std::make_shared<Value>(1.0f);
    auto b = std::make_shared<Value>(6.881373587f);
    auto x1w1 = x1 * w1;
    auto x2w2 = x2 * w2;
    auto x1w1x2w2 = x1w1 + x2w2;
    auto n = x1w1x2w2 + b;
    auto o = (exp(2*n) - 1) / (exp(2*n) + 1);

    o->backward();

    std::cout << "x1|data=" << x1->data << "|grad=" << x1->grad << std::endl;
    std::cout << "x2|data=" << x2->data << "|grad=" << x2->grad << std::endl;
    std::cout << "w1|data=" << w1->data << "|grad=" << w1->grad << std::endl;
    std::cout << "w2|data=" << w2->data << "|grad=" << w2->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
    std::cout << "x1w1|data=" << x1w1->data << "|grad=" << x1w1->grad << std::endl;
    std::cout << "x2w2|data=" << x2w2->data << "|grad=" << x2w2->grad << std::endl;
    std::cout << "x1w1x2w2|data=" << x1w1x2w2->data << "|grad=" << x1w1x2w2->grad << std::endl;
    std::cout << "n|data=" << n->data << "|grad=" << n->grad << std::endl;
    std::cout << "o|data=" << o->data << "|grad=" << o->grad << std::endl;
  }

  {
    auto a = std::make_shared<Value>(3.0f);
    auto b = a + a;
    b->backward();
    std::cout << "a|data=" << a->data << "|grad=" << a->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
  }

  {
    auto a = std::make_shared<Value>(3.0f);
    auto b = a + 5;
    b->backward();
    std::cout << "a|data=" << a->data << "|grad=" << a->grad << std::endl;
    std::cout << "b|data=" << b->data << "|grad=" << b->grad << std::endl;
    b->print();
  }

  {
    auto x = std::vector<std::shared_ptr<Value>> {
      std::make_shared<Value>(2.0f),
      std::make_shared<Value>(2.0f),
      std::make_shared<Value>(-1.0f),
    };
    auto n = MLP(3, {4, 4, 1});
    auto out = n(x);
    std::cout << n(x)[0]->data << std::endl;
  }

  {
    auto n = MLP(3, {4, 4, 1});
    auto xs = std::vector<std::vector<std::shared_ptr<Value>>> {
      {
        std::make_shared<Value>(2.0f),
        std::make_shared<Value>(3.0f),
        std::make_shared<Value>(-1.0f),
      },
      {
        std::make_shared<Value>(3.0f),
        std::make_shared<Value>(-1.0f),
        std::make_shared<Value>(0.5f),
      },
      {
        std::make_shared<Value>(5.0f),
        std::make_shared<Value>(1.0f),
        std::make_shared<Value>(1.0f),
      },
      {
        std::make_shared<Value>(1.0f),
        std::make_shared<Value>(1.0f),
        std::make_shared<Value>(-1.0f),
      },
    };
    auto ys = std::vector<std::shared_ptr<Value>> {
      std::make_shared<Value>(1.0f),
      std::make_shared<Value>(-1.0f),
      std::make_shared<Value>(-1.0f),
      std::make_shared<Value>(1.0f),
    };

    auto ypred = std::vector<std::shared_ptr<Value>>(4);
    auto err = std::vector<std::shared_ptr<Value>>(4);
    auto loss = std::make_shared<Value>(0.0f);
    std::vector<std::shared_ptr<Value>> params;

    for (int k = 0; k < 2000; k += 1) {
      // Forward pass
      ypred = std::vector<std::shared_ptr<Value>> {
        n(xs[0])[0],
        n(xs[1])[0],
        n(xs[2])[0],
        n(xs[3])[0],
      };
      err = std::vector<std::shared_ptr<Value>> {
        pow(ypred[0] - ys[0], 2.0f),
        pow(ypred[1] - ys[1], 2.0f),
        pow(ypred[2] - ys[2], 2.0f),
        pow(ypred[3] - ys[3], 2.0f),
      };
      loss = err[0] + err[1] + err[2] + err[3];

      // Backward pass
      params.clear();
      n.parameters(params);
      for (auto& p : params) {
        p->grad = 0.0f;
      }
      loss->backward();

      // Update
      for (auto& p : params) {
        p->data += -0.02f * p->grad;
      }

      std::cout << k << " " << loss->data << std::endl;
    }

    ypred[0]->print();
    ypred[1]->print();
    ypred[2]->print();
    ypred[3]->print();
  }

  return 0;
}
