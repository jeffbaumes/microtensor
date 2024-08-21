# microtensor

A from-scratch C++ implementation of the functionality of the [Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) series by Andrej Karpathy.
This is an educational codebase, not a serious-use library.

* lesson-01.cpp - [The spelled-out intro to neural networks and backpropagation: building micrograd](https://youtu.be/VMj-3S1tku0)
* lesson-02.cpp - [The spelled-out intro to language modeling: building makemore](https://youtu.be/PaCmpygFfXo)
* lesson-03.cpp - [Building makemore Part 2: MLP](https://youtu.be/TCH_1BHY58I)
* lesson-04.cpp - [Building makemore Part 3: Activations & Gradients, BatchNorm](https://youtu.be/P6sfmUTpUmc)
* lesson-05.cpp - [Building makemore Part 4: Becoming a Backprop Ninja](https://youtu.be/q8SA3rM6ckI) (in progress)

## Features and discoveries

As I went along, I discovered this ended up in some ways a reimplementation of the CPU backend of the pytorch ATEN
library with added back propagation, since I was mostly trying to follow pytorch/numpy API. But I didn't look at the ATEN
implementation to write this.

It utilizes C++ lambdas for backprop similar to the pattern Andrej uses in micrograd library.
It seems elegant to capture what's needed for backprop in the C++ lambda capture.

Broadcasts of addition and multiplication are supported. It was fun to figure out that a broadcast can work as intended just by
thinking of the broadcasted dimensions as having a stride of zero, which will end up reusing and accumulating data as desired.

I learned that iterating an n-dimensional tensor (especially one with arbitrary strides that is not assumed to be contiguous)
is mind-bending. You clearly can't have N nested loops in your code since you don't know what N is in general.
My iteration code ended up having roughly the following pattern:

```
for i from 0 to num_elements (product of shape vector):
   construct the n-dimensional index for element i (a for loop with n iterations)
   convert the n-dimensional index to a linear index based on strides
   after getting the linear index for each relevant input/output tensor, you can do what you need to do
```

It's not as fast as pytorch (even just using pytorch CPU), but to get even within an order of magnitude with a fairly minimal library was I think a success.
I was able to replicate training the models shown in the videos fairly accurately.

I made both an `Array` class (which is an N-dimensional array/tensor) and a `Tensor` class (which supports autograd) which had two Arrays: `data` and `grad`.
I like knowing when I'm working at a gradient-computing level (`Tensor`) and a basic operations level (`Array`) but it did mean that I needed
to replicate the class API, operators, and other functions for both `Tensor` and `Array`. Not much code other than the function signatures is duplicated,
but it still seems to be more code/API than necessary.

I played with using Intel MKL for optimizing matrix multiplication, I think on my modest matrices in the tutorials it had maybe a 30% speedup. Not earth shattering but I'm glad it at least had some effect. I may experiment with some CUDA backend stuff at some point.

It makes very heavy use of `std::shared_ptr`, which let me focus on logic and not worry much about memory leaks.

There is a bit of testing with Google Test.

By implementing a hand-rolled cross entropy as described in lesson 5, I got a 20x speedup on the forward pass and a 100x speedup on the backward
pass, which was very satisfying. Math FTW!

## To configure

```
ccmake -S . -B build
```

## To blow things away and start over

```
rm -rf build && ccmake -S . -B build
```

## To build

```
cmake --build build
```

## To build and test

```
cmake --build build && cd build && (ctest || true) && cd ..
```

## To build and run a lesson

```
cmake --build build && ./build/lesson04
```
