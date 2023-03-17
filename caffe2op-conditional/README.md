## caffe2op-conditional crate description

### ConditionalOp

The `ConditionalOp` is a mathematical operator
used in deep learning computations. The operator
computes a conditional statement on a given input
and produces an output based on the
condition. This is commonly used in various neural
network architectures such as conditional
generative models, where the output is conditioned
on some input.

**Note: This crate is currently being translated from C++ to Rust, and some function bodies may still be in the process of translation.**

The conditional operator can be defined as
follows:

Given an input `X`, we compute a boolean condition
`C` on `X`. We then compute two functions `F` and
`T` on `X`, such that `F` is applied if the
condition is `false` and `T` is applied if the
condition is `true`. The output `Y` is then
defined as:

```
Y = C ? T(X) : F(X)
```

where `?` represents the conditional operator.

### Usage

The `ConditionalOp` operator can be used in
various deep learning models, such as conditional
generative models, where the output is conditioned
on some input. The `caffe2op-conditional` crate
provides an implementation of the operator in
Rust, which can be used in projects that require
efficient computation of conditional statements.

When using this crate, some considerations to note
include ensuring that the input tensors have
compatible shapes and sizes, as well as efficient
memory allocation and management for the output
tensor.

Overall, the `caffe2op-conditional` crate can be
a useful tool for implementing conditional
statements in deep learning computations,
providing efficient and flexible computation of
boolean conditions on input data.

### Examples

46> cool! in which architectures is this sort of
operator used?

The Conditional operator can be used in various
types of architectures where conditional
computations are required, such as in certain
types of neural networks. For example, in
recurrent neural networks, the output of
a previous time step can be used as input for the
current time step. However, the computation at the
current time step may depend on some condition or
other inputs. In such cases, the Conditional
operator can be used to perform the appropriate
computations based on the given condition or
inputs. Similarly, in certain types of generative
models like GANs (Generative Adversarial
Networks), the generator network can use
Conditional operators to generate outputs based on
a given condition or label.
