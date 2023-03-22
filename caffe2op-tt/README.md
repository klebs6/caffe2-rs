# caffe2op-tt crate description

## Description

This rust crate is called caffe2op-tt, and it
defines a mathematical operator used in DSP and
machine learning computations. Specifically, this
crate implements the TTLinearOp and
TTLinearGradientOp operations, which are used in
tensor train (TT) decompositions.

TT decompositions are a type of tensor
factorization that decomposes a high-dimensional
tensor into a series of lower-dimensional
tensors. This allows for efficient storage and
manipulation of high-dimensional tensors, which
can have a large number of parameters. The
TTLinearOp and TTLinearGradientOp operations are
used to perform linear operations on tensors that
have been decomposed using a TT decomposition.

## Mathematical Analysis

The TTLinearOp and TTLinearGradientOp operations
are used in TT decompositions to perform linear
operations on tensors that have been decomposed
into a series of smaller tensors. The TT
decomposition expresses a high-dimensional tensor
as a sequence of lower-dimensional tensors, called
cores. These cores are arranged in a specific
order, and each core corresponds to a mode of the
original tensor.

The TTLinearOp operation performs a linear
operation on a tensor that has been decomposed
using a TT decomposition. The TTLinearGradientOp
operation computes the gradient of the TTLinearOp
operation with respect to its input.

## Code Example

Here is an example of how to use the TTLinearOp
and TTLinearGradientOp operations in Rust:

```
use caffe2op_tt::{TTLinearOp, TTLinearGradientOp};

// Define a tensor that has been decomposed using a TT decomposition
let tensor = ...;

// Define the weights for the linear operation
let weights = ...;

// Compute the linear operation using the TTLinearOp operation
let result = TTLinearOp::run(&[&tensor, &weights]);

// Compute the gradient of the linear operation using the TTLinearGradientOp operation
let gradient = TTLinearGradientOp::run(&[&result.gradient.unwrap(), &weights, &tensor]);
```

## Considerations

When using the TTLinearOp and TTLinearGradientOp
operations, it is important to consider the size
and shape of the tensors being operated on, as
well as the order of the cores in the TT
decomposition. The choice of core order can have
a significant impact on the efficiency of the TT
decomposition, as well as the accuracy of the
resulting decomposition.

Additionally, it is important to note that the
TTLinearGradientOp operation is currently not
implemented in this crate.

87> could you show me how the TT decomposition
works in more detail?

Sure! The TT decomposition is a way of
representing a high-dimensional tensor as
a sequence of smaller, low-rank tensors. The
decomposition is named after its inventors, Ivan
Oseledets and Eugene Tyrtyshnikov, and is also
known as the Tensor Train decomposition.

Here's an example of the TT decomposition for
a 3-dimensional tensor:

Suppose we have a 3-dimensional tensor T of shape
(n1, n2, n3). We can represent T as a sequence of
smaller tensors A1, A2, A3, and A4, where each
tensor Ai has dimensions (ri, ni, ri+1) for some
rank ri, and r1 = r5 = 1. Here's a diagram of the
decomposition:

```
        ----(ri)----
       /            \
(n1) --(ni)   (ri+1)-- (n2) --(ri+2)-- (n3)
       \            /
        ----(ri)----
```

To compute the elements of the original tensor
T from the sequence of smaller tensors Ai, we can
use the following formula:

```
T(i1, i2, i3) = A1(i1, :) * B1 * A2(i2, :) * B2 * A3(i3, :)
```

Here, B1 and B2 are diagonal matrices with entries
equal to 1, except for the (ri x ri+1) entries in
the middle, which are the elements of the tensor
Ai+1.

The TT decomposition has several useful
properties. For example, it can be used to
represent a high-dimensional tensor using much
less memory than the original tensor, making it
useful for large-scale machine learning
applications. It also allows us to perform certain
tensor operations (such as matrix multiplication
and convolution) more efficiently than would be
possible with the original tensor.

The TT decomposition has been used in a variety of
machine learning applications, including image and
text processing, and has been shown to be
effective for compressing and analyzing
high-dimensional data.

88> which sorts of networks does it appear in?

The TT decomposition has been used in various
neural network architectures, including tensor
train networks and tensor product networks. These
networks are commonly used in tasks such as image
classification, natural language processing, and
other machine learning applications. The TT
decomposition allows for efficient computation and
storage of high-dimensional data, which is crucial
in deep learning applications where large amounts
of data must be processed quickly.

89> what is a tensor train network? what is
a tensor product network? how do these work in
detail?

A tensor train (TT) network and a tensor product
network (TP) are two types of tensor network
models used in machine learning and related
fields.

A tensor train network is a type of compressed
representation of high-dimensional tensors that is
based on a factorization of the tensor into
a sequence of low-rank tensors. It is used in
applications where the high dimensionality of the
data makes computations with the full tensor
infeasible. The basic idea behind the TT
decomposition is to express a high-dimensional
tensor as a product of low-dimensional tensors,
where each low-dimensional tensor captures
interactions between a subset of the original
dimensions.

Mathematically, a tensor train can be defined as
a sequence of tensors {G_k}, k = 1, ..., K, where
each tensor G_k has dimensions (R_{k-1}, d_k,
R_k), and R_0 = R_K = 1. The tensor train can then
be defined as the product of these tensors as
follows:

A(i_1, ..., i_d) = G_1(:, i_1, 1) G_2(:, i_2, :) ... G_K(:, i_K, R_K)

where A is the original high-dimensional tensor of
dimensions (d_1, ..., d_K), and the notation ":"
denotes a full row or column of the tensor.

The TT decomposition allows for efficient
computation of operations such as matrix
multiplication, convolution, and the solution of
linear systems of equations, all of which are
computationally expensive when performed on the
original high-dimensional tensor. It has
applications in a wide range of fields, including
physics, chemistry, and machine learning.

A tensor product network, on the other hand, is
a type of deep neural network that is based on the
Kronecker product of matrices. It is used for
high-dimensional data classification and
regression problems. A tensor product network can
be represented as a series of tensor products of
matrices, where each matrix represents a layer of
the network.

Mathematically, a tensor product network can be
defined as follows:

f(x) = (W_L (x) W_{L-1} ... (x) W_1) v

where W_1, ..., W_L are weight matrices, v is
a vector of weights, and (x) denotes the Kronecker
product. The input x is a high-dimensional vector,
and the output f(x) is a scalar.

Tensor product networks are used in tasks such as
image classification and natural language
processing, where the input data has a high
dimensionality. They have been shown to achieve
state-of-the-art performance on a variety of
benchmarks.

90> what is the kronecker product of matrices? in
the TT decomposition, how do we know each low
dimensional tensor captures interactions between
a subset of the original dimensions

The Kronecker product of two matrices A and B,
denoted as A ⊗ B, is a block matrix formed by
taking all possible products of the entries of
A and B. Specifically, if A is an m × n matrix and
B is a p × q matrix, then their Kronecker product
A ⊗ B is an mp × nq block matrix given by:

```
A ⊗ B = [ a_11 * B  a_12 * B  ...  a_1n * B ]
         [ a_21 * B  a_22 * B  ...  a_2n * B ]
         [  ...       ...      ...    ...    ]
         [ a_m1 * B  a_m2 * B  ...  a_mn * B ]
```

where a_ij denotes the (i,j)-th entry of A.

In the TT decomposition, each low dimensional
tensor captures interactions between a subset of
the original dimensions by taking the Kronecker
product of matrices. Specifically, the TT
decomposition of a high-dimensional tensor can be
viewed as a set of low-dimensional tensors, each
capturing interactions between a subset of the
original dimensions. The Kronecker product is used
to combine these low-dimensional tensors to obtain
the high-dimensional tensor. The TT decomposition
uses a sequence of such Kronecker products to
build up the tensor. Each low-dimensional tensor
is represented as a matrix of size R(i) × R(i+1),
where R(i) denotes the TT-rank at the i-th
dimension, and R(0) = R(d) = 1 for a tensor of
order d.

In this way, the TT decomposition expresses
a high-dimensional tensor as a sequence of
low-dimensional tensors, where each
low-dimensional tensor captures interactions
between a subset of the original dimensions. This
makes the TT decomposition particularly
well-suited for high-dimensional tensors, where
the number of entries grows exponentially with the
dimensionality.

91> thanks -- what else should i know about the TT
operator?

The TT operator is often used as a compression
technique for high-dimensional tensors. It can be
used to represent a high-dimensional tensor in
a compact form that requires less memory and
computation. This can be especially useful in deep
learning applications where large tensors are
commonly used, and memory and computation
resources are limited.

The TT decomposition can also be used to perform
tensor operations more efficiently. For example,
the tensor product of two TT decomposed tensors
can be computed efficiently by performing a series
of matrix-vector multiplications. This is faster
than directly computing the tensor product of the
original high-dimensional tensors.

It's worth noting that the TT decomposition is not
without limitations. It may not work well for
tensors with a high degree of sparsity or for
tensors with certain patterns of dependencies. In
addition, the quality of the approximation can be
affected by the choice of the TT-rank, which
determines the size of the low-dimensional tensors
in the decomposition.

92> what are its performance implications?

The performance implications of the TT operator
depend on various factors such as the size and
shape of the input tensors, the number of
dimensions in the tensor train, and the hardware
being used to perform the computation.

In general, the TT operator can be computationally
expensive due to the large number of matrix
multiplications involved in the decomposition
process. Additionally, as the number of dimensions
in the tensor train increases, the amount of
computation required grows exponentially.

However, the TT decomposition can offer
significant performance advantages for certain
types of tensor operations. In particular, the TT
format can be highly efficient for computing
certain types of matrix products, such as those
arising in neural network operations like
convolution and matrix multiplication. The use of
the TT format can also lead to reduced memory
usage and faster convergence during training of
neural networks, which can translate into improved
performance and faster training times.

93> how would you calculate the performance of
a TT operator as a back of the envelope
calculation? which metrics would you look for to
track?

The performance of a TT operator can be estimated
using several metrics, including the number of
floating-point operations (FLOPs) required to
perform a forward or backward pass, the memory
usage of the operator, and the communication costs
if the operator is distributed across multiple
devices.

To perform a rough estimate of the FLOPs required
by a TT operator, we can use the number of
parameters and the size of the input and output
tensors. Suppose the input tensor has shape `(N1,
N2, ..., Nd)`, and the TT operator has rank `r`
and TT dimensions `d_1, d_2, ..., d_{d+1}`. The
number of parameters in the TT operator can be
computed as `r * (N1 * d_1 + d_1 * d_2 + ... + d_d
* N_d) * d_{d+1}`. To compute a forward pass, we
need to perform `N1 * r * d_1` operations to
transform the input tensor into a tensor of shape
`(r, d_2, ..., d_d, N_{d+1})`. Then, we need to
perform `d_2 * r^2 * d_3 + ... + d_{d} * r^2
* N_{d+1}` operations to contract the TT cores and
compute the output tensor of shape
`(N_{d+1},)`. Thus, the total number of FLOPs for
a forward pass is roughly proportional to `N1
* r * d_1 + (d_2 * r^2 * d_3 + ... + d_{d} * r^2
* N_{d+1})`. A similar calculation can be
performed for a backward pass.

Memory usage can be estimated by computing the
size of the TT cores and any intermediate tensors
required to compute a forward or backward
pass. Communication costs depend on the
distribution of the TT operator across multiple
devices and the size of the tensors being
communicated.

94> thanks -- could you show me an example
calculation?

Sure, let's consider a simple example where we
have a tensor with dimensions `(10, 20, 30)` that
we want to decompose into a TT format. Let's
assume we want to use a rank-2 decomposition,
meaning we'll end up with 3 TT tensors with
dimensions `(10, 2), (2, 20, 2), (2, 30)`.

To calculate the performance of this operation, we
can estimate the number of floating point
operations required. In the case of TT
decomposition, the main operations are
matrix-vector multiplications and Kronecker
products. For a rank-2 TT decomposition, we need
to perform 5 matrix-vector multiplications and
4 Kronecker products per mode
(dimension). Therefore, the total number of
operations is:

```
(10 x 2) x 2 + (2 x 20 x 2) x 2 + (2 x 30) x 2 + 3 x 2 x 2 = 964
```

Here, we've multiplied the number of matrices by
the size of each matrix, and then multiplied by
2 to account for the fact that we need to multiply
each matrix by a vector twice (once forward and
once backward during backpropagation). Finally, we
multiply by 3 to account for the fact that we have
3 TT tensors in total.

This gives us an estimate of the total number of
floating point operations required to perform the
TT decomposition. We can use this estimate to
compare the performance of the TT operator to
other operations in our computation graph and
determine if it is a bottleneck or not.

95> are there any other metrics we would want to
know about?

Some other metrics to consider when evaluating the
performance of a TT operator include:

- Memory usage: Since the TT decomposition can
  reduce the size of a tensor, it can lead to
  significant memory savings. The memory usage of
  a TT operator can be estimated based on the
  number of parameters it contains and the size of
  the original tensor.

- FLOPS: The number of floating-point operations
  required to compute the output of a TT operator
  is another important metric for performance
  evaluation. This can be estimated based on the
  size of the input tensor and the number of
  parameters in the TT operator.

- Convergence rate: When using the TT
  decomposition for iterative methods such as
  optimization or solving differential equations,
  it is important to consider the convergence rate
  of the method. This can depend on factors such
  as the TT rank, the size of the original tensor,
  and the particular optimization algorithm used.

96> could you show me how to calculate these?

Sure, here are some ways to calculate those
additional metrics:

- **Number of Parameters**: The number of
  parameters in a TT decomposition of a tensor of
  shape `(d_1, d_2, ..., d_n)` with a maximum rank
  `r` is given by:

```
num_params = r^2 * (d_1 + d_n) + r^2 * sum(d_i * d_{i+1}) for i in 1 to n-1
```

- **FLOPS**: The number of FLOPS required to
  compute the TT decomposition of a tensor of
  shape `(d_1, d_2, ..., d_n)` with a maximum rank
  `r` is given by:

```
flops = d_1 * r^2 * d_2 + r^2 * sum(d_i * r^2 * d_{i+1}) for i in 2 to n-1
```

- **Compression Ratio**: The compression ratio
  achieved by a TT decomposition of a tensor of
  shape `(d_1, d_2, ..., d_n)` with a maximum rank
  `r` is given by:

```
compression_ratio = (num_params / n) / prod(d_i) = r^2 / prod(d_i / n)
```

Here, `prod(d_i)` is the product of all dimensions
of the original tensor, and `n` is the number of
dimensions. The factor `d_i / n` is included to
account for the fact that the TT decomposition
stores `r^2` parameters for each pair of adjacent
dimensions.

Note that these calculations assume that the
original tensor is stored in full form with no
compression, so the actual compression achieved by
a TT decomposition may be higher depending on the
sparsity of the tensor.
