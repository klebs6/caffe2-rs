# caffe2op-lengthstop

The `LengthsTopKOp` and `LengthsTopKGradientOp`
operators implemented in this Rust crate are used
in deep learning and digital signal processing
applications to perform top-K pooling on data with
varying lengths. Given an input tensor of rank
r >= 1 and a scalar K, the `LengthsTopKOp`
operator returns the top-K elements for each
sequence in the outermost dimension of the input
tensor, where K can be different for each
sequence. The output tensor has the same shape as
the input tensor, except that the outermost
dimension is now padded with zeros to ensure that
all sequences have the same length. The
`LengthsTopKGradientOp` operator computes the
gradients of the `LengthsTopKOp` operator.

Mathematically, let `X` be an input tensor of
shape `(N, L, d_2, ..., d_r)` where `N` is the
batch size, `L` is the maximum length of the
sequences, and `d_2, ..., d_r` are the dimensions
of the remaining axes. Let `K` be a scalar
indicating the number of elements to be selected
for each sequence, and let `Y` be the output
tensor of shape `(N, L, d_2, ..., d_r)` obtained
by padding each sequence with zeros to length
`L`. Then the `LengthsTopKOp` operator can be
defined mathematically as follows:

```
for n in 0..N:
    for l in 0..L:
        values = X[n, l, :]
        topk_indices = indices_of_top_k_elements(values, K[n])
        Y[n, l, topk_indices, ...] = X[n, l, topk_indices, ...]
```

The `LengthsTopKGradientOp` operator computes the
gradients of the `LengthsTopKOp` operator with
respect to the input tensor `X`. Let `dY` be the
output gradient tensor of shape `(N, L, d_2, ...,
d_r)`. Then the `LengthsTopKGradientOp` operator
can be defined mathematically as follows:

```
for n in 0..N:
    for l in 0..L:
        topk_indices = indices_of_top_k_elements(X[n, l, :], K[n])
        dX[n, l, topk_indices, ...] = dY[n, l, topk_indices, ...]
```

When using these operators, it is important to
note that `K` must be a scalar or a 1D tensor of
size `N`, where `N` is the batch size. The input
tensor `X` must have a known maximum sequence
length `L`. The `LengthsTopKOp` operator may
introduce padding to the sequences in the output
tensor, which can increase memory
usage. Therefore, it is recommended to use the
`LengthsTopKOp` operator sparingly and only when
necessary for the application.

32> could you describe top k pooling in more
detail? why do we use it?

Sure! Top-k pooling is a type of pooling operation
commonly used in deep learning for selecting the
k largest values from a set of values. It is used
to reduce the dimensionality of the input data
while preserving the most relevant information.

In the case of the LengthsTopK operator, the
pooling operation is performed on
a multi-dimensional tensor with an additional
lengths tensor specifying the number of elements
to consider for each sample. The output tensor is
constructed by selecting the top k elements from
each sample and concatenating them together.

Mathematically, given an input tensor X with shape
`(batch_size, sequence_length, feature_dim)` and
a lengths tensor L with shape `(batch_size,)`, the
LengthsTopK operator computes an output tensor
Y with shape `(batch_size, k, feature_dim)` as
follows:

Given an input tensor `data` of shape `(N, L, D)`
and a `lengths` tensor of shape `(N,)`, the
`LengthsTopKOp` operator computes the following
output tensor `output` of shape `(N, K, D)`:

```
output[i, j, :] = data[i, indices[i, j], :]
```

where `indices` is a tensor of shape `(N, K)`
containing the indices of the `K` largest elements
in each slice of `data` along the `L` dimension,
as determined by the corresponding
`lengths[i]`. Specifically:

```
for i in range(N):
    topk_indices = indices[i, :]
    topk_values = data[i, topk_indices, :]
    sorted_indices = topk_indices.argsort()
    output[i, :, :] = topk_values[sorted_indices, :]
```

The `LengthsTopKGradientOp` operator computes the
gradients of the `LengthsTopKOp` operator with
respect to the input tensor `data`. Given input
tensors `data`, `lengths`, `output_grad` (the
gradient of the output tensor with respect to some
downstream loss), and `indices` (the indices used
by `LengthsTopKOp` to compute `output`), the
`LengthsTopKGradientOp` computes the gradient
tensor `data_grad` of the same shape as `data` as
follows:

```
data_grad[i, j, :] = output_grad[i, k, :] if indices[i, k] == j else 0.0
```

where `k` is the index such that `indices[i, k] ==
j`.

35> why might we want to use such an operator as
this?

The LengthsTopK operator is useful in scenarios
where we need to perform a pooling operation
across variable-length sequences or sets. This is
a common task in natural language processing,
where we often deal with sequences of varying
lengths such as sentences of different lengths or
documents of different lengths. The operator can
be used to extract the top-k elements from each
sequence or set and output them in a fixed-length
tensor, which can then be used as input to
subsequent layers in a neural network. The
fixed-length tensor representation allows us to
feed the sequence data to other layers that expect
fixed-size inputs, such as fully connected layers
or convolutional layers. Overall, the LengthsTopK
operator provides a useful tool for processing
variable-length sequences or sets in a neural
network.

36> what are the performance characteristics of
this operator? what is the memory footprint? could
you please perform an example calculation of each?

The performance characteristics of this operator
depend on the size of the input tensor and the
value of the `k` parameter. In the worst case,
where `k` is close to the size of the outer-most
dimension of the input tensor, the operator has to
perform a full sort of the input tensor, which has
time complexity O(N log N) where N is the size of
the input tensor. However, in practice, `k` is
often much smaller than the size of the input
tensor, which reduces the complexity of the
operator.

The memory footprint of the operator is also
dependent on the size of the input tensor and the
value of the `k` parameter. In the worst case,
where `k` is close to the size of the outer-most
dimension of the input tensor, the operator may
need to allocate additional memory to perform the
sort. However, in practice, the memory footprint
is often smaller than the size of the input
tensor.

Here is an example calculation of the performance
characteristics and memory footprint of the
operator for an input tensor of size 100x100 with
`k=10`:

- Performance: The operator needs to perform
  a partial sort of the input tensor along the
  outer-most dimension, which has time complexity
  O(N log k) where N is the size of the input
  tensor and k is the value of the `k`
  parameter. In this case, the time complexity is
  O(10000 log 10) = O(40000).

- Memory footprint: The operator needs to allocate
  memory for the output tensor, which has the same
  size as the input tensor, and for the indices of
  the top k elements along the outer-most
  dimension, which has size 10. In this case, the
  memory footprint is 10000 + 10 = 10010.

37> where is it commonly found in a dag? does it
have parameters? if so, what are they?

The `LengthsTopKOp` operator can be useful in
various deep learning models, particularly in
natural language processing tasks that involve
variable-length sequences. It can be located at
different places within a deep learning model,
depending on the specific application.

This operator does have one parameter, which is
`k` (the number of top elements to keep). This
parameter specifies the number of elements in each
sequence that will be kept, with the others being
discarded.

In a typical deep learning model, the
`LengthsTopKOp` operator may be used in
conjunction with other operations, such as
embedding layers, recurrent layers, or
convolutional layers. It can be used to select the
most important features from a sequence of data
before feeding them into subsequent layers.
