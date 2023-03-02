
The `caffe2op-accum` crate provides an operator
for accumulating values in networked digital
signal processing and deep learning
computations. This crate defines the
`AccumulateOp` which performs accumulation on the
input tensors according to a specified axis. The
resulting tensor is returned as the output. The
`Accumulated` token is used to refer to the output
tensor.

The operation performed by `AccumulateOp` can be
represented mathematically as follows:

If the input tensor is of shape `[a, b, c, ..., m,
n]` and the accumulation axis is `k`, then the
output tensor will be of shape `[a, b, c, ..., l,
n]` where `l` is the size of the tensor along the
accumulation axis. The elements of the output
tensor are calculated as follows:

```
output[i1, i2, ..., il, j] 

= sum(input[i1, i2, ..., i_{k-1}, l, i_{k+1}, ..., i_{m}, j])

```

where the sum is over all `l` along the
accumulation axis.


The `caffe2op-accum` crate also provides tokens
such as `Accumulation`, `accumulates`,
`accumulations`, `depends`, `fiddles`, `interim`,
and `reshaped`, which are used within the
implementation of the accumulation operation.


