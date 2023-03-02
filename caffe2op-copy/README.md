# caffe2op-copy

## Description

This Rust crate provides operators and functions
for efficiently copying and manipulating data in
DSP and machine learning computations. One of the
core operators in this crate is
`CopyRowsToTensorOp`, which copies a set of rows
from an input tensor to an output tensor, creating
a new tensor with the specified rows. The
`CopyRowsToTensorGradientOp` performs the gradient
computation for this operation.

Other operators in this crate include
`CopyCPUToGPUOp`, which copies data from a CPU
tensor to a GPU tensor, and `CopyGPUToCPUOp`,
which performs the reverse operation. The
`CopyOnDeviceLikeOp` copies data from one tensor
to another tensor on the same device, and the
`Identity` operator simply returns its input as
its output.

This crate also provides functions for working
with blobs, including `FeedBlob` and `FetchBlob`,
which respectively load and retrieve data from
a blob, and `ResetWorkspace`, which clears the
workspace of any previously loaded blobs.

## Symbols

- `CopyRowsToTensorOp`: The main operator of this
  crate, which copies a set of rows from an input
  tensor to an output tensor.

- `CopyRowsToTensorGradientOp`: The gradient
  computation for `CopyRowsToTensorOp`.

- `CopyCPUToGPUOp`: Copies data from a CPU tensor
  to a GPU tensor.

- `CopyGPUToCPUOp`: Copies data from a GPU tensor
  to a CPU tensor.

- `CopyOnDeviceLikeOp`: Copies data from one
  tensor to another tensor on the same device.

- `Identity`: Returns its input as its output.

- `FeedBlob`: Loads data into a blob.

- `FetchBlob`: Retrieves data from a blob.

- `ResetWorkspace`: Clears the workspace of any
  previously loaded blobs.

## Mathematical Analysis

The `CopyRowsToTensorOp` operator can be expressed
mathematically as follows, where `input` and
`output` are the input and output tensors,
respectively, and `rows` is a vector specifying
which rows to copy:

```
output[i, :] = input[rows[i], :]
```

The gradient computation for this operation,
`CopyRowsToTensorGradientOp`, can be expressed
mathematically as follows:

```
grad_input[rows[i], :] += grad_output[i, :]
```

where `grad_input` and `grad_output` are the
gradient tensors for the input and output,
respectively.
