# caffe2op-stump

## Overview

`caffe2op-stump` is a Rust crate that defines two
mathematical operators used in digital signal
processing and machine learning computations. The
operators perform a thresholding operation on
a given input tensor and return either the
thresholded values or the indices of the elements
above and below the threshold.

The crate is in the process of being translated
from C++ to Rust, so some of the function bodies
may still be in the process of translation.

## Operators

### `StumpFuncOp`

`StumpFuncOp` is a thresholding operator that
takes an input tensor of floats and converts each
element into either a high or low value based on
a given threshold.

Specifically, given an input tensor `X` and
threshold `threshold`, the operator computes an
output tensor `Y` of the same shape as `X`, where
the element-wise value of `Y` at index `i` is
defined as:

```
Y[i] = low_value if X[i] <= threshold
       else high_value
```

### `StumpFuncIndexOp`

`StumpFuncIndexOp` is a thresholding operator that
returns the indices of the elements that are above
and below the threshold in the input tensor.

Specifically, given an input tensor `X` of floats
and threshold `threshold`, the operator computes
two output tensors, `Index_Low` and `Index_High`,
both of data type `int64`, with the same shape as
`X`. The element-wise value of `Index_Low` at
index `i` is equal to the index of the i-th
element of `X` if the value of the i-th element is
less than or equal to `threshold`. Similarly, the
element-wise value of `Index_High` at index `i` is
equal to the index of the i-th element of `X` if
the value of the i-th element is greater than
`threshold`.

## Naming

The operators are named `StumpFuncOp` and
`StumpFuncIndexOp`. It's possible that the authors
of the crate chose this name based on the concept
of a decision stump in machine
learning. A decision stump is a simple model used
in binary classification problems, where the goal
is to partition the input space into two regions
based on a threshold applied to a single
feature. The name "stump" comes from the fact that
the model is a simple tree with one decision node
(i.e., a stump).

Alternatively, the name may be related to
a specific use case or domain that is not
immediately apparent from the code snippets.
