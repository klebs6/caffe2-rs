# `ONNXWhileOp`

The `ONNXWhileOp` is a mathematical operator used
in DSP and machine learning computations that
provides a looping mechanism to execute a subgraph
multiple times. This operator takes in two inputs:
a condition tensor `cond` and a loop body `body`
which is executed repeatedly as long as the
condition is true. The loop continues until the
condition is false, or a maximum number of
iterations is reached.

The `ONNXWhileOp` is typically used in deep
learning frameworks to implement recurrent neural
networks (RNNs) and other sequence models. In
these models, the `cond` tensor represents
a stopping condition based on the input sequence,
and the `body` tensor represents the operations
performed on each element of the sequence.

Mathematically, the `ONNXWhileOp` can be
represented as follows:

```
while (cond) {
  body
}
```

where `cond` is a boolean tensor and `body` is
a subgraph of operations to be executed during
each iteration of the loop.

# `OnnxWhileOpLocalScope`

The `OnnxWhileOpLocalScope` is a Rust crate struct
that provides a local scope for a `ONNXWhileOp`
operator. This allows for the creation of new
tensors and variables that are local to the loop
body, and are not visible outside of the loop.

# `allow_inplace`

The `allow_inplace` flag is a boolean value that
determines whether the `ONNXWhileOp` operator
allows for in-place computation. In-place
computation can lead to more efficient memory
usage, but can also introduce errors if not used
carefully.

# `do_run_with_type`

The `do_run_with_type` function is a Rust crate
function that executes the `ONNXWhileOp` operator
on a given input tensor with a specified data
type.

# `iteration`

The `iteration` parameter is an integer value that
determines the maximum number of iterations that
the `ONNXWhileOp` operator will execute. If this
parameter is not set, the loop will continue
indefinitely until the condition tensor becomes
false.

# `lcd_tensor`

The `lcd_tensor` is a Rust crate struct that
represents a low-level tensor object used in the
execution of the `ONNXWhileOp` operator.

# `net`

The `net` parameter is a string value that
specifies the name of the neural network that the
`ONNXWhileOp` operator is a part of. This
parameter is used to group related operations
together.

# `register_cpu_operator`

The `register_cpu_operator` function is a Rust
crate function that registers the `ONNXWhileOp`
operator as a CPU operator for use in machine
learning computations.

# `run_on_device`

The `run_on_device` function is a Rust crate
function that executes the `ONNXWhileOp` operator
on a specified device (e.g., CPU or GPU).

# `set_iteration`

The `set_iteration` function is a Rust crate
function that sets the maximum number of
iterations for the `ONNXWhileOp` operator.

# `workspace`

The `workspace` is a Rust crate struct that
represents the memory workspace used by the
`ONNXWhileOp` operator during its execution. The
workspace contains the intermediate results of the
loop body computations and is reused for
subsequent iterations of the loop.
