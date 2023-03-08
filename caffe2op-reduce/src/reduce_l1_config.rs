crate::ix!();

/**
  | Computes the **L1 norm** of the input
  | tensor's elements along the provided
  | `axes`. The resulting tensor has the
  | same rank as the input if the `keepdims`
  | argument equals 1 (default). If `keepdims`
  | is set to 0, then the `axes` dimensions
  | are pruned.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc
  |
  */
register_cpu_operator!{
    ReduceL1,
    ReduceOp<TensorTypes<float>, CPUContext, L1Reducer<CPUContext>>
}

num_inputs!{ReduceL1, 1}

num_outputs!{ReduceL1, 1}

inputs!{ReduceL1, 
    0 => ("X", "(*Tensor`<float>`*): input tensor")
}

outputs!{ReduceL1, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceL1, 
    0 => ("axes", "(*Tuple(int)*): list of axes to reduce"),
    1 => ("keepdims", "(*int*): set to 1 to keep the reduced dimension(s) 
        (default=1), else set to 0 to not keep the reduced dimension(s)")
}

register_cpu_operator!{
    ReduceL1Gradient,
    ReduceGradientOp<TensorTypes<float>, CPUContext, L1Reducer<CPUContext>>
}

num_inputs!{ReduceL1Gradient, 3}

num_outputs!{ReduceL1Gradient, 1}
