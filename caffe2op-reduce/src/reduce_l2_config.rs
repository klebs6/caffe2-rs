crate::ix!();

/**
  | Computes the **L2 norm** of the input
  | tensor's elements along the provided
  | `axes`.
  | 
  | The resulting tensor has the same rank
  | as the input if the `keepdims` argument
  | equals 1 (default). If `keepdims` is
  | set to 0, then the `axes` dimensions
  | are pruned.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc
  |
  */
register_cpu_operator!{
    ReduceL2,
    ReduceOp<TensorTypes<float>, CPUContext, L2Reducer<CPUContext>>
}

num_inputs!{ReduceL2, 1}

num_outputs!{ReduceL2, 1}

inputs!{ReduceL2, 
    0 => ("X", "(*Tensor`<float>`*): input tensor")
}

outputs!{ReduceL2, 
    0 => ("Y", "(*Tensor`<float>`*): reduced tensor")
}

args!{ReduceL2, 
    0 => ("axes", "(*Tuple(int)*): list of axes to reduce"),
    1 => ("keepdims", "(*int*): set to 1 to keep the reduced dimension(s) (default=1), else set to 0 to not keep the reduced dimension(s)")
}

inherit_onnx_schema!{ReduceL2, "ReduceMean"}

///--------------------------------
register_cpu_operator!{
    ReduceL2Gradient,
    ReduceGradientOp<TensorTypes<float>, CPUContext, L2Reducer<CPUContext>>
}

num_inputs!{ReduceL2Gradient, 3}

num_outputs!{ReduceL2Gradient, 1}
