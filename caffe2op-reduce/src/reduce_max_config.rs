crate::ix!();

/**
  | Computes the max of the input tensor's
  | element along the provided axes.
  | 
  | The resulted tensor has the same rank
  | as the input if keepdims equal True.
  | 
  | If keepdims equal false, then the resulted
  | tensor have the reduced dimension pruned.
  |
  */
register_cpu_operator!{
    ReduceMax,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MaxReducer<CPUContext>>
}

num_inputs!{ReduceMax, 1}

num_outputs!{ReduceMax, 1}

inputs!{ReduceMax, 
    0 => ("data", "An input tensor.")
}

outputs!{ReduceMax, 
    0 => ("reduced", "Reduced output tensor.")
}

args!{ReduceMax, 
    0 => ("axes", "A list of integers, along which to reduce."),
    1 => ("keepdims", "Keep the reduced dimension(s) or not, default True keeps the reduced dimension(s).")
}

///------------------------
register_cpu_operator!{
    ReduceMaxGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MaxReducer<CPUContext>>
}

num_inputs!{ReduceMaxGradient, 3}

num_outputs!{ReduceMaxGradient, 1}
