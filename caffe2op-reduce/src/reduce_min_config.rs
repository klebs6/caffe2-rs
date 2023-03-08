crate::ix!();

/**
  | Computes the min of the input tensor's
  | element along the provided axes.
  | 
  | The resulted tensor has the same rank
  | as the input if keepdims equal True.
  | 
  | If keepdims equal false, then the resulted
  | tensor have the reduced dimension pruned.
  |
  */
register_cpu_operator!{ReduceMin,
    ReduceOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MinReducer<CPUContext>>
}

num_inputs!{ReduceMin, 1}

num_outputs!{ReduceMin, 1}

inputs!{ReduceMin, 
    0 => ("data", "An input tensor.")
}

outputs!{ReduceMin, 
    0 => ("reduced", "Reduced output tensor.")
}

args!{ReduceMin, 
    0 => ("axes", "A list of integers, along which to reduce."),
    1 => ("keepdims", "Keep the reduced dimension(s) or not, default True keeps the reduced dimension(s).")
}

///------------------------
register_cpu_operator!{ReduceMinGradient,
    ReduceGradientOp<
        TensorTypes<std::int32_t, std::int64_t, float, double>,
        CPUContext,
        MinReducer<CPUContext>>
}

num_inputs!{ReduceMinGradient, 3}

num_outputs!{ReduceMinGradient, 1}
