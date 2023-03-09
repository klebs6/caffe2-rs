crate::ix!();

/**
  | Calculates the sine of the given input
  | tensor, element-wise.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sin_op.cc
  |
  */
register_cpu_operator!{Sin,
    UnaryElementwiseOp<TensorTypes<float>, CPUContext, SinFunctor<CPUContext>>}

num_inputs!{Sin, 1}

num_outputs!{Sin, 1}

inputs!{Sin, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Sin, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor calculated as the sine of the input tensor, element-wise.")
}

identical_type_and_shape!{Sin}

