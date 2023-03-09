crate::ix!();

/**
  | Transpose the input tensor by permuting
  | the axes of the input according to the
  | `axes` argument.
  | 
  | Similar to numpy's [transpose](https://docs.scipy.org/doc/numpy/reference/generated/numpy.transpose.html)
  | function.
  | 
  | For example, when axes=(1, 0, 2), given
  | an input tensor of shape (1, 2, 3), the
  | output shape will be (2, 1, 3).
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/transpose_op.cc
  |
  */
#[USE_DISPATCH_HELPER]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TransposeOp<Context> {
    storage: OperatorStorage,
    context: Context,
    axes:    Vec<i32>,
}

register_cpu_operator!{Transpose, TransposeOp<CPUContext>}

num_inputs!{Transpose, 1}

num_outputs!{Transpose, 1}

inputs!{Transpose, 
    0 => ("X", "*(type: Tensor)* Input tensor.")
}

outputs!{Transpose, 
    0 => ("Y", "*(type: Tensor)* Transposed output.")
}

args!{Transpose, 
    0 => ("axes", "*(type: Tuple(int))* Order to permute axes of input tensor. Reverses the dimensions by default.")
}

inherit_onnx_schema!{Transpose}
