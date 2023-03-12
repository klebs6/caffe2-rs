crate::ix!();

/**
  | Element-wise application of the floor
  | function ($y=floor(x)$) to the input
  | tensor `X`. Output tensor shape is the
  | same as the input tensor. This operator
  | can be used in an in-place fashion by
  | using the same input blob as the output
  | blob.
  | 
  | Github Link:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/floor_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FloorOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{Floor, 1}

num_outputs!{Floor, 1}

inputs!{Floor, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Floor, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

allow_inplace!{Floor, vec![(0, 0)]}

register_cpu_operator!{
    Floor, 
    FloorOp<f32, CPUContext>
}

// TODO: Write gradient for this when needed
gradient_not_implemented_yet!{Floor}
