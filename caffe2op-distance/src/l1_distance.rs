crate::ix!();

/**
  | Computes the row-wise L1 Distance between
  | the two input tensors $X$ and $Y$, which
  | is defined as
  | 
  | $$L1Distance(\mathbf{x},\mathbf{y})
  | = \sum_{i}\mid x_i - y_i\mid$$
  | 
  | Note, both inputs must either be 1-dimensional
  | or 2-dimensional and both must have
  | the same shape.
  | 
  | The output $Z$ will be 1-dimensional
  | regardless and its length will equal
  | the number of rows in the inputs.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.h
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/distance_op.cc
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct L1DistanceOp<T, Context> {

    storage: OperatorStorage,
    context: Context,

    /**
      | Input: X, Y;
      | 
      | Output: Distance
      |
      */
    phantom: PhantomData<T>,
}

register_cpu_operator!{
    L1Distance, 
    L1DistanceOp<f32, CPUContext>
}

num_inputs!{L1Distance, 2}

num_outputs!{L1Distance, 1}

inputs!{L1Distance, 
    0 => ("X", "First input tensor. (1D or 2D)"),
    1 => ("Y", "Second input tensor. (must have the same shape as $X$)")
}

outputs!{L1Distance, 
    0 => ("Z", "1D output tensor. One value for each row of the inputs.")
}

identical_type_and_shape_of_input_dim!{L1Distance, (0, 0)}

impl<T,Context> L1DistanceOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

