crate::ix!();

/**
  | Swish takes one input data (Tensor)
  | and produces one output data (Tensor)
  | where the swish function, y = x / (1 + exp(-x)),
  | is applied to the tensor elementwise.
  |
  */
pub struct SwishFunctor<T, Context> {

    /**
      | Input: X,
      | 
      | output: Y
      |
      */
    phantomA: PhantomData<T>,
    phantomB: PhantomData<Context>,
}

num_inputs!{Swish, 1}

num_outputs!{Swish, 1}

inputs!{Swish, 
    0 => ("X", "1D input tensor")
}

outputs!{Swish, 
    0 => ("Y", "1D output tensor")
}

identical_type_and_shape!{Swish}
