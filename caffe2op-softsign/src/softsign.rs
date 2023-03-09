crate::ix!();

/**
  | Softsign takes one input data tensor
  | $X$ and produces one output data $Y,$
  | where the softsign function, $y = \frac{x}{1+
  | |x|}$, is applied to $X$ elementwise.
  | 
  | This operation can be done in an in-place
  | fashion too, by providing the same input
  | and output blobs.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/softsign_op.cc
  |
  */
pub struct SoftsignFunctor<Context> { 

    phantom: PhantomData<Context>,
}

num_inputs!{Softsign, 1}

num_outputs!{Softsign, 1}

inputs!{Softsign, 
    0 => ("input", "Input data blob to be operated on.")
}

outputs!{Softsign, 
    0 => ("output", "Output data blob with same shape as input")
}

identical_type_and_shape!{Softsign}

allow_inplace!{Softsign, vec![(0, 0)]}

inherit_onnx_schema!{Softsign}
