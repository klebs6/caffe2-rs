crate::ix!();

/**
  | Apply the Sigmoid function element-wise
  | to the input tensor. This is often used
  | as a non-linear activation function
  | in a neural network. The sigmoid function
  | is defined as:
  | 
  | $$Sigmoid(x) = \frac{1}{1+\exp(-x)}$$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/sigmoid_op.cc
  |
  */
pub struct SigmoidFunctor<Context> {
    
    // Input: X, output: Y
    phantom: PhantomData<Context>,
}

num_inputs!{Sigmoid, 1}

num_outputs!{Sigmoid, 1}

inputs!{Sigmoid, 
    0 => ("X", "*(type: Tensor`<float>`)* Input tensor.")
}

outputs!{Sigmoid, 
    0 => ("Y", "*(type: Tensor`<float>`)* Output tensor.")
}

identical_type_and_shape!{Sigmoid}

allow_inplace!{Sigmoid, vec![(0, 0)]}

inherit_onnx_schema!{Sigmoid}
