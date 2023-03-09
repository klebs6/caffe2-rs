crate::ix!();

/**
  | SigmoidGradient takes both Y and dY
  | and uses this to update dX according
  | to the chain rule and derivatives of
  | the sigmoid function.
  |
  */
pub struct SigmoidGradientFunctor<Context> {

    /**
      | Input: Y, dY
      | 
      | output: dX
      |
      */
    phantom: PhantomData<Context>,
}

register_cpu_operator!{Sigmoid,
    UnaryElementwiseOp<
        TensorTypes<f32>,
        CPUContext,
        SigmoidFunctor<CPUContext>>}

num_inputs!{SigmoidGradient, 2}

num_outputs!{SigmoidGradient, 1}

identical_type_and_shape_of_input!{SigmoidGradient, 1}

allow_inplace!{SigmoidGradient, vec![(1, 0)]}
