crate::ix!();

/**
  | EluGradient takes both Y and dY and uses
  | this to update dX according to the chain
  | rule and derivatives of the rectified
  | linear function.
  |
  */
pub struct EluGradientFunctor<Context> {
    alpha: f32,

    /**
      | Input: Y, dY,
      | 
      | output: dX
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{EluGradient, 2}

num_outputs!{EluGradient, 1}

allow_inplace!{EluGradient, vec![(1, 0)]}

register_cpu_gradient_operator!{
    EluGradient,
    BinaryElementwiseWithArgsOp<
        TensorTypes<f32>,
        CPUContext,
        EluGradientFunctor<CPUContext>>
}

impl<Context> EluGradientFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : alpha(op.GetSingleArgument<float>("alpha", 1.0f))
        */
    }
}
