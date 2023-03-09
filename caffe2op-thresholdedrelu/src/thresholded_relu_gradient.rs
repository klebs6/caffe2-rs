crate::ix!();

/**
  | ThresholdedReluGradient takes both
  | Y and dY and uses this to update dX according
  | to the chain rule and derivatives of
  | the rectified linear function.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ThresholdedReluGradientOp<T,Context> {

    storage: OperatorStorage,
    context: Context,
    alpha:   T,

    /*
      | Input: Y, dY,
      | 
      | output: dX
      |
      */
}

num_inputs!{ThresholdedReluGradient, 2}

num_outputs!{ThresholdedReluGradient, 1}

allow_inplace!{ThresholdedReluGradient, vec![(1, 0)]}

impl<T,Context> ThresholdedReluGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...) 

        alpha_ = this->template GetSingleArgument<T>("alpha", 1.0);
        */
    }
}
