crate::ix!();

/**
  | HardSigmoidGradient takes both Y and
  | dY as well as an argument alpha and uses
  | this to update dX according to the chain
  | rule and derivatives of the hard sigmoid
  | function.
  |
  */
pub struct HardSigmoidGradientFunctor<Context> {

    alpha: f32,

    /**
      | Input: Y, dY
      | 
      | Output: dX
      |
      */
    phantom: PhantomData<Context>,
}

num_inputs!{HardSigmoidGradient, 2}

num_outputs!{HardSigmoidGradient, 1}

allow_inplace!{HardSigmoidGradient, vec![(1, 0)]}

impl<Context> HardSigmoidGradientFunctor<Context> {
    
    pub fn new(op: &mut OperatorStorage) -> Self {
        todo!();
        /*
            : alpha(op.GetSingleArgument<float>("alpha", 0.2f))
        */
    }
}
