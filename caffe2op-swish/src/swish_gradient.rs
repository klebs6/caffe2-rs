crate::ix!();

/**
  | SwishGradient takes X, Y and dY and uses
  | this to update dX according to the chain
  | rule and derivatives of the swish function.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SwishGradientOp<Context> {

    storage: OperatorStorage,
    context: Context,

    /*
      | Input: X, Y, dY,
      | 
      | output: dX
      |
      */
}

num_inputs!{SwishGradient, 3}

num_outputs!{SwishGradient, 1}

allow_inplace!{SwishGradient, vec![(2, 0)]}

input_tags!{
    SwishGradientOp {
        X,
        Y,
        Dy
    }
}

output_tags!{
    SwishGradientOp {
        Dx
    }
}

impl<Context> SwishGradientOp<Context> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(X));
        */
    }
}
