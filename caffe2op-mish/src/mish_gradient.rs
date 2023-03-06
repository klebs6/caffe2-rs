crate::ix!();

/**
  | MishGradient takes X, Y and dY and uses
  | this to update dX according to the chain
  | rule and derivatives of the Mish function.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MishGradientOp<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_operator!{
    MishGradient, 
    MishGradientOp<CPUContext>
}

/**
  | Input: X, Y, dY,
  | 
  | output: dX
  |
  */
num_inputs!{MishGradient, 3}

num_outputs!{MishGradient, 1}

impl<Context> MishGradientOp<Context> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<TensorTypes<float, double>>::call(this, Input(INPUT));
        */
    }
}

input_tags!{
    MishGradientOp {
        Input,
        Output,
        OutputGrad
    }
}

output_tags!{
    MishGradientOp {
        InputGrad
    }
}
