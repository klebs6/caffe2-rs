crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BooleanMaskOpGradient<Context> {
    storage: OperatorStorage,
    context: Context,
}

register_cpu_gradient_operator!{ 
    BooleanMaskGradient, 
    BooleanMaskOpGradient<CPUContext>
}

register_cpu_operator!{
    BooleanMaskLengths, 
    BooleanMaskLengthsOp<CPUContext>
}

num_inputs!{BooleanMaskGradient, 2}

num_outputs!{BooleanMaskGradient, 1}

no_gradient!{BooleanMaskLengths}

impl<Context> BooleanMaskOpGradient<Context> {
    
    pub fn new(operator_def: &OperatorDef, ws: *mut Workspace) -> Self {
        todo!();
        /*
            : Operator<Context>(operator_def, ws)
        */
    }
    
    /**
      | Calculating the gradient of the Boolean
      | Mask operator requires access to the
      | original mask that's passed in, and
      | the gradient to backpropagate.
      |
      */
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return DispatchHelper<
            TensorTypes<bool, std::int32_t, std::int64_t, float, double>>::
            call(this, Input(1));
        */
    }
}
