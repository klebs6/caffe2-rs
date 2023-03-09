crate::ix!();

/**
  | TODO: Complete after verifying utility
  | of TT-layer's forward pass.
  |
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TTLinearGradientOp<T,Context,Engine> {

    storage:         OperatorStorage,
    context:         Context,

    bias_multiplier: Tensor, // default = {Context::GetDeviceType()};
    phantom:         PhantomData<T>,
    phantomE:        PhantomData<Engine>,
}

impl<T,Context,Engine> TTLinearGradientOp<T,Context,Engine> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            return false;
        */
    }
}

register_cpu_operator!{TT, TTLinearOp<float, CPUContext>}

register_cpu_operator!{TTLinearGradient, TTLinearGradientOp<float, CPUContext>}

gradient_not_implemented_yet!{TT}
