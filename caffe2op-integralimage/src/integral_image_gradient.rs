crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct IntegralImageGradientOp<T, Context> {
    storage:         OperatorStorage,
    context:         Context,
    row_pass_buffer: Tensor,

    // Input: X, dY (aka "gradOutput"); 
    // Output: dX (aka "gradInput")

    phantom: PhantomData<T>,
}

impl<T, Context> IntegralImageGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }
}

num_inputs!{IntegralImageGradient, 2}

num_outputs!{IntegralImageGradient, 1}

register_gradient!{
    IntegralImage, 
    GetIntegralImageGradient
}

register_cpu_operator!{
    IntegralImageGradient, 
    IntegralImageGradientOp<f32, CPUContext>
}
