crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct SoftmaxGradientOp<T,Context> {
    storage:         OperatorStorage,
    context:         Context,
    axis:            i32,
    scale:           Tensor,
    sum_multiplier:  Tensor,

    // Input: Y, dY. 
    // Output: dX
    phantom:         PhantomData<T>,
}

num_inputs!{SoftmaxGradient, 2}

num_outputs!{SoftmaxGradient, 1}

impl<T,Context> SoftmaxGradientOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
    
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int>("axis", 1))
        */
    }
}
