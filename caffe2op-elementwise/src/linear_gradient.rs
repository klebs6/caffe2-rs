crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct ElementwiseLinearGradientOp<T, Context, Engine> {

    storage:  OperatorStorage,
    context:  Context,
    axis:     i32,

    phantom:  PhantomData<T>,
    phantomE: PhantomData<Engine>,
}

num_inputs!{ElementwiseLinearGradient, 3}

num_outputs!{ElementwiseLinearGradient, 3}

register_cpu_operator!{
    ElementwiseLinearGradient,
    ElementwiseLinearGradientOp<f32, CPUContext>
}

impl<T,Context,Engine> ElementwiseLinearGradientOp<T, Context, Engine> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            axis_(this->template GetSingleArgument<int>("axis", 1))
        */
    }
}
