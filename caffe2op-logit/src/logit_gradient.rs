crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LogitGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    eps:     f32,
    phantom: PhantomData<T>,
}

num_inputs!{LogitGradient, 2}

num_outputs!{LogitGradient, 1}

inputs!{LogitGradient, 
    0 => ("X", "input float tensor"),
    1 => ("dY", "input float tensor")
}

outputs!{LogitGradient, 
    0 => ("dX", "output float tensor")
}

args!{LogitGradient, 
    0 => ("eps", "small positive epsilon value, the default is 1e-6.")
}


impl<T,Context> LogitGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            eps_(this->template GetSingleArgument<float>("eps", 1e-6f))
        */
    }
}
