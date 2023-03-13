crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LeakyReluGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,

    alpha: T,
}

num_inputs!{LeakyReluGradient, 2}

num_outputs!{LeakyReluGradient, 1}

args!{LeakyReluGradient, 
    0 => ("alpha", "Coefficient of leakage")
}

identical_type_and_shape_of_input!{LeakyReluGradient, 1}

allow_inplace!{LeakyReluGradient, vec![(1, 0)]}

inherit_onnx_schema!{LeakyReluGradient}

register_cpu_operator!{
    LeakyReluGradient,
    LeakyReluGradientOp<f32, CPUContext>
}

impl<T, Context> LeakyReluGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...), alpha_(0.01) 

        if (HasArgument("alpha")) {
          alpha_ = static_cast<T>(
              this->template GetSingleArgument<float>("alpha", 0.01));
        }
        */
    }
}

