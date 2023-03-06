crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BernoulliJSDGradientOp<T, Context> {
    storage: OperatorStorage,
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{BernoulliJSDGradient, 3}

num_outputs!{BernoulliJSDGradient, 1}

register_gradient!{
    BernoulliJSD, 
    GetBernoulliJSDGradient
}

register_cpu_operator!{
    BernoulliJSDGradient, 
    BernoulliJSDGradientOp<f32, CPUContext>
}
