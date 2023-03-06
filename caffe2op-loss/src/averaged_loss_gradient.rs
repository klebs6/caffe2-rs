crate::ix!();

///----------------------------------
pub struct AveragedLossGradient<T, Context> {
    base: SumElementsGradientOp<T, Context>,
    phantom: PhantomData<T>,
}

num_inputs!{AveragedLossGradient, 2}

num_outputs!{AveragedLossGradient, 1}

impl<T, Context> AveragedLossGradient<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : SumElementsGradientOp<T, Context>(std::forward<Args>(args)..., true)
        */
    }
}

register_cpu_operator!{AveragedLoss, AveragedLoss<f32, CPUContext>}

register_cpu_operator!{AveragedLossGradient, AveragedLossGradient<f32, CPUContext>}
