crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct MSRAFillOp<T, Context> {

    base:    FillerOp<Context>,
    phantom: PhantomData<T>,
}

num_inputs!{MSRAFill, (0,1)}

num_outputs!{MSRAFill, 1}

allow_inplace!{MSRAFill, vec![(0, 0)]}

tensor_inference_function!{MSRAFill, FillerTensorInference}

impl<T, Context> MSRAFillOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...)
        */
    }
    
    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            const int fan_out = output->numel() / output->dim32(1);
        T scale = std::sqrt(T(2) / fan_out);
        math::RandGaussian<T, Context>(
            output->numel(),
            0.0,
            scale,
            output->template mutable_data<T>(),
            &context_);
        return true;
        */
    }
}
