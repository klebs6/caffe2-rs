crate::ix!();

/**
  | This is mostly used just as a debugging
  | purpose stuff: it fills a tensor sequentially
  | with values 0, 1, 2..., which can then be used
  | to check e.g. reshape operations by allowing
  | one to read the indices more easily.
  */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct RangeFillOp<T, Context> {

    base:    FillerOp<Context>,
    phantom: PhantomData<T>,
}

num_inputs!{RangeFill, (0,1)}

num_outputs!{RangeFill, 1}

allow_inplace!{RangeFill, vec![(0, 0)]}

tensor_inference_function!{RangeFill, FillerTensorInference }

impl<T, Context> RangeFillOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : FillerOp<Context>(std::forward<Args>(args)...)
        */
    }
}

impl RangeFillOp<f32, CPUContext> {

    #[inline] pub fn fill(&mut self, output: *mut Tensor) -> bool {
        
        todo!();
        /*
            float* data = output->template mutable_data<float>();
      for (int i = 0; i < output->numel(); ++i) {
        data[i] = i;
      }
      return true;
        */
    }
}
