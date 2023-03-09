crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct FlexibleTopKGradientOp<T, Context> {
    context: Context,
    phantom: PhantomData<T>,
}

num_inputs!{FlexibleTopKGradient, 4}

num_outputs!{FlexibleTopKGradient, 1}

register_cpu_operator!{FlexibleTopKGradient, FlexibleTopKGradientOp<f32, CPUContext>}

impl<T, Context> FlexibleTopKGradientOp<T, Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...)
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        todo!();
        /*
            auto& original_input = Input(0);
          auto& k = Input(1);
          auto& values = Input(2);
          auto& indices = Input(3);

          const int64_t* k_data = k.template data<int64_t>();
          const T* values_data = values.template data<T>();
          const int64_t* indices_data = indices.template data<int64_t>();

          // Resize output tensors to be as orignial_input size and initialized with 0
          CAFFE_ENFORCE_GT(original_input.dim(), 0);
          vector<int64_t> original_dims = original_input.sizes().vec();
          auto* output = Output(0, original_dims, at::dtype<T>());
          T* output_data = output->template mutable_data<T>();
          math::Set<T, Context>(
              output->numel(), static_cast<T>(0), output_data, &context_);

          int64_t index_offset = 0;
          for (int64_t i = 0; i < k.numel(); ++i) {
            // offset of output_data
            int64_t output_offset = i * original_dims.back();
            for (int64_t j = 0; j < k_data[i]; ++j) {
              int64_t index = indices_data[index_offset + j];
              T value = values_data[index_offset + j];
              output_data[output_offset + index] = value;
            }
            index_offset += k_data[i];
          }

          return true;
        */
    }
}
