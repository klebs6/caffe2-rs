crate::ix!();

#[USE_OPERATOR_CONTEXT_FUNCTIONS]
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct TopKGradientOp<T,Context> {
    context: Context,
    axis:    i32,
    phantom: PhantomData<T>,
}

num_inputs!{TopKGradient, 3}

num_outputs!{TopKGradient, 1}

register_cpu_operator!{TopKGradient, TopKGradientOp<float, CPUContext>}

impl<T,Context> TopKGradientOp<T,Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "axis", axis_, -1)
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& values = Input(0);
      const auto& indices = Input(1);
      const auto& original_input = Input(2);
      auto* output = Output(0);
      at::IntArrayRef values_dims = values.sizes();
      at::IntArrayRef origin_dims = original_input.sizes();
      CAFFE_ENFORCE_EQ(values_dims.size(), origin_dims.size());
      output->Resize(origin_dims);
      const T* values_data = values.template data<T>();
      const int64_t* indices_data = indices.template data<int64_t>();
      T* output_data = output->template mutable_data<T>();
      if (axis_ == -1) {
        axis_ = values_dims.size() - 1;
      }
      const int k = values_dims[axis_];
      math::Set<T, Context>(output->numel(), T(0), output_data, &context_);
      const int64_t prev_size = std::accumulate(
          values_dims.cbegin(),
          values_dims.cbegin() + axis_,
          int64_t(1),
          std::multiplies<int64_t>());
      const int64_t next_size = std::accumulate(
          values_dims.cbegin() + axis_ + 1,
          values_dims.cend(),
          int64_t(1),
          std::multiplies<int64_t>());
      const int64_t src_offset_stride = k * next_size;
      const int64_t dst_offset_stride = origin_dims[axis_] * next_size;
      int64_t src_offset = 0;
      int64_t dst_offset = 0;
      for (int64_t i = 0; i < prev_size; ++i) {
        for (int64_t j = 0; j < next_size; ++j) {
          SetTopKGradient(
              values_data,
              indices_data,
              k,
              src_offset + j,
              dst_offset + j,
              next_size,
              output_data);
        }
        src_offset += src_offset_stride;
        dst_offset += dst_offset_stride;
      }
      return true;
        */
    }
}
