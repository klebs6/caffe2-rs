crate::ix!();

impl<T,Context> TopKOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            OP_SINGLE_ARG(int, "k", k_, -1),
            OP_SINGLE_ARG(int, "axis", axis_, -1)
        */
    }

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& input = Input(0);
      auto* values = Output(0);
      auto* indices = Output(1);
      auto* flatten_indices = OutputSize() > 2 ? Output(2) : nullptr;

      int64_t k = k_;
      if(k == -1 && InputSize() == 2) {
        k = Input(1).template data<int64_t>()[0];
      }
      CAFFE_ENFORCE(k >= 1, "k argument must be >= 1");

      at::IntArrayRef input_dims = input.sizes();
      if (axis_ == -1) {
        axis_ = input_dims.size() - 1;
      }
      CAFFE_ENFORCE_GE(axis_, 0);
      CAFFE_ENFORCE_LT(axis_, input_dims.size());

      std::vector<int64_t> output_dims = input_dims.vec();
      output_dims[axis_] = k;
      values->Resize(output_dims);
      indices->Resize(output_dims);
      if (flatten_indices != nullptr) {
        flatten_indices->Resize(indices->numel());
      }
      const T* input_data = input.template data<T>();
      T* values_data = values->template mutable_data<T>();
      int64_t* indices_data = indices->template mutable_data<int64_t>();
      int64_t* flatten_indices_data = flatten_indices == nullptr
          ? nullptr
          : flatten_indices->template mutable_data<int64_t>();
      // init values as the default value
      math::Set<T, Context>(values->numel(), T(0), values_data, &context_);
      math::Set<int64_t, Context>(
          indices->numel(), int64_t(-1), indices_data, &context_);
      if (flatten_indices_data != nullptr) {
        math::Set<int64_t, Context>(
            flatten_indices->numel(), int64_t(-1), flatten_indices_data, &context_);
      }

      const int64_t prev_size = std::accumulate(
          input_dims.cbegin(),
          input_dims.cbegin() + axis_,
          int64_t(1),
          std::multiplies<int64_t>());
      const int64_t next_size = std::accumulate(
          input_dims.cbegin() + axis_ + 1,
          input_dims.cend(),
          int64_t(1),
          std::multiplies<int64_t>());
      const int64_t src_offset_stride = input_dims[axis_] * next_size;
      const int64_t dst_offset_stride = k * next_size;
      int64_t src_offset = 0;
      int64_t dst_offset = 0;
      for (int64_t i = 0; i < prev_size; ++i) {
        for (int64_t j = 0; j < next_size; ++j) {
          GetTopK(
              input_data,
              input_dims[axis_],
              k,
              src_offset + j,
              dst_offset + j,
              next_size,
              values_data,
              indices_data,
              flatten_indices_data);
        }
        src_offset += src_offset_stride;
        dst_offset += dst_offset_stride;
      }
      return true;
        */
    }
}

