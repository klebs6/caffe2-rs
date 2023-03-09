crate::ix!();

impl<Context> WeightedMultiSamplingOp<Context> {
    
    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            num_samples_( this->template GetSingleArgument<int64_t>("num_samples", 0)) 

        CAFFE_ENFORCE_GE(num_samples_, 0);
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& weight = Input(0);
      CAFFE_ENFORCE_EQ(weight.dim(), 1, "Input should be 1-D vector");
      auto dims = weight.sizes().vec();
      size_t data_size = weight.dim32(0);

      std::vector<int64_t> indices_sizes;
      auto num_samples = num_samples_;
      if (InputSize() == 2) {
        CAFFE_ENFORCE(
            !OperatorStorage::HasArgument("num_samples"),
            "New shape is specified by the input blob, do not pass in "
            "the argument `num_samples`.");
        num_samples = Input(1).numel();
        indices_sizes = Input(1).sizes().vec();
      } else {
        indices_sizes = {num_samples};
      }

      auto* indices = Output(0, indices_sizes, at::dtype<int>());
      int* indices_data = indices->template mutable_data<int>();
      if (data_size == 0) {
        indices->Resize(0);
        return true;
      }

      const float* weight_data = weight.template data<float>();

      for (int i = 0; i < num_samples; ++i) {
        float r;
        math::RandUniform<float, Context>(
            1, 0.0f, weight_data[data_size - 1], &r, &context_);
        auto lb = std::lower_bound(weight_data, weight_data + data_size, r);
        CAFFE_ENFORCE(
            lb != weight_data + data_size, "Cannot find ", r, " in input CDF.");
        indices_data[i] = static_cast<int>(lb - weight_data);
      }
      return true;
        */
    }
}
