crate::ix!();

register_cpu_operator!{
    SparseDropoutWithReplacement,
    SparseDropoutWithReplacementOp<CPUContext>
}

impl SparseDropoutWithReplacementOp<CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      CAFFE_ENFORCE_EQ(X.ndim(), 1, "Input tensor should be 1-D");
      const int64_t* Xdata = X.data<int64_t>();
      auto& Lengths = Input(1);
      CAFFE_ENFORCE_EQ(Lengths.ndim(), 1, "Lengths tensor should be 1-D");
      auto* OutputLengths = Output(1, Lengths.size(), at::dtype<int32_t>());
      int32_t const* input_lengths_data = Lengths.template data<int32_t>();
      int32_t* output_lengths_data =
          OutputLengths->template mutable_data<int32_t>();
      // Check that input lengths add up to the length of input data
      int total_input_length = 0;
      for (int i = 0; i < Lengths.numel(); ++i) {
        total_input_length += input_lengths_data[i];
      }
      CAFFE_ENFORCE_EQ(
          total_input_length,
          X.numel(),
          "Inconsistent input data. Number of elements should match total length.");

      at::bernoulli_distribution<double> dist(1. - ratio_);
      auto* gen = context_.RandGenerator();
      int32_t total_output_length = 0;
      vector<bool> selected(Lengths.numel(), true);
      for (int i = 0; i < Lengths.numel(); ++i) {
        if (dist(gen) > 0.5) {
          output_lengths_data[i] = input_lengths_data[i];
        } else {
          // Replace with a single dropout value.  Even if input length is 0.
          output_lengths_data[i] = 1;
          selected[i] = false;
        }
        total_output_length += output_lengths_data[i];
      }

      auto* Y = Output(0, {total_output_length}, at::dtype<int64_t>());
      int64_t* Ydata = Y->template mutable_data<int64_t>();

      int input_index = 0;
      int output_index = 0;
      for (int i = 0; i < Lengths.numel(); ++i) {
        if (selected[i]) {
          // Copy logical elements from input to output
          for (int j = input_index; j < input_index + input_lengths_data[i]; ++j) {
            Ydata[output_index++] = Xdata[j];
          }
        } else {
          Ydata[output_index++] = replacement_value_;
        }
        input_index += input_lengths_data[i];
      }
      return true;
        */
    }
}
