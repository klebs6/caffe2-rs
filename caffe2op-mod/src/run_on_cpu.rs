crate::ix!();

impl ModOp<CPUContext> {
    
    #[inline] pub fn do_run_with_type<T>(&mut self) -> bool {
        todo!();
        /*
            auto& data = Input(DATA);
      auto N = data.numel();
      const auto* data_ptr = data.template data<T>();

      auto* output = Output(0, Input(DATA).sizes(), at::dtype<T>());
      auto* output_ptr = output->template mutable_data<T>();

      for (auto i = 0; i < N; i++) {
        output_ptr[i] = data_ptr[i] % divisor_;
        if (output_ptr[i] && sign_follow_divisor_ &&
            ((output_ptr[i] > 0) != (divisor_ > 0))) {
          output_ptr[i] += divisor_;
        }
      }
      return true;
        */
    }
}
