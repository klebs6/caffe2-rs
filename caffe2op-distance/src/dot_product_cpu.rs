crate::ix!();

impl DotProductOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(X_IN);
      auto& Y = Input(Y_IN);

      CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
      for (int i = 0; i < X.dim(); ++i) {
        CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i), "dimension at ", i);
      }
      int N, D;
      if (X.numel() > 0) {
        N = X.dim() > 0 ? X.dim32(0) : 1;
        D = X.numel() / N;
      } else {
        N = 0;
        D = 0;
      }
      auto* result = Output(DOT_OUT, {N}, at::dtype<float>());
      float* result_data = result->template mutable_data<float>();
      const float* X_data = X.template data<float>();
      const float* Y_data = Y.template data<float>();
      for (int i = 0; i < N; ++i) { // TODO: multithreading
        auto offset = i * D;
        math::Dot<float, CPUContext>(
            D, X_data + offset, Y_data + offset, result_data + i, &context_);
      }
      return true;
        */
    }
}

