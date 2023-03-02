crate::ix!();

///------------------------------------------

impl SquaredL2DistanceOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto& Y = Input(1);

      CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
      for (int i = 0; i < X.dim(); ++i) {
        CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
      }
      int N = X.dim() > 0 ? X.dim32(0) : 1;
      auto* distance = Output(0, {N}, at::dtype<float>());
      int D = N > 0 ? X.numel() / N : 0;
      float* distance_data = distance->template mutable_data<float>();
      const float* X_data = X.data<float>();
      const float* Y_data = Y.data<float>();
      for (int i = 0; i < N; ++i) {
        float Xscale, Yscale, cross;
        math::Dot<float, CPUContext>(
            D, X_data + i * D, X_data + i * D, &Xscale, &context_);
        math::Dot<float, CPUContext>(
            D, Y_data + i * D, Y_data + i * D, &Yscale, &context_);
        math::Dot<float, CPUContext>(
            D, X_data + i * D, Y_data + i * D, &cross, &context_);
        distance_data[i] = (Xscale + Yscale) * 0.5 - cross;
      }
      return true;
        */
    }
}

