crate::ix!();

///------------------------------------------

impl L1DistanceGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto& Y = Input(1);
      auto& dDistance = Input(2);

      CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
      for (int i = 0; i < X.dim(); ++i) {
        CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
      }
      int N = X.dim() > 0 ? X.dim32(0) : 1;
      int D = N > 0 ? X.numel() / N : 0;
      CAFFE_ENFORCE(X.dim() == Y.dim());
      for (int i = 0; i < X.dim(); ++i) {
        CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
      }
      CAFFE_ENFORCE(dDistance.dim() == 1);
      CAFFE_ENFORCE(dDistance.dim32(0) == N);
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      auto* dY = Output(1, Y.sizes(), at::dtype<float>());

      for (int i = 0; i < N; ++i) {
        auto offset = i * D;
        for (int j = 0; j < D; ++j) {
          const float temp =
              (X.data<float>())[offset + j] - (Y.data<float>())[offset + j];
          const float kEps = 1e-12f;
          if (temp < -kEps) {
            dX->template mutable_data<float>()[offset + j] =
                -(dDistance.data<float>())[i];
            dY->template mutable_data<float>()[offset + j] =
                (dDistance.data<float>())[i];
          } else if (temp > kEps) {
            dX->template mutable_data<float>()[offset + j] =
                (dDistance.data<float>())[i];
            dY->template mutable_data<float>()[offset + j] =
                -(dDistance.data<float>())[i];
          } else {
            dX->template mutable_data<float>()[offset + j] = 0;
            dY->template mutable_data<float>()[offset + j] = 0;
          }
        }
      }
      return true;
        */
    }
}
