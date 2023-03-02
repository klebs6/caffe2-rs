crate::ix!();

///------------------------------------------

impl L1DistanceOp<f32, CPUContext> {

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

      const float* X_data = X.data<float>();
      const float* Y_data = Y.data<float>();

      for (int i = 0; i < N; ++i) {
        (distance->template mutable_data<float>())[i] =
            (ConstEigenVectorMap<float>(X_data + i * D, D).array() -
             ConstEigenVectorMap<float>(Y_data + i * D, D).array())
                .abs()
                .sum();
      }
      return true;
        */
    }
}
