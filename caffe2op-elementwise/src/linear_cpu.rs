crate::ix!();

impl ElementwiseLinearOp<f32, CPUContext, DefaultEngine> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const auto& a = Input(1);
      const auto& b = Input(2);

      const auto canonical_axis = X.canonical_axis_index(axis_);
      const int N = X.size_to_dim(canonical_axis);
      const int D = X.size_from_dim(canonical_axis);

      CAFFE_ENFORCE_EQ(a.dim(), 1, a.dim());
      CAFFE_ENFORCE_EQ(a.size(0), D, a.dim());
      CAFFE_ENFORCE_EQ(b.dim(), 1, b.dim());
      CAFFE_ENFORCE_EQ(b.size(0), D, b.dim());

      auto* Y = Output(0, X.sizes(), at::dtype<float>());

      const float* X_data = X.data<float>();
      const float* a_data = a.data<float>();
      const float* b_data = b.data<float>();
      float* Y_data = Y->template mutable_data<float>();

      int p = 0;
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          Y_data[p] = X_data[p] * a_data[d] + b_data[d];
          p++;
        }
      }
      return true;
        */
    }
}
