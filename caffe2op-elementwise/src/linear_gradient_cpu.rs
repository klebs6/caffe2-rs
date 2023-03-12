crate::ix!();

impl ElementwiseLinearGradientOp<f32, CPUContext, DefaultEngine> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& g_o = Input(0);
      const auto& X = Input(1);
      const auto& a = Input(2);

      const auto canonical_axis = X.canonical_axis_index(axis_);
      const int N = X.size_to_dim(canonical_axis);
      const int D = X.size_from_dim(canonical_axis);

      CAFFE_ENFORCE_EQ(a.dim(), 1, a.dim());
      CAFFE_ENFORCE_EQ(a.size(0), D, a.dim());

      auto* g_X = Output(0, X.sizes(), at::dtype<float>());
      auto* g_a = Output(1, a.sizes(), at::dtype<float>());
      auto* g_b = Output(2, a.sizes(), at::dtype<float>());

      const float* g_o_data = g_o.data<float>();
      const float* X_data = X.data<float>();
      const float* a_data = a.data<float>();
      float* g_X_data = g_X->template mutable_data<float>();
      float* g_a_data = g_a->template mutable_data<float>();
      float* g_b_data = g_b->template mutable_data<float>();

      math::Set<float, CPUContext>(g_a->numel(), 0.f, g_a_data, &context_);
      math::Set<float, CPUContext>(g_b->numel(), 0.f, g_b_data, &context_);

      int p = 0;
      for (int n = 0; n < N; ++n) {
        for (int d = 0; d < D; ++d) {
          g_X_data[p] = g_o_data[p] * a_data[d];
          g_a_data[d] += g_o_data[p] * X_data[p];
          g_b_data[d] += g_o_data[p];
          p++;
        }
      }
      return true;
        */
    }
}

