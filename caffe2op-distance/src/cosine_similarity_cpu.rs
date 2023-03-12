crate::ix!();

///-----------------------
impl CosineSimilarityOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(X_IN);
      auto& Y = Input(Y_IN);

      CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
      for (int i = 0; i < X.dim(); ++i) {
        CAFFE_ENFORCE_EQ(X.dim32(i), Y.dim32(i));
      }
      const int N = X.dim() > 0 ? X.dim32(0) : 1;
      const int D = X.size_from_dim(1);
      auto* result = Output(COS_OUT, {N}, at::dtype<float>());
      float* result_data = result->template mutable_data<float>();
      const float* X_data = X.data<float>();
      const float* Y_data = Y.data<float>();
      float X2, Y2;
      const float kEps = 1e-12f;
      for (int i = 0; i < N; ++i) { // TODO: multithreading
        auto offset = i * D;
        math::Dot<float, CPUContext>(
            D, X_data + offset, X_data + offset, &X2, &context_);
        math::Dot<float, CPUContext>(
            D, Y_data + offset, Y_data + offset, &Y2, &context_);
        math::Dot<float, CPUContext>(
            D, X_data + offset, Y_data + offset, result_data + i, &context_);
        result_data[i] /= std::sqrt(std::max(X2, kEps) * std::max(Y2, kEps));
      }
      return true;
        */
    }
}
