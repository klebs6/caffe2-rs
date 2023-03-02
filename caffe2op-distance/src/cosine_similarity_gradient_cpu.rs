crate::ix!();

///-----------------------
impl CosineSimilarityGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(X_IN);
      auto& Y = Input(Y_IN);
      auto& dCos = Input(DER_COS_IN);

      const int N = X.dim() > 0 ? X.dim32(0) : 1;
      const int D = X.size_from_dim(1);
      CAFFE_ENFORCE(X.dim() == Y.dim());
      for (int i = 0; i < X.dim(); ++i) {
        CAFFE_ENFORCE(X.dim32(i) == Y.dim32(i));
      }
      CAFFE_ENFORCE(dCos.dim() == 1);
      CAFFE_ENFORCE(dCos.dim32(0) == N);
      auto* dX = Output(DER_X_OUT, X.sizes(), at::dtype<float>());
      auto* dY = Output(DER_Y_OUT, Y.sizes(), at::dtype<float>());

      const auto* X_data = X.template data<float>();
      const auto* Y_data = Y.template data<float>();
      const auto* dCos_data = dCos.template data<float>();
      auto* dX_data = dX->template mutable_data<float>();
      auto* dY_data = dY->template mutable_data<float>();
      float XN, YN, XY;
      const float kEps = 1e-12f;
      for (int i = 0; i < N; ++i) { // TODO: multithreading
        auto offset = i * D;

        // TODO: cache these result from the forward pass
        // ||x||
        math::Dot<float, CPUContext>(
            D, X_data + offset, X_data + offset, &XN, &context_);
        XN = std::sqrt(std::max(XN, kEps));
        // ||y||
        math::Dot<float, CPUContext>(
            D, Y_data + offset, Y_data + offset, &YN, &context_);
        YN = std::sqrt(std::max(YN, kEps));
        // ||x|| * || y ||
        float XYN = XN * YN;
        // x^Ty
        math::Dot<float, CPUContext>(
            D, X_data + offset, Y_data + offset, &XY, &context_);

        math::Scale<float, float, CPUContext>(
            D, dCos_data[i] / XYN, Y_data + offset, dX_data + offset, &context_);
        math::Axpy(
            D,
            -dCos_data[i] * XY / (XN * XN * XYN),
            X_data + offset,
            dX_data + offset,
            &context_);

        math::Scale<float, float, CPUContext>(
            D, dCos_data[i] / XYN, X_data + offset, dY_data + offset, &context_);
        math::Axpy(
            D,
            -dCos_data[i] * XY / (YN * YN * XYN),
            Y_data + offset,
            dY_data + offset,
            &context_);
      }

      return true;
        */
    }
}
