crate::ix!();

impl LabelCrossEntropyGradientOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0);
      auto& label = Input(1);
      auto& dY = Input(2);

      int N, D;
      if (X.dim() > 1) {
        N = X.dim32(0);
        D = X.size_from_dim(1);
      } else {
        N = 1;
        D = X.dim32(0);
      }
      CAFFE_ENFORCE(
          (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == 1));
      CAFFE_ENFORCE_EQ(label.dim32(0), N);
      CAFFE_ENFORCE_EQ(dY.dim(), 1);
      CAFFE_ENFORCE_EQ(dY.dim32(0), N);
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      math::Set<float, CPUContext>(
          dX->numel(), 0.f, dX->template mutable_data<float>(), &context_);
      const float* Xdata = X.data<float>();
      const float* dYdata = dY.data<float>();
      const int* labelData = label.data<int>();
      float* dXdata = dX->template mutable_data<float>();
      for (int i = 0; i < N; ++i) {
        dXdata[i * D + labelData[i]] =
            -dYdata[i] / std::max(Xdata[i * D + labelData[i]], kLOG_THRESHOLD());
      }
      return true;
        */
    }
}
