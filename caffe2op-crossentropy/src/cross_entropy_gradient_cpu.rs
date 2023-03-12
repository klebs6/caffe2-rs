crate::ix!();

impl CrossEntropyGradientOp<f32, CPUContext> {
    
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
          (label.dim() == 1) || (label.dim() == 2 && label.dim32(1) == D));
      CAFFE_ENFORCE_EQ(label.dim32(0), N);
      CAFFE_ENFORCE_EQ(dY.dim(), 1);
      CAFFE_ENFORCE_EQ(dY.dim32(0), N);
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      math::Set<float, CPUContext>(
          dX->numel(), 0.f, dX->template mutable_data<float>(), &context_);
      const float* Xdata = X.data<float>();
      const float* dYdata = dY.data<float>();
      const float* labelData = label.data<float>();
      float* dXdata = dX->template mutable_data<float>();
      EigenArrayMap<float>(dXdata, D, N) =
          (ConstEigenArrayMap<float>(labelData, D, N) /
           ConstEigenArrayMap<float>(Xdata, D, N).cwiseMax(kLOG_THRESHOLD()))
              .rowwise() *
          (-ConstEigenVectorArrayMap<float>(dYdata, N).transpose());
      return true;
        */
    }
}
