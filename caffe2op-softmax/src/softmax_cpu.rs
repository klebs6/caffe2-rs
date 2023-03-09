crate::ix!();

/// Implementation for the CPU context.
impl SoftmaxOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            const auto& X = Input(0);
      const int canonical_axis = X.canonical_axis_index(axis_);
      const int N = X.size_to_dim(canonical_axis);
      const int D = X.size_from_dim(canonical_axis);
      auto* Y = Output(0, X.sizes(), at::dtype<float>());
      const float* X_data = X.data<float>();
      float* Y_data = Y->mutable_data<float>();
      if (N == 0 || D == 0) {
        return true;
      }
      if (!scale_.defined()) {
        scale_ = caffe2::empty({N}, at::dtype<float>().device(CPU));
      } else if (scale_.numel() != N) {
        scale_.Resize(N);
      }
      softmax_utils::SoftmaxCPU<float>(
          N, D, false, X_data, Y_data, scale_.mutable_data<float>(), &context_);
      return true;
        */
    }
}

