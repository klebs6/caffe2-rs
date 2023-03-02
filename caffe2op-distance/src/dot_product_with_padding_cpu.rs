crate::ix!();

impl DotProductWithPaddingOp<f32, CPUContext> {

    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(X_IN);
      auto& Y = Input(Y_IN);

      CAFFE_ENFORCE_EQ(X.dim(), Y.dim());
      CAFFE_ENFORCE_EQ(X.dim32(0), Y.dim32(0));

      int N, D, DX, DY, restD;
      if (X.numel() > 0) {
        N = X.dim() > 0 ? X.dim32(0) : 1;
        DX = X.numel() / N;
        DY = Y.numel() / N;
      } else {
        N = 0;
        DX = 0;
        DY = 0;
      }

      D = std::min(DX, DY);
      restD = std::max(DX, DY) - D;
      auto* result = Output(DOT_OUT, {N}, at::dtype<float>());
      float* result_data = result->template mutable_data<float>();
      const float* X_data = X.data<float>();
      const float* Y_data = Y.data<float>();
      for (int i = 0; i < N; ++i) { // TODO: multithreading
        auto offsetX = i * DX, offsetY = i * DY;
        if (replicate_) {
          // L_ for longer vector and S_ for shorter vector
          const float *L_data, *S_data;
          int DL, DS;
          if (DX > DY) {
            L_data = X_data + offsetX;
            S_data = Y_data + offsetY;
            DL = DX;
            DS = DY;
          } else {
            L_data = Y_data + offsetY;
            S_data = X_data + offsetX;
            DL = DY;
            DS = DX;
          }
          float sum = 0.0;
          float tmp = 0.0;
          for (int j = 0; j < DL / DS; j++) {
            math::Dot<float, CPUContext>(
                DS, L_data + j * DS, S_data, &tmp, &context_);
            sum += tmp;
          }
          *(result_data + i) = sum;
        } else {
          math::Dot<float, CPUContext>(
              D, X_data + offsetX, Y_data + offsetY, result_data + i, &context_);
        }

        if (!replicate_ && DX != DY) {
          const float* rest_data;
          float rest_sum = 0;
          if (DX > DY) {
            rest_data = X_data + offsetX + D;
          } else {
            rest_data = Y_data + offsetY + D;
          }
          math::Sum<float, CPUContext>(restD, rest_data, &rest_sum, &context_);
          result_data[i] += rest_sum * pad_value_;
        }
      }
      return true;
        */
    }
}
