crate::ix!();

impl<T,Context> SpatialSoftmaxWithLossGradientOp<T, Context> {

    #[inline] pub fn run_on_device_f32_cpu_context(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Logits
      auto& T = Input(1); // Labels / targets
      // Input(2) is weights if given
      auto& P = Input(InputSize() - 2); // Probabilities from softmax
      auto& d_avg_loss = Input(InputSize() - 1); // Gradient w.r.t. avg loss

      const float* weights = (InputSize() > 4 ? Input(2).data<float>() : nullptr);
      int N, D;
      N = X.dim32(0);
      D = X.dim32(1);
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      CAFFE_ENFORCE_EQ(T.dim32(0), N);
      CAFFE_ENFORCE_EQ(X.dim(), 4);
      CAFFE_ENFORCE_EQ(T.dim(), 3);

      int H = X.dim32(2);
      int W = X.dim32(3);

      const float* Pdata = P.data<float>();
      float* dX_data = dX->template mutable_data<float>();
      const int* label_data = T.data<int>();

      // Copy softmax probabilities into dX. All but the neuron
      // corresponding to the correct label has gradient equaling e(x_j)
      // which is the probability under softmax.
      context_.CopyFromCPU<float>(P.numel(), Pdata, dX_data);

      float total_weight = 0.0f;
      for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
          for (int i = 0; i < N; ++i) {
            int label_idx = i * H * W + y * W + x;
            int label = label_data[label_idx];

            if (label != DONT_CARE) {
              int idx = i * (H * W * D) + label * (H * W) + y * W + x;

              dX_data[idx] = (dX_data[idx] - 1.0);

              if (weights != nullptr) {
                float weight = weights[label_idx];
                for (int c = 0; c < D; ++c) {
                  int k = i * (H * W * D) + c * (H * W) + y * W + x;
                  dX_data[k] *= weight;
                }
                total_weight += weight;
              } else {
                total_weight += 1.0;
              }
            } else {
              // Set gradient to zero for coordinates where we have dont care
              for (int c = 0; c < D; ++c) {
                int idx = i * (H * W * D) + c * (H * W) + y * W + x;
                dX_data[idx] = 0;
              }
            }
          }
        }
      }

      if (total_weight > 0) {
        math::Scale<float, float, CPUContext>(
            dX->numel(),
            scale_ / total_weight,
            dX->data<float>(),
            dX_data,
            &context_);
      }
      math::Scale<float, float, CPUContext>(
          dX->numel(),
          d_avg_loss.data<float>(),
          dX->data<float>(),
          dX->template mutable_data<float>(),
          &context_);
      return true;
        */
    }
}
