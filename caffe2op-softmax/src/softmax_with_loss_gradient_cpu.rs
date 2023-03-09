crate::ix!();

impl SoftmaxWithLossGradientOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Logits
      auto& T = Input(1); // Labels / targets
      // Input(2) is weights if given
      auto& P = Input(InputSize() - 2); // Probabilities from softmax
      auto& d_avg_loss = Input(InputSize() - 1); // Gradient w.r.t. avg loss

      const float* weights = (InputSize() > 4 ? Input(2).data<float>() : nullptr);

      const auto canonical_axis = X.canonical_axis_index(axis_);
      int N, D;
      N = X.size_to_dim(canonical_axis); // batch size
      D = X.size_from_dim(canonical_axis);
      auto* dX = Output(0, X.sizes(), at::dtype<float>());
      float avg_denominator;

      if (label_prob_mode_) {
        CAFFE_ENFORCE_GE(T.dim(), 2);
        CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
        CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), D);
      } else {
        if (T.dim() == canonical_axis) {
          CAFFE_ENFORCE_EQ(T.numel(), N);
        } else {
          CAFFE_ENFORCE_EQ(T.size_to_dim(canonical_axis), N);
          CAFFE_ENFORCE_EQ(T.size_from_dim(canonical_axis), 1);
        }
      }

      const float* Pdata = P.data<float>();
      float* dX_data = dX->template mutable_data<float>();

      // Copy softmax probabilities into dX. All but the neuron
      // corresponding to the correct label has gradient equaling e(x_j)
      // which is the probability under softmax.
      context_.CopyFromCPU<float>(P.numel(), Pdata, dX_data);

      // Compute gradient for the matching labels.
      float total_weight = 0.0f;
      if (!label_prob_mode_) {
        const int* label_data = T.data<int>();

        if (weights) {
          for (int i = 0; i < N; ++i) {
            int idx = i * D + label_data[i];
            float weight = weights[i];
            dX_data[idx] = Pdata[idx] - 1.0;
            for (int d = 0; d < D; d++) {
              int k = i * D + d;
              dX_data[k] *= weight;
            }

            total_weight += weight;
          }
        } else {
          for (int i = 0; i < N; ++i) {
            int idx = i * D + label_data[i];
            dX_data[idx] = Pdata[idx] - 1.0f;
          }
          total_weight = N;
        }
      } else {
        const float* label_data = T.data<float>();

        if (weights) {
          for (int i = 0; i < N; ++i) {
            float weight = weights[i];
            for (int j = 0; j < D; ++j) {
              int idx = i * D + j;
              dX_data[idx] = (Pdata[idx] - label_data[idx]) * weight;
            }
            total_weight += weight;
          }
        } else {
          for (int i = 0; i < N; ++i) {
            for (int j = 0; j < D; ++j) {
              int idx = i * D + j;
              dX_data[idx] = Pdata[idx] - label_data[idx];
            }
          }
          total_weight = N;
        }
      }

      // Scale by d_avg_loss / N
      if (total_weight > 0) {
        if (average_by_batch_size_) {
          avg_denominator = N;
        } else {
          avg_denominator = total_weight;
        }
        math::Scale<float, float, CPUContext>(
            dX->numel(),
            scale_ / avg_denominator * d_avg_loss.data<float>()[0],
            dX->data<float>(),
            dX_data,
            &context_);
      }
      return true;
        */
    }
}
