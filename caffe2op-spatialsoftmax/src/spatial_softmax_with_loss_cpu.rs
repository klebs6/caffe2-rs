crate::ix!();

register_cpu_operator!{
    SpatialSoftmaxWithLoss,
    SpatialSoftmaxWithLossOp<f32, CPUContext>
}

register_cpu_operator!{
    SpatialSoftmaxWithLossGradient,
    SpatialSoftmaxWithLossGradientOp<f32, CPUContext>
}

pub const DONT_CARE: i32 = -1;

impl<T,Context> SpatialSoftmaxWithLossOp<T,Context> {

    #[inline] pub fn run_on_device_f32_cpu_context(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Logits
      auto& T = Input(1); // Labels / targets

      int N, D;
      N = X.dim32(0);
      D = X.dim32(1);
      auto* P =
          Output(0, X.sizes(), at::dtype<float>()); // Probabilities from softmax

      if (!sum_multiplier_.defined()) {
        sum_multiplier_ = caffe2::empty({D}, at::dtype<float>().device(CPU));
        math::Set<float, CPUContext>(
            D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
      } else if (sum_multiplier_.numel() != D) {
        sum_multiplier_.Resize(D);
        math::Set<float, CPUContext>(
            D, 1.f, sum_multiplier_.mutable_data<float>(), &context_);
      }

      float* Pdata = P->template mutable_data<float>();
      const float* weights = (InputSize() > 2 ? Input(2).data<float>() : nullptr);
      CAFFE_ENFORCE_EQ(X.dim(), 4);
      CAFFE_ENFORCE_EQ(T.dim(), 3);
      CAFFE_ENFORCE_EQ(T.dim32(0), N);

      int H = X.dim32(2);
      int W = X.dim32(3);

      const float* Xdata = X.data<float>();

      for (int i = 0; i < N; ++i) {
        for (int y = 0; y < H; ++y) {
          for (int x = 0; x < W; ++x) {
            // Subtract max on each cell for numerical reasons
            float max_val = (-1e20f);
            for (int c = 0; c < D; ++c) {
              // TODO optimize
              int idx = i * (H * W * D) + c * (H * W) + y * W + x;
              max_val = std::max(max_val, Xdata[idx]);
            }

            // Exponentiate
            float expsum = 0.0f;
            for (int c = 0; c < D; ++c) {
              int idx = i * (H * W * D) + c * (H * W) + y * W + x;
              float expx = exp(Xdata[idx] - max_val);
              Pdata[idx] = expx;
              expsum += expx;
            }

            // Normalize
            for (int c = 0; c < D; ++c) {
              int idx = i * (H * W * D) + c * (H * W) + y * W + x;
              Pdata[idx] /= expsum;
            }
          }
        }
      }

      // Compute the avg cross-entropy loss
      auto* avg_loss =
          Output(1, vector<int64_t>(), at::dtype<float>()); // Average loss
      float* avg_loss_data = avg_loss->template mutable_data<float>();
      const int* label_data = T.data<int>();

      float sum_label_xent = 0.0f;
      float total_weight = 0.0;

      for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
          for (int i = 0; i < N; i++) {
            int label_idx = i * H * W + y * W + x;
            int label = label_data[label_idx];
            if (label != DONT_CARE) {
              CAFFE_ENFORCE(
                  label < D && label >= 0,
                  "Label seems incorrect:label value larger than number of classes",
                  label_data[i],
                  " vs ",
                  D);
              int idx = i * (H * W * D) + label * (H * W) + y * W + x;
              float w = weights ? weights[label_idx] : 1.0;
              total_weight += w;
              sum_label_xent += -log(std::max(Pdata[idx], 1e-20f)) * w;
            }
          }
        }
      }
      if (total_weight != 0.0) {
        *avg_loss_data = sum_label_xent / total_weight;
      } else {
        *avg_loss_data = 0.0;
      }
      return true;
        */
    }
}
