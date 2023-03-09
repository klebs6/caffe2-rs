crate::ix!();

pub const DONT_CARE: i32 = -1;

impl SoftmaxWithLossOp<f32, CPUContext> {
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& X = Input(0); // Logits
      auto& T = Input(1); // Labels / targets

      const auto canonical_axis = X.canonical_axis_index(axis_);
      int64_t N, D;
      N = X.size_to_dim(canonical_axis); // batch size
      D = X.size_from_dim(canonical_axis);
      auto* P =
          Output(0, X.sizes(), at::dtype<float>()); // Probabilities from softmax

      float* Pdata = P->template mutable_data<float>();
      const float* weights = (InputSize() > 2 ? Input(2).data<float>() : nullptr);

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

      if (!losses_.defined()) {
        losses_ = caffe2::empty({N}, at::dtype<float>().device(CPU));
      } else if (losses_.numel() != N) {
        losses_.Resize(N);
      }

      softmax_utils::SoftmaxCPU<float>(
          N,
          D,
          !label_prob_mode_,
          X.data<float>(),
          Pdata,
          losses_.mutable_data<float>(),
          &context_);

      // Then compute cross entropy
      float loss_sum = 0.0;
      float weight_sum = 0.0;
      if (!label_prob_mode_) {
        const int* label_data = T.data<int>();
        const float* Xdata = X.data<float>();

        for (int i = 0; i < N; ++i) {
          CAFFE_ENFORCE(
              label_data[i] < D && label_data[i] >= 0,
              "Label seems incorrect: label value larger than number of classes: ",
              label_data[i],
              " vs ",
              D);
          float weight = weights ? weights[i] : 1.0;
          float l = -Pdata[i * D + label_data[i]] * weight;
          loss_sum += l;
          weight_sum += weight;
        }
        math::Exp(N * D, Pdata, Pdata, &context_);
      } else {
        const float* label_data = T.data<float>();

        for (int i = 0; i < N; ++i) {
          float l = 0.0;
          float total_prob = 0.0;
          float weight = weights ? weights[i] : 1.0;
          for (int j = 0; j < D; ++j) {
            CAFFE_ENFORCE(
                label_data[i * D + j] >= 0,
                "Label prob seems incorrect: label prob value must be nonnegative:",
                " ",
                label_data[i * D + j]);
            l += -log(std::max(Pdata[i * D + j], 1e-20f)) * label_data[i * D + j] *
                weight;
            total_prob += label_data[i * D + j];
          }
          loss_sum += l;
          CAFFE_ENFORCE(
              std::abs(total_prob - 1.) < 1e-5f,
              "Label prob seems incorrect: label prob values do not sum to 1.0: ",
              total_prob,
              " vs 1.0 (+/- 1e-5)");
          weight_sum += weight;
        }
      }

      auto* avg_loss =
          Output(1, vector<int64_t>(), at::dtype<float>()); // Average loss

      float* avg_loss_data = avg_loss->template mutable_data<float>();
      if (weight_sum != 0.0) {
        if (average_by_batch_size_) {
          avg_loss_data[0] = loss_sum * scale_ / N;
        } else {
          avg_loss_data[0] = loss_sum * scale_ / weight_sum;
        }
      } else {
        avg_loss_data[0] = 0.0;
      }
      return true;
        */
    }
}
