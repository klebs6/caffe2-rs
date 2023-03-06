crate::ix!();

impl LambdaRankNdcgOp<f32, CPUContext> {
    
    #[inline] pub fn resize_inv_log_itensor(&mut self, size: i32)  {
        
        todo!();
        /*
          int old_size = inv_log_i_.numel();
          int new_size = std::max(old_size, 1);
          while (new_size < size) {
            new_size <<= 1;
          }
          if (new_size != old_size) {
            ReinitializeTensor(&inv_log_i_, {new_size}, at::dtype<float>().device(CPU));
            auto* data = inv_log_i_.template mutable_data<float>();
            EigenVectorArrayMap<float> vec(data, inv_log_i_.numel());
            const float log2f_ = std::log(2.f);
            vec = log2f_ *
                (Eigen::ArrayXf::LinSpaced(new_size, 2, 1 + new_size).log().inverse());
          }
          return;
        */
    }
    
    #[inline] pub fn compute_discounts(&mut self, idx: *mut i32, n: i32)  {
        
        todo!();
        /*
          ReinitializeTensor(&discount_, {N}, at::dtype<float>().device(CPU));
          auto* discount_data = discount_.template mutable_data<float>();
          auto* inv_log_i_data = inv_log_i_.template mutable_data<float>();
          for (int i = 0; i < N; i++) {
            discount_data[idx[i]] = inv_log_i_data[i];
          }
          return;
        */
    }
    
    #[inline] pub fn lambda_rank_ndcg_session(&mut self, 
        start_index: i32,
        end_index:   i32,
        y:           &Tensor,
        r:           &Tensor,
        dy:          *mut *mut Tensor) -> f32 {
        
        todo!();
        /*
          CAFFE_ENFORCE(start_index >= 0);
          CAFFE_ENFORCE(start_index < y.numel());
          const auto* y_data = y.template data<float>();
          const auto* r_data = r.template data<float>();

          int N = end_index - start_index + 1;

          ConstEigenVectorArrayMap<float> y_vec(&y_data[start_index], N);
          ConstEigenVectorArrayMap<float> r_vec(&r_data[start_index], N);

          if (N <= 0) {
            return 0;
          }

          ReinitializeTensor(&ideal_idx_, {N}, at::dtype<int>().device(CPU));
          ReinitializeTensor(&rank_idx_, {N}, at::dtype<int>().device(CPU));
          auto* rank_idx_data = rank_idx_.template mutable_data<int>();
          auto* ideal_idx_data = ideal_idx_.template mutable_data<int>();

          // current ranked list is obtained by sorting by current score
          arg_sort(&y_data[start_index], rank_idx_data, N, true);
          // ideal ranked list is same as sorting by label
          arg_sort(&r_data[start_index], ideal_idx_data, N, true);

          auto* dy_data = (*dy)->template mutable_data<float>();
          EigenVectorArrayMap<float> dy_vec(&dy_data[start_index], N);
          float loss = 0;
          dy_vec = 0;
          // in case that all docs in a session have zero ratings, no op
          if (r_vec.abs().sum() < 1e-6) {
            return 0;
          }

          const double log2f_ = std::log(2.f);
          ReinitializeTensor(&gain_, {N}, at::dtype<float>().device(CPU));
          auto* gain_data = gain_.template mutable_data<float>();
          EigenVectorArrayMap<float> gain_vec(gain_data, gain_.numel());

          if (use_ndcg_as_loss_ && !use_exp_gain_) {
            gain_vec = r_vec;
          } else {
            // Gain vector = 2^rel = exp{rel * log(2)}
            gain_vec = (r_vec * log2f_).exp();
          }
          ResizeInvLogITensor(N);
          ComputeDiscounts(ideal_idx_data, N);
          auto* ideal_discount_data = discount_.template mutable_data<float>();
          EigenVectorArrayMap<float> ideal_discount_vec(
              ideal_discount_data, discount_.numel());
          // ideal dcg = \sum gain_i * ideal_discount_i
          double idcg = (gain_vec * ideal_discount_vec).sum();

          ComputeDiscounts(rank_idx_data, N);
          auto* discount_data = discount_.template mutable_data<float>();
          EigenVectorArrayMap<float> discount_vec(discount_data, discount_.numel());
          // similar to ideal but replace with actual discounts
          double dcg = (gain_vec * discount_vec).sum();

          ReinitializeTensor(&lambda_, {N * N}, at::dtype<float>().device(CPU));
          auto* lambda_data = lambda_.template mutable_data<float>();
          EigenArrayMap<float> lambda_mat(lambda_data, N, N);
          // computes lambda weight (i, j) = abs(gain_dff * discount_diff)
          lambda_mat =
              (PAIRWISE_DIFF(discount_vec, N) * PAIRWISE_DIFF(gain_vec, N)).abs();

          // dy_i =
          //    \sum_j lambda_{i, j} -sign(i > j) * sigm( -sign(i > j)*(yi - yj) )
          //                         |++ gradient of rank loss between i & j  ++|
          dy_vec =
              -(lambda_mat * CWISE_SIGN(PAIRWISE_DIFF(r_vec, N)) *
                CWISE_SIGM(
                    -CWISE_SIGN(PAIRWISE_DIFF(r_vec, N)) * PAIRWISE_DIFF(y_vec, N)))
                   .rowwise()
                   .sum();
          if (use_ndcg_as_loss_) {
            // DCG loss function
            loss = (idcg - dcg);
          } else {
            loss = -(lambda_mat *
                     CWISE_LOG_SIGM(
                         CWISE_SIGN(PAIRWISE_DIFF(r_vec, N)) * PAIRWISE_DIFF(y_vec, N),
                         100))
                        .sum();
          }

          // if use_idcg_normalization_ is true, the loss function is normalized by idcg
          // (e.g. NDCG), else un-normalized loss function (e.g. DCG)
          // Note that normalization is mathematically correct if idcg is guaranteed to
          // be positive!
          if (use_idcg_normalization_) {
            dy_vec /= std::max(idcg, 1e-5);
            loss /= std::max(idcg, 1e-5);
          }
          return loss;
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            auto& y = Input(PRED);
      auto& r = Input(REL);
      auto& sid = Input(SESSION_LENS);

      auto* dy = Output(DPRED);

      const auto* session_lengths = sid.template data<int>();
      CAFFE_ENFORCE(y.dim() == 1);
      CAFFE_ENFORCE(y.numel() == r.numel());
      dy->Resize(y.numel());
      auto* loss = Output(LOSS, {sid.numel()}, at::dtype<float>());
      auto loss_vec = loss->template mutable_data<float>();
      int start_id = 0;
      for (int i = 0; i < sid.numel(); i++) {
        loss_vec[i] = LambdaRankNdcgSession(
            start_id, session_lengths[i] + start_id - 1, y, r, &dy);
        start_id += session_lengths[i];
      }

      return true;
        */
    }
}
