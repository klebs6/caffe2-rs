crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/batch_norm_kernel.cpp]

pub fn batch_norm_cpu_collect_linear_and_constant_terms<Scalar>(
        alpha:        *mut Scalar,
        beta:         *mut Scalar,
        n_channel:    i64,
        weight:       &Tensor,
        bias:         &Tensor,
        save_mean:    &Tensor,
        save_invstd:  &Tensor,
        running_mean: &Tensor,
        running_var:  &Tensor,
        train:        bool,
        eps:          f64)  {

    todo!();
        /*
            const Scalar* weight_data = weight.defined() ? weight.data_ptr<Scalar>() : nullptr;
      const Scalar* bias_data = bias.defined() ? bias.data_ptr<Scalar>() : nullptr;

      auto save_mean_a = conditional_accessor_1d<Scalar>(save_mean);
      auto save_invstd_a = conditional_accessor_1d<Scalar>(save_invstd);
      auto running_mean_a = conditional_accessor_1d<Scalar>(running_mean);
      auto running_var_a = conditional_accessor_1d<Scalar>(running_var);

      /// Collect the linear and constant terms regarding the input.
      /// output(n, c, h, w)
      ///     = (input(n, c, h, w) - mean(c)) / sqrt(var(c) + eps) * weight(c)
      ///         + bias(c)
      ///     = input(n, c, h, w) * inv_var(c) * weight(c)
      ///         - mean(c) * inv_var(c) * weight(c) + bias(c),
      /// where inv_var(c) = 1 / sqrt(var(c) + eps).
      /// So the linear term, alpha(c) = inv_var(c) * weight(c),
      ///   the constant term beta(c) = bias(c) - mean(c) * inv_var(c) * weight(c)
      /// Note that this is only a good idea if (input_size >> c), in degenerate
      /// cases where image_size == 1 && batch_size == 1, it is slow.
      for (i64 c = 0; c < n_channel; c++) {
        Scalar mean, invstd;
        if (train) {
          mean = save_mean_a[c];
          invstd = save_invstd_a[c];
        } else {
          mean = running_mean_a[c];
          invstd = 1 / sqrt(running_var_a[c] + static_cast<Scalar>(eps));
        }
        Scalar weight_v = weight_data ? weight_data[c] : 1;
        Scalar bias_v = bias_data ? bias_data[c] : 0;
        alpha[c] = invstd * weight_v;
        beta[c] = bias_v - mean * alpha[c];
      }
        */
}

/**
  | A fast path for CPU inference and training
  | forward when all tensors are contiguous.
  |
  */
pub fn batch_norm_cpu_contiguous_impl<Scalar>(
        output:       &mut Tensor,
        input:        &Tensor,
        weight:       &Tensor,
        bias:         &Tensor,
        save_mean:    &Tensor,
        save_invstd:  &Tensor,
        running_mean: &Tensor,
        running_var:  &Tensor,
        train:        bool,
        eps:          f64)  {

    todo!();
        /*
            using Vec = Vectorized<Scalar>;
      i64 n_batch = input.size(0);
      i64 n_channel = input.size(1);
      i64 image_size = input.numel() / n_batch / n_channel;

      Tensor alpha = empty({n_channel}, input.options());
      Tensor beta = empty({n_channel}, input.options());
      Scalar* alpha_data = alpha.data_ptr<Scalar>();
      Scalar* beta_data = beta.data_ptr<Scalar>();

      batch_norm_cpu_collect_linear_and_constant_terms<Scalar>(
         alpha_data, beta_data, n_channel, weight, bias,
         save_mean, save_invstd, running_mean, running_var, train, eps);

      Scalar* output_data = output.data_ptr<Scalar>();
      const Scalar* input_data = input.data_ptr<Scalar>();

      // Apply the linear terms to the input,
      // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
      if (image_size != 1) {
        const i64 loop_size = image_size - (image_size % Vec::size());
        parallel_for(0, n_batch * n_channel, 1, [&](i64 begin, i64 end) {
          i64 n = 0;
          i64 c = 0;
          data_index_init(begin, n, n_batch, c, n_channel);

          for (i64 i = begin; i < end; i++) {
            const Vec alpha_vec(alpha_data[c]);
            const Vec beta_vec(beta_data[c]);
            i64 offset = i * image_size;
            i64 d = 0;
            for (; d < loop_size; d += Vec::size()) {
              Vec data_vec = Vec::loadu(input_data + offset + d);
              Vec output_vec = data_vec * alpha_vec + beta_vec;
              output_vec.store(output_data + offset + d);
            }
            if (image_size - d > 0) {
              Vec data_vec = Vec::loadu(input_data + offset + d, image_size - d);
              Vec output_vec = data_vec * alpha_vec + beta_vec;
              output_vec.store(output_data + offset + d, image_size - d);
            }
            // move on to next index
            data_index_step(n, n_batch, c, n_channel);
          }
        });
      } else {
        // image_size == 1
        const i64 loop_size = n_channel - (n_channel % Vec::size());
        parallel_for(0, n_batch, 1, [&](i64 begin, i64 end) {
          for (i64 n = begin; n < end; n++) {
            i64 offset = n * n_channel;
            i64 d = 0;
            for (; d < loop_size; d += Vec::size()) {
              Vec alpha_vec = Vec::loadu(alpha_data + d);
              Vec beta_vec = Vec::loadu(beta_data + d);
              Vec data_vec = Vec::loadu(input_data + offset + d);
              Vec output_vec = data_vec * alpha_vec + beta_vec;
              output_vec.store(output_data + offset + d);
            }
            if (n_channel - d > 0) {
              Vec alpha_vec = Vec::loadu(alpha_data + d, n_channel - d);
              Vec beta_vec = Vec::loadu(beta_data + d, n_channel - d);
              Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
              Vec output_vec = data_vec * alpha_vec + beta_vec;
              output_vec.store(output_data + offset + d, n_channel - d);
            }
          }
        });
      }
        */
}


pub fn batch_norm_cpu_channels_last_impl<Scalar>(
        output:       &mut Tensor,
        input:        &Tensor,
        weight:       &Tensor,
        bias:         &Tensor,
        save_mean:    &Tensor,
        save_invstd:  &Tensor,
        running_mean: &Tensor,
        runnning_var: &Tensor,
        train:        bool,
        eps:          f64)  {

    todo!();
        /*
            using Vec = Vectorized<Scalar>;
      i64 n_batch = input.size(0);
      i64 n_channel = input.size(1);
      i64 image_size = input.numel() / n_batch / n_channel;

      Tensor alpha = empty({n_channel}, input.options());
      Tensor beta = empty({n_channel}, input.options());
      Scalar* alpha_data = alpha.data_ptr<Scalar>();
      Scalar* beta_data = beta.data_ptr<Scalar>();

      batch_norm_cpu_collect_linear_and_constant_terms<Scalar>(
          alpha_data, beta_data, n_channel, weight, bias,
          save_mean, save_invstd, running_mean, runnning_var, train, eps);

      Scalar* output_data = output.data_ptr<Scalar>();
      const Scalar* input_data = input.data_ptr<Scalar>();

      // Apply the linear terms to the input,
      // output(n, c, h, w) = input(n, c, h, w) * alpha(c) + beta(c)
      const i64 loop_size = n_channel - (n_channel % Vec::size());
      parallel_for(0, n_batch * image_size, 1, [&](i64 begin, i64 end) {
        for (i64 i = begin; i < end; i++) {
          i64 offset = i * n_channel;
          i64 d = 0;
          // vectorize on channel dimension, for normal batch_norm input size,
          // alpha/beta should fit in L1 cache, otherwise consider blocking.
          for (; d < loop_size; d += Vec::size()) {
            Vec alpha_vec = Vec::loadu(alpha_data + d);
            Vec beta_vec = Vec::loadu(beta_data + d);
            Vec data_vec = Vec::loadu(input_data + offset + d);
            Vec output_vec = data_vec * alpha_vec + beta_vec;
            output_vec.store(output_data + offset + d);
          }
          if (n_channel - d > 0) {
            Vec alpha_vec = Vec::loadu(alpha_data + d, n_channel - d);
            Vec beta_vec = Vec::loadu(beta_data + d, n_channel - d);
            Vec data_vec = Vec::loadu(input_data + offset + d, n_channel - d);
            Vec output_vec = data_vec * alpha_vec + beta_vec;
            output_vec.store(output_data + offset + d, n_channel - d);
          }
        }
      });
        */
}

pub fn batch_norm_cpu_collect_stats_contiguous_impl<Scalar>(
        mean:    &mut Tensor,
        var_sum: &mut Tensor,
        input:   &Tensor)  {

    todo!();
        /*
            using accscalar_t = acc_type<Scalar, false>;
      i64 n_batch = input.size(0);
      i64 n_channel = input.size(1);
      i64 image_size = input.numel() / n_batch / n_channel;
      i64 N = input.numel() / n_channel;

      const Scalar* input_data = input.data_ptr<Scalar>();
      Scalar* mean_data = mean.data_ptr<Scalar>();
      Scalar* var_sum_data = var_sum.data_ptr<Scalar>();

      // parallel dim reduce on 'channel'
      parallel_for(0, n_channel, 1, [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          // compute mean per input
          accscalar_t sum = 0;
          for (i64 n = 0; n < n_batch; n++) {
            for (i64 i = 0; i < image_size; i++) {
              auto offset = n * n_channel * image_size + c * image_size + i;
              sum += input_data[offset];
            }
          }
          Scalar mean = sum / N;
          mean_data[c] = mean;

          // compute variance per input
          accscalar_t _var_sum = 0;
          for (i64 n = 0; n < n_batch; n++) {
            for (i64 i = 0; i < image_size; i++) {
              auto offset = n * n_channel * image_size + c * image_size + i;
              auto x = input_data[offset];
              _var_sum += (x - mean) * (x - mean);
            }
          }
          var_sum_data[c] = _var_sum;
        }
      });
        */
}

pub fn batch_norm_cpu_collect_stats_channels_last_impl<Scalar>(
        mean:    &mut Tensor,
        var_sum: &mut Tensor,
        input:   &Tensor)  {

    todo!();
        /*
            using Vec = Vectorized<Scalar>;
      using accscalar_t = acc_type<Scalar, false>;
      i64 n_channel = input.size(1);
      i64 N = input.numel() / n_channel;

      const Scalar* input_data = input.data_ptr<Scalar>();
      Scalar* mean_data = mean.data_ptr<Scalar>();
      Scalar* var_sum_data = var_sum.data_ptr<Scalar>();

      // Typical vertical reduce from shape of {NHW, C} to {C}.
      // Apply two path parallel reduction:
      // First path: allocate an immediate buffer of size {max_threads, C}, parallel along dim0,
      //    {NHW, C} => {max_threads, C}
      //
      // Second path: parallel along dim1 of the immediate buffer,
      //    {max_threads, C} => {C}
      //
      // Normal size of C should fit in L1, otherwise consider blocking on C.
      //
      int num_threads = get_num_threads();
      Tensor buffer = empty({num_threads, n_channel}, input.options()).zero_();
      Scalar* buffer_data = buffer.data_ptr<Scalar>();

      // compute mean per input
      parallel_for(0, N, 1, [&](i64 begin, i64 end) {
        int tid = get_thread_num();
        TORCH_CHECK(tid < num_threads,
                    "expect thread id smaller than ", num_threads, ", got thread id ", tid);
        Scalar* buffer_ptr = buffer_data + tid * n_channel;
        for (i64 i = begin; i < end; i++) {
          const Scalar* x_ptr = input_data + i * n_channel;
          vec::map2<Scalar>(
              [](Vec x, Vec y) { return x + y; },
              buffer_ptr,
              x_ptr,
              buffer_ptr,
              n_channel);
        }
      });

      parallel_for(0, n_channel, 1, [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          accscalar_t sum = 0;
          for (i64 t = 0; t < num_threads; t++) {
            sum += buffer_data[t * n_channel + c];
          }
          Scalar mean = sum / N;
          mean_data[c] = mean;
        }
      });

      // compute variance per input, reuse the immediate buffer
      buffer.zero_();
      parallel_for(0, N, 1, [&](i64 begin, i64 end) {
        int tid = get_thread_num();
        TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
        Scalar* buffer_ptr = buffer_data + tid * n_channel;
        for (i64 i = begin; i < end; i++) {
          const Scalar* x_ptr = input_data + i * n_channel;
          vec::map3<Scalar>(
              [](Vec x, Vec y, Vec mean) { return y + (x - mean) * (x - mean); },
              buffer_ptr,
              x_ptr,
              buffer_ptr,
              mean_data,
              n_channel);
        }
      });

      parallel_for(0, n_channel, 1, [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          accscalar_t _var_sum = 0;
          for (i64 t = 0; t < num_threads; t++) {
            _var_sum += buffer_data[t * n_channel + c];
          }
          var_sum_data[c] = _var_sum;
        }
      });
        */
}

pub fn batch_norm_cpu_backward_contiguous_impl<Scalar>(
        grad_input:   &mut Tensor,
        grad_weight:  &mut Tensor,
        grad_bias:    &mut Tensor,
        grad_output:  &Tensor,
        input:        &Tensor,
        weight:       &Tensor,
        running_mean: &Tensor,
        running_var:  &Tensor,
        save_mean:    &Tensor,
        save_invstd:  &Tensor,
        train:        bool,
        eps:          f64)  {

    todo!();
        /*
            using Vec = Vectorized<Scalar>;
      using accscalar_t = acc_type<Scalar, false>;
      i64 n_batch = input.size(0);
      i64 n_channel = input.size(1);
      i64 image_size = input.numel() / n_batch / n_channel;
      i64 N = input.numel() / n_channel;

      const Scalar* grad_output_data = grad_output.data_ptr<Scalar>();
      const Scalar* input_data = input.data_ptr<Scalar>();

      Scalar* grad_input_data = grad_input.defined() ? grad_input.data_ptr<Scalar>() : nullptr;
      Scalar* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<Scalar>() : nullptr;
      Scalar* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<Scalar>() : nullptr;
      const bool grad_input_null = grad_input_data == nullptr;
      const bool grad_weight_null = grad_weight_data == nullptr;
      const bool grad_bias_null = grad_bias_data == nullptr;

      auto weight_a = conditional_accessor_1d<Scalar>(weight);
      auto save_mean_a = conditional_accessor_1d<Scalar>(save_mean);
      auto save_invstd_a = conditional_accessor_1d<Scalar>(save_invstd);
      auto running_mean_a = conditional_accessor_1d<Scalar>(running_mean);
      auto running_var_a = conditional_accessor_1d<Scalar>(running_var);

      // parallel dim reduce on 'channel'
      parallel_for(0, n_channel, 1, [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          Scalar w = weight.defined() ? weight_a[c] : 1;

          Scalar mean, invstd;
          if (train) {
            mean = save_mean_a[c];
            invstd = save_invstd_a[c];
          } else {
            mean = running_mean_a[c];
            invstd = 1 / sqrt(running_var_a[c] + eps);
          }

          // reduce over grad_output in feature plane
          // compute 1) sum; 2) dot product of Q(X) and dY.
          // fuse into a single loop to reuse dY
          //
          accscalar_t sum = 0;
          accscalar_t dotp = 0;
          for (i64 n = 0; n < n_batch; n++) {
            const Scalar* x_ptr = input_data + n * n_channel * image_size + c * image_size;
            const Scalar* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

            sum += vec::reduce_all<Scalar>(
                [](Vec& x, Vec& y) { return x + y; },
                dy_ptr,
                image_size);

            dotp += vec::map2_reduce_all<Scalar>(
                [mean](Vec x, Vec dy) { return (x - Vec(mean)) * dy; },
                [](Vec x, Vec y) { return x + y; },
                x_ptr,
                dy_ptr,
                image_size);
          }

          if (!grad_input_null) {
            if (train) {
              Scalar k = (Scalar) dotp * invstd * invstd / N;
              Scalar grad_mean = sum / N;

              for (i64 n = 0; n < n_batch; n++) {
                const Scalar* x_ptr = input_data + n * n_channel * image_size + c * image_size;
                Scalar* dx_ptr = grad_input_data + n * n_channel * image_size + c * image_size;
                const Scalar* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

                // Scalar math:
                // for (i64 j = 0; j < image_size; ++j) {
                //   Scalar dx = (x_ptr[j] - mean) * k;
                //   dx_ptr[j] = (dy_ptr[j] - grad_mean - dx) * invstd * w;
                // }
                vec::map2<Scalar>(
                    [=](Vec x, Vec dy) {
                      Vec dx = (x - Vec(mean)) * Vec(k);
                      return (dy - Vec(grad_mean) - dx) * Vec(invstd) * Vec(w);
                    },
                    dx_ptr,
                    x_ptr,
                    dy_ptr,
                    image_size);
              }
            } else { // evaluation mode
              for (i64 n = 0; n < n_batch; n++) {
                Scalar* dx_ptr = grad_input_data + n * n_channel * image_size + c * image_size;
                const Scalar* dy_ptr = grad_output_data + n * n_channel * image_size + c * image_size;

                // Scalar math:
                // for (i64 j = 0; j < image_size; ++j) {
                //   dx_ptr[j] = dy_ptr[j] * invstd * w;
                // }
                vec::map<Scalar>(
                    [=](Vec dy) { return dy * Vec(invstd) * Vec(w); },
                    dx_ptr,
                    dy_ptr,
                    image_size);
              }
            }
          }

          if (!grad_weight_null) {
            grad_weight_data[c] = dotp * invstd;
          }

          if (!grad_bias_null) {
            grad_bias_data[c] = sum;
          }
        }
      });
        */
}

pub fn batch_norm_cpu_backward_channels_last_impl<Scalar>(
        grad_input:   &mut Tensor,
        grad_weight:  &mut Tensor,
        grad_bias:    &mut Tensor,
        grad_output:  &Tensor,
        input:        &Tensor,
        weight:       &Tensor,
        running_mean: &Tensor,
        running_var:  &Tensor,
        save_mean:    &Tensor,
        save_invstd:  &Tensor,
        train:        bool,
        eps:          f64)  {

    todo!();
        /*
            using Vec = Vectorized<Scalar>;
      using accscalar_t = acc_type<Scalar, false>;
      i64 n_channel = input.size(1);
      i64 N = input.numel() / n_channel;

      const Scalar* grad_output_data = grad_output.data_ptr<Scalar>();
      const Scalar* input_data = input.data_ptr<Scalar>();

      Scalar* grad_input_data = grad_input.defined() ? grad_input.data_ptr<Scalar>() : nullptr;
      Scalar* grad_weight_data = grad_weight.defined() ? grad_weight.data_ptr<Scalar>() : nullptr;
      Scalar* grad_bias_data = grad_bias.defined() ? grad_bias.data_ptr<Scalar>() : nullptr;

      Scalar* save_mean_data = conditional_data_ptr<Scalar>(save_mean);
      Scalar* save_invstd_data = conditional_data_ptr<Scalar>(save_invstd);
      Scalar* running_mean_data = conditional_data_ptr<Scalar>(running_mean);
      Scalar* running_var_data = conditional_data_ptr<Scalar>(running_var);

      Tensor weight_ = weight.defined() ? weight : ones({n_channel}, input.options());
      const Scalar* weight_data = weight_.data_ptr<Scalar>();

      Scalar* mean_ptr = nullptr;
      Scalar* invstd_ptr = nullptr;
      Tensor invstd = empty({0}, input.options());
      if (train) {
        mean_ptr = save_mean_data;
        invstd_ptr = save_invstd_data;
      } else {
        mean_ptr = running_mean_data;

        invstd.resize_({n_channel});
        invstd_ptr = invstd.data_ptr<Scalar>();
        for (i64 c = 0; c < n_channel; c++) {
          invstd_ptr[c] = 1 / sqrt(running_var_data[c] + eps);
        }
      }

      // Typical vertical reduce from shape of {NHW, C} to {C}.
      // Apply two path parallel reduction:
      // First path: allocate an immediate buffer of size {2, max_threads, C}, parallel along dim0,
      //    sum = buffer[0], dotp = buffer[2]
      //
      // Second path: parallel along dim1 of the immediate buffer.
      //
      int num_threads = get_num_threads();
      Tensor buffer = empty({2, num_threads, n_channel}, input.options()).zero_();
      Scalar* sum_data = buffer.data_ptr<Scalar>();
      Scalar* dotp_data = sum_data + num_threads * n_channel;

      // compute sum and dotp per feature plain,
      // fuse into a single loop to reuse grad_output in L1.
      parallel_for(0, N, 1, [&](i64 begin, i64 end) {
        int tid = get_thread_num();
        TORCH_CHECK(tid < num_threads, "expect thread id smaller than ", num_threads, ", got thread id ", tid);
        Scalar* sum_ptr = sum_data + tid * n_channel;
        Scalar* dotp_ptr = dotp_data + tid * n_channel;
        for (i64 i = begin; i < end; i++) {
          const Scalar* x_ptr = input_data + i * n_channel;
          const Scalar* dy_ptr = grad_output_data + i * n_channel;

          vec::map2<Scalar>(
              [](Vec sum, Vec dy) { return sum + dy; },
              sum_ptr,
              sum_ptr,
              dy_ptr,
              n_channel);

          vec::map4<Scalar>(
              [](Vec dotp, Vec x, Vec mean, Vec dy) { return dotp + (x - mean) * dy; },
              dotp_ptr,
              dotp_ptr,
              x_ptr,
              mean_ptr,
              dy_ptr,
              n_channel);
        }
      });

      parallel_for(0, n_channel, 1, [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          // store the final result of sum and dotp in the 1st lane of immediate buffer,
          // so that we won't need to allocate anther buffer to store the temp values.
          accscalar_t _sum = 0;
          for (i64 t = 0; t < num_threads; t++) {
            _sum += sum_data[t * n_channel + c];
          }
          sum_data[/* 0 * n_channel + */c] = _sum;

          accscalar_t _dotp = 0;
          for (i64 t = 0; t < num_threads; t++) {
            _dotp += dotp_data[t * n_channel + c];
          }
          dotp_data[/* 0 * n_channel + */c] = _dotp;
        }
      });

      // compute grad_input
      const i64 loop_size = n_channel - (n_channel % Vec::size());
      if (grad_input.defined()) {
        parallel_for(0, N, 1, [&](i64 begin, i64 end) {
          for (i64 i = begin; i < end; i++) {
            Scalar* dx_ptr = grad_input_data + i * n_channel;
            const Scalar* x_ptr = input_data + i * n_channel;
            const Scalar* dy_ptr = grad_output_data + i * n_channel;
            if (train) {
              i64 d = 0;
              for (; d < loop_size; d += Vec::size()) {
                Vec x = Vec::loadu(x_ptr + d);
                Vec mean = Vec::loadu(mean_ptr + d);
                Vec dotp = Vec::loadu(dotp_data + d);
                Vec invstd = Vec::loadu(invstd_ptr + d);
                Vec k = dotp * invstd * invstd / Vec(N);
                Vec dx = (x - mean) * k;
                Vec dy = Vec::loadu(dy_ptr + d);
                Vec grad_mean = Vec::loadu(sum_data + d) / Vec(N);
                Vec w = Vec::loadu(weight_data + d);
                dx = (dy - grad_mean - dx) * invstd * w;
                dx.store(dx_ptr + d);
              }
              if (n_channel - d > 0) {
                Vec x = Vec::loadu(x_ptr + d, n_channel - d);
                Vec mean = Vec::loadu(mean_ptr + d, n_channel - d);
                Vec dotp = Vec::loadu(dotp_data + d, n_channel - d);
                Vec invstd = Vec::loadu(invstd_ptr + d, n_channel - d);
                Vec k = dotp * invstd * invstd / Vec(N);
                Vec dx = (x - mean) * k;
                Vec dy = Vec::loadu(dy_ptr + d, n_channel - d);
                Vec grad_mean = Vec::loadu(sum_data + d, n_channel - d) / Vec(N);
                Vec w = Vec::loadu(weight_data + d, n_channel - d);
                dx = (dy - grad_mean - dx) * invstd * w;
                dx.store(dx_ptr + d, n_channel - d);
              }
            } else { // evaluation mode
              i64 d = 0;
              for (; d < loop_size; d += Vec::size()) {
                Vec dy = Vec::loadu(dy_ptr + d);
                Vec invstd = Vec::loadu(invstd_ptr + d);
                Vec w = Vec::loadu(weight_data + d);
                Vec dx = dy * invstd * w;
                dx.store(dx_ptr + d);
              }
              if (n_channel - d > 0) {
                Vec dy = Vec::loadu(dy_ptr + d, n_channel - d);
                Vec invstd = Vec::loadu(invstd_ptr + d, n_channel - d);
                Vec w = Vec::loadu(weight_data + d, n_channel - d);
                Vec dx = dy * invstd * w;
                dx.store(dx_ptr + d, n_channel - d);
              }
            }
          }
        });
      }

      if (grad_weight.defined()) {
        // grad_weight = dotp * invstd
        vec::map2<Scalar>(
            [](Vec dotp, Vec invstd) { return dotp * invstd; },
            grad_weight_data,
            dotp_data,
            invstd_ptr,
            n_channel);
      }

      // grad_bias = sum
      if (grad_bias.defined()) {
        vec::map<Scalar>(
            [](Vec sum) { return sum; },
            grad_bias_data,
            sum_data,
            n_channel);
      }
        */
}

pub fn batch_norm_cpu_kernel(
        output:       &mut Tensor,
        input:        &Tensor,
        weight:       &Tensor,
        bias:         &Tensor,
        save_mean:    &Tensor,
        save_invstd:  &Tensor,
        running_mean: &Tensor,
        running_var:  &Tensor,
        train:        bool,
        eps:          f64)  {
    
    todo!();
        /*
            switch (input.suggest_memory_format()) {
        case MemoryFormat::Contiguous: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_contiguous", [&] {
            batch_norm_cpu_contiguous_impl<Scalar>(output, input, weight, bias,
                save_mean, save_invstd, running_mean, running_var, train, eps);
          });
          break;
        }
        case MemoryFormat::ChannelsLast: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_channels_last", [&] {
            batch_norm_cpu_channels_last_impl<Scalar>(output, input, weight, bias,
                save_mean, save_invstd, running_mean, running_var, train, eps);
          });
          break;
        }
        default:
          TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
        */
}

pub fn batch_norm_cpu_collect_stats_kernel(
        mean:    &mut Tensor,
        var_sum: &mut Tensor,
        input:   &Tensor)  {
    
    todo!();
        /*
            i64 image_size = input.numel() / input.size(0) / input.size(1);
      switch (input.suggest_memory_format()) {
        case MemoryFormat::Contiguous: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_collect_stats_contiguous", [&] {
            if (image_size == 1) { // NC11 is also channels last
              batch_norm_cpu_collect_stats_channels_last_impl<Scalar>(mean, var_sum, input);
            } else {
              batch_norm_cpu_collect_stats_contiguous_impl<Scalar>(mean, var_sum, input);
            }
          });
          break;
        }
        case MemoryFormat::ChannelsLast: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_collect_stats_channels_last", [&] {
            batch_norm_cpu_collect_stats_channels_last_impl<Scalar>(mean, var_sum, input);
          });
          break;
        }
        default:
          TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
        */
}

pub fn batch_norm_cpu_backward_kernel(
        grad_input:   &mut Tensor,
        grad_weight:  &mut Tensor,
        grad_bias:    &mut Tensor,
        grad_output:  &Tensor,
        input:        &Tensor,
        weight:       &Tensor,
        running_mean: &Tensor,
        running_var:  &Tensor,
        save_mean:    &Tensor,
        save_invstd:  &Tensor,
        train:        bool,
        eps:          f64)  {
    
    todo!();
        /*
            i64 image_size = input.numel() / input.size(0) / input.size(1);
      switch (input.suggest_memory_format()) {
        case MemoryFormat::Contiguous: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_backward_contiguous", [&] {
            if (image_size == 1) { // NC11 is also channels last
              batch_norm_cpu_backward_channels_last_impl<Scalar>(grad_input, grad_weight, grad_bias,
                  grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
            } else {
              batch_norm_cpu_backward_contiguous_impl<Scalar>(grad_input, grad_weight, grad_bias,
                  grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
            }
          });
          break;
        }
        case MemoryFormat::ChannelsLast: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "batch_norm_cpu_backward_channels_last", [&] {
            batch_norm_cpu_backward_channels_last_impl<Scalar>(grad_input, grad_weight, grad_bias,
                grad_output, input, weight, running_mean, running_var, save_mean, save_invstd, train, eps);
          });
          break;
        }
        default:
          TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
        */
}

register_dispatch!{batch_norm_cpu_stub               , &batch_norm_cpu_kernel}
register_dispatch!{batch_norm_cpu_collect_stats_stub , &batch_norm_cpu_collect_stats_kernel}
register_dispatch!{batch_norm_cpu_backward_stub      , &batch_norm_cpu_backward_kernel}
