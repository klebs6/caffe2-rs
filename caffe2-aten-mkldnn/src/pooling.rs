crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/mkldnn/Pooling.cpp]

#[cfg(not(feature = "mkldnn"))]
pub mod mkldnn_disabled {

    use super::*;

    pub fn mkldnn_max_pool2d(
            self_:       &Tensor,
            kernel_size: &[i32],
            stride:      &[i32],
            padding:     &[i32],
            dilation:    &[i32],
            ceil_mode:   bool) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_max_pool2d: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_max_pool3d(
            self_:       &Tensor,
            kernel_size: &[i32],
            stride:      &[i32],
            padding:     &[i32],
            dilation:    &[i32],
            ceil_mode:   bool) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_max_pool3d: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_avg_pool2d(
            self_:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool2d: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_avg_pool2d_out<'a>(
            self_:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>,
            output:            &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool2d_out: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_avg_pool3d(
            self_:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool3d: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_avg_pool3d_out<'a>(
            self_:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>,
            output:            &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool3d_out: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_adaptive_avg_pool2d(
            input:       &Tensor,
            output_size: &[i32]) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_adaptive_avg_pool2d: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_adaptive_avg_pool2d_out<'a>(
            input:       &Tensor,
            output_size: &[i32],
            output:      &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_adaptive_avg_pool2d_out: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_max_pool2d_backward(
            grad_output: &Tensor,
            output:      &Tensor,
            input:       &Tensor,
            kernel_size: &[i32],
            stride:      &[i32],
            padding:     &[i32],
            dilation:    &[i32],
            ceil_mode:   bool) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_max_pool2d_backward: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_max_pool3d_backward(
            grad_output: &Tensor,
            output:      &Tensor,
            input:       &Tensor,
            kernel_size: &[i32],
            stride:      &[i32],
            padding:     &[i32],
            dilation:    &[i32],
            ceil_mode:   bool) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_max_pool3d_backward: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_avg_pool2d_backward_out<'a>(
            grad_output:       &Tensor,
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>,
            grad_input:        &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool2d_backward_out: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_avg_pool2d_backward(
            grad_output:       &Tensor,
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool2d_backward: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_avg_pool3d_backward_out<'a>(
            grad_output:       &Tensor,
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>,
            grad_input:        &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool3d_backward_out: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_avg_pool3d_backward(
            grad_output:       &Tensor,
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool3d_backward: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_adaptive_avg_pool2d_backward_out<'a>(
            grad_input:  &mut Tensor,
            grad_output: &Tensor,
            input:       &Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_adaptive_avg_pool2d_backward_out: ATen not compiled with MKLDNN support");
            */
    }

    pub fn mkldnn_adaptive_avg_pool2d_backward(
            grad_output: &Tensor,
            input:       &Tensor) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_adaptive_avg_pool2d_backward: ATen not compiled with MKLDNN support");
            */
    }
}

#[cfg(feature = "mkldnn")]
pub mod mkldnn_enabled {

    use super::*;

    pub fn mkldnn_pooling(
            input:       &Tensor,
            kernel_size: &[i32],
            stride:      &[i32],
            padding:     &[i32],
            dilation:    &[i32],
            ceil_mode:   bool,
            algo:        Algorithm) -> Tensor {
        
        todo!();
            /*
                const i64 dims = input.dim() - 2;
          auto kernel_size_vec = expand_param_if_needed(kernel_size, "kernel_size", dims);
          if (stride.empty()) stride = kernel_size;
          auto stride_vec = expand_param_if_needed(stride, "stride", dims);
          auto padding_vec = expand_param_if_needed(padding, "padding", dims);
          auto padding_vec_l = padding_vec;
          auto padding_vec_r = padding_vec;
          auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

          const ideep::tensor& x = itensor_from_mkldnn(input);
          std::vector<i64> output_sizes;

          if (ceil_mode) {
            // MKLDNN does not support ceil mode, so we adjust padding
            // on the right side to match behavior. Adjust output size
            // accordingly.
            const std::vector<i64> output_sizes_ceil = pool_output_sizes(
                input.sizes(),
                kernel_size_vec,
                stride_vec,
                padding_vec_l,
                padding_vec_r,
                dilation_vec,
                true /* ceil_mode */);

            // adjust padding until output sizes agree
            bool all_equal = false;
            while (!all_equal) {
              output_sizes = pool_output_sizes(
                  input.sizes(),
                  kernel_size_vec,
                  stride_vec,
                  padding_vec_l,
                  padding_vec_r,
                  dilation_vec,
                  false /*ceil_mode */);

              all_equal = true;
              for (usize i = 2; i < input.sizes().size(); ++i) {
                if (output_sizes[i] < output_sizes_ceil[i]) {
                   padding_vec_r[i - 2]++;
                   all_equal = false;
                }
              }
            }
          } else {
            output_sizes = pool_output_sizes(
                input.sizes(),
                kernel_size_vec,
                stride_vec,
                padding_vec_l,
                padding_vec_r,
                dilation_vec,
                false /*ceil_mode */);
          }

          auto aprop_kind = ideep::prop_kind::forward;
          // for max_pool, prop_kind::forward will save indices as workspace for backward use,
          // for inference, don't need the indices, set aprop_kind to prop_kind::forward_inference
          // can reduce the memory use.
          if (ideep::algorithm::pooling_max == algo
              && !(input.requires_grad() && at::GradMode::is_enabled())) {
            aprop_kind = ideep::prop_kind::forward_inference;
          }

          ideep::tensor y;
          ideep::pooling_forward::compute(
              x,
              {output_sizes.cbegin(), output_sizes.cend()},
              y,
              {stride_vec.cbegin(), stride_vec.cend()},
              {kernel_size_vec.cbegin(), kernel_size_vec.cend()},
              {padding_vec_l.cbegin(), padding_vec_l.cend()},
              {padding_vec_r.cbegin(), padding_vec_r.cend()},
              algo,
              aprop_kind);

          return new_with_itensor_mkldnn(std::move(y), optTypeMetaToScalarType(input.options().dtype_opt()), input.options().device_opt());
            */
    }

    pub fn mkldnn_pooling_backward(
        grad_output: &Tensor,
        output:      &Tensor,
        input:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool,
        algo:        Algorithm) -> Tensor {

        todo!();
            /*
                const i64 dims = input.dim() - 2;
          auto kernel_size_vec = expand_param_if_needed(kernel_size, "kernel_size", dims);
          auto stride_vec = expand_param_if_needed(stride, "stride", dims);
          auto padding_vec = expand_param_if_needed(padding, "padding", dims);
          auto padding_vec_l = padding_vec;
          auto padding_vec_r = padding_vec;
          auto dilation_vec = expand_param_if_needed(dilation, "dilation", dims);

          if (ceil_mode) {
            // MKLDNN does not support ceil mode, so we adjust padding
            // on the right side to match behavior. Adjust output size
            // accordingly.
            const std::vector<i64> output_sizes_ceil = pool_output_sizes(
                input.sizes(),
                kernel_size_vec,
                stride_vec,
                padding_vec_l,
                padding_vec_r,
                dilation_vec,
                true /* ceil_mode */);

            // adjust padding until output sizes agree
            bool all_equal = false;
            std::vector<i64> output_sizes;
            while (!all_equal) {
              output_sizes = pool_output_sizes(
                  input.sizes(),
                  kernel_size_vec,
                  stride_vec,
                  padding_vec_l,
                  padding_vec_r,
                  dilation_vec,
                  false /*ceil_mode */);

              all_equal = true;
              for (usize i = 2; i < input.sizes().size(); ++i) {
                if (output_sizes[i] < output_sizes_ceil[i]) {
                   padding_vec_r[i - 2]++;
                   all_equal = false;
                }
              }
            }
          }

          const ideep::tensor& grady = itensor_from_mkldnn(grad_output);
          const ideep::tensor& y = itensor_from_mkldnn(output);
          const ideep::tensor& x = itensor_from_mkldnn(input);
          ideep::tensor gradx;
          ideep::pooling_backward::compute(
              grady,
              y,
              x,
              gradx,
              {stride_vec.cbegin(), stride_vec.cend()},
              {kernel_size_vec.cbegin(), kernel_size_vec.cend()},
              {padding_vec_l.cbegin(), padding_vec_l.cend()},
              {padding_vec_r.cbegin(), padding_vec_r.cend()},
              algo);

          return new_with_itensor_mkldnn(std::move(gradx),
                                         optTypeMetaToScalarType(grad_output.options().dtype_opt()),
                                         grad_output.options().device_opt());
            */
    }

    pub fn mkldnn_max_pool2d(
        input:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> Tensor {

        todo!();
            /*
                TORCH_CHECK(std::all_of(dilation.cbegin(), dilation.cend(), [](i64 i) { return 1 == i; }),
              "mkldnn_max_pool2d does not support dilation case");
          if (input.scalar_type() == ScalarType::BFloat16) {
            TORCH_CHECK(mkldnn_bf16_device_check(),
                "mkldnn_max_pool2d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
          }

          return _mkldnn_pooling(
              input,
              kernel_size,
              stride,
              padding,
              dilation,
              ceil_mode,
              ideep::algorithm::pooling_max);
            */
    }

    pub fn mkldnn_max_pool3d(
        input:       &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        dilation:    &[i32],
        ceil_mode:   bool) -> Tensor {

        todo!();
            /*
                TORCH_CHECK(std::all_of(dilation.cbegin(), dilation.cend(), [](i64 i) { return 1 == i; }),
              "mkldnn_max_pool3d does not support dilation case");
          if (input.scalar_type() == ScalarType::BFloat16) {
            TORCH_CHECK(mkldnn_bf16_device_check(),
                "mkldnn_max_pool3d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
          }

          return _mkldnn_pooling(
              input,
              kernel_size,
              stride,
              padding,
              dilation,
              ceil_mode,
              ideep::algorithm::pooling_max);
            */
    }

    pub fn mkldnn_avg_pool2d(
        input:             &Tensor,
        kernel_size:       &[i32],
        stride:            &[i32],
        padding:           &[i32],
        ceil_mode:         bool,
        count_include_pad: bool,
        divisor_override:  Option<i64>) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(!divisor_override.has_value(),
              "mkldnn_avg_pool2d operator does not support divisor");
          if (input.scalar_type() == ScalarType::BFloat16) {
            TORCH_CHECK(mkldnn_bf16_device_check(),
                "mkldnn_avg_pool2d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
          }

          return _mkldnn_pooling(
              input,
              kernel_size,
              stride,
              padding,
              /*dilation*/ std::vector<i64>{1, 1},
              ceil_mode,
              count_include_pad ? ideep::algorithm::pooling_avg_include_padding
                                : ideep::algorithm::pooling_avg_exclude_padding);
            */
    }


    pub fn mkldnn_avg_pool2d_out<'a>(
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>,
            output:            &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool2d_out: in-place mkldnn operations are not supported yet");
            */
    }

    pub fn mkldnn_avg_pool3d(
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(!divisor_override.has_value(), "mkldnn_avg_pool3d operator does not support divisor");
          if (input.scalar_type() == ScalarType::BFloat16) {
            TORCH_CHECK(mkldnn_bf16_device_check(),
                "mkldnn_avg_pool3d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
          }

          return _mkldnn_pooling(
              input,
              kernel_size,
              stride,
              padding,
              /*dilation*/ std::vector<i64>{1, 1, 1},
              ceil_mode,
              count_include_pad ? ideep::algorithm::pooling_avg_include_padding
                                : ideep::algorithm::pooling_avg_exclude_padding);
            */
    }

    pub fn mkldnn_avg_pool3d_out<'a>(
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>,
            output:            &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool3d_out: in-place mkldnn operations are not supported yet");
            */
    }

    pub fn mkldnn_adaptive_avg_pool2d(
            input:       &Tensor,
            output_size: &[i32]) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(input.dim() == 4, "mkldnn_adaptive_avg_pool2d: Expect 2D input");
          if (input.scalar_type() == ScalarType::BFloat16) {
            TORCH_CHECK(mkldnn_bf16_device_check(),
                "mkldnn_adaptive_avg_pool2d: bf16 path needs the cpu support avx512bw, avx512vl and avx512dq");
          }
          auto output_size_vec =
              expand_param_if_needed(output_size, "output_size", input.dim() - 2);
          std::vector<i64> kernel_size(input.dim() - 2);
          for (i64 i = 2; i < input.dim(); ++i) {
            auto s1 = input.size(i);
            auto s2 = output_size_vec[i - 2];
            TORCH_CHECK(s2 != 0, "output size can not be zero");
            TORCH_CHECK(
                s1 % s2 == 0,
                "input size is not divisible by the output size is not supported yet");
            kernel_size[i - 2] = s1 / s2;
          }
          return _mkldnn_pooling(
              input,
              kernel_size,
              /*stride*/ kernel_size,
              /*padding*/ {0, 0},
              /*dilation*/ {1, 1},
              /*ceil_mode*/ false,
              /*algo*/ ideep::algorithm::pooling_avg);
            */
    }

    pub fn mkldnn_adaptive_avg_pool2d_out<'a>(
            input:       &Tensor,
            output_size: &[i32],
            output:      &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_adaptive_avg_pool2d_out: in-place mkldnn operations are not supported yet");
            */
    }

    pub fn mkldnn_max_pool2d_backward(
            grad_output: &Tensor,
            output:      &Tensor,
            input:       &Tensor,
            kernel_size: &[i32],
            stride:      &[i32],
            padding:     &[i32],
            dilation:    &[i32],
            ceil_mode:   bool) -> Tensor {
        
        todo!();
            /*
                return _mkldnn_pooling_backward(
              grad_output,
              output,
              input,
              kernel_size,
              stride,
              padding,
              dilation,
              ceil_mode,
              ideep::algorithm::pooling_max);
            */
    }

    pub fn mkldnn_max_pool3d_backward(
            grad_output: &Tensor,
            output:      &Tensor,
            input:       &Tensor,
            kernel_size: &[i32],
            stride:      &[i32],
            padding:     &[i32],
            dilation:    &[i32],
            ceil_mode:   bool) -> Tensor {
        
        todo!();
            /*
                return _mkldnn_pooling_backward(
              grad_output,
              output,
              input,
              kernel_size,
              stride,
              padding,
              dilation,
              ceil_mode,
              ideep::algorithm::pooling_max);
            */
    }

    pub fn mkldnn_avg_pool2d_backward(
            grad_output:       &Tensor,
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>) -> Tensor {
        
        todo!();
            /*
                return _mkldnn_pooling_backward(
              grad_output,
              grad_output,
              input,
              kernel_size,
              stride,
              padding,
              /*dilation*/ std::vector<i64>{1, 1},
              ceil_mode,
              count_include_pad ? ideep::algorithm::pooling_avg_include_padding
                                : ideep::algorithm::pooling_avg_exclude_padding);
            */
    }

    pub fn mkldnn_avg_pool2d_backward_out<'a>(
            grad_output:       &Tensor,
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>,
            grad_input:        &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool2d_backward_out: in-place mkldnn operations are not supported yet");
            */
    }

    pub fn mkldnn_avg_pool3d_backward(
            grad_output:       &Tensor,
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>) -> Tensor {
        
        todo!();
            /*
                return _mkldnn_pooling_backward(
              grad_output,
              grad_output,
              input,
              kernel_size,
              stride,
              padding,
              /*dilation*/ std::vector<i64>{1, 1, 1},
              ceil_mode,
              count_include_pad ? ideep::algorithm::pooling_avg_include_padding
                                : ideep::algorithm::pooling_avg_exclude_padding);
            */
    }

    pub fn mkldnn_avg_pool3d_backward_out<'a>(
            grad_output:       &Tensor,
            input:             &Tensor,
            kernel_size:       &[i32],
            stride:            &[i32],
            padding:           &[i32],
            ceil_mode:         bool,
            count_include_pad: bool,
            divisor_override:  Option<i64>,
            grad_input:        &mut Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_avg_pool3d_backward_out: in-place mkldnn operations are not supported yet");
            */
    }

    pub fn mkldnn_adaptive_avg_pool2d_backward(
        grad_output: &Tensor,
        input:       &Tensor) -> Tensor {
        
        todo!();
            /*
                TORCH_CHECK(input.dim() == 4, "mkldnn_adaptive_avg_pool2d: Input is expected a 4D tenosor");

          auto output_size_vec = grad_output.sizes();
          std::vector<i64> kernel_size(input.dim() - 2);
          for (const auto i: c10::irange(2, input.dim())) {
            auto s1 = input.size(i);
            auto s2 = output_size_vec[i];
            TORCH_CHECK(s2 != 0, "output size can not be zero");
            TORCH_CHECK(
                s1 % s2 == 0,
                "input size is not divisible by the output size is not supported yet");
                kernel_size[i - 2] = s1 / s2;
          }
          return _mkldnn_pooling_backward(
              grad_output,
              grad_output,
              input,
              kernel_size,
              /*stride*/ kernel_size,
              /*padding*/ {0, 0},
              /*dilation*/{1, 1},
              false,
              /*algo*/ ideep::algorithm::pooling_avg);
            */
    }

    pub fn mkldnn_adaptive_avg_pool2d_backward_out<'a>(
            grad_input:  &mut Tensor,
            grad_output: &Tensor,
            input:       &Tensor) -> &'a mut Tensor {
        
        todo!();
            /*
                TORCH_CHECK(false, "mkldnn_adaptive_avg_pool2d_backward_out: in-place mkldnn operations are not supported yet");
            */
    }
}
