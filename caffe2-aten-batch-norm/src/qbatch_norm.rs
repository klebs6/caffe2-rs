crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qbatch_norm.cpp]

define_dispatch!{qbatch_norm_stub}
define_dispatch!{qbatch_norm_relu_stub}

pub fn compute_fused_params(
        channels:     i64,
        weight_data:  *const f32,
        bias_data:    *const f32,
        mean_data:    *const f32,
        var_data:     *const f32,
        eps:          f64,
        input_scale:  f64,
        output_scale: f64,
        alpha_data:   *mut f32,
        beta_data:    *mut f32)  {
    
    todo!();
        /*
            // Batch Normalization
      // output(n, c, h, w)
      //     = (input(n, c, h, w) - mean(c)) / sqrt(var(c) + eps) * weight(c)
      //         + bias(c)
      // We factor out inv_sigma(c) = 1 / sqrt(var(c) + eps).
      for (i64 c = 0; c < channels; c++) {
        float inv_sigma = 1.0 / sqrt(var_data[c] + static_cast<float>(eps));
        float weight_v = weight_data ? weight_data[c] : 1;
        float bias_v = bias_data ? bias_data[c] : 0;
        alpha_data[c] = inv_sigma * weight_v * (input_scale / output_scale);
        beta_data[c] = (bias_v - mean_data[c] * inv_sigma * weight_v) / output_scale;
      }
        */
}

pub fn q_batch_norm1d_impl<const ReluFused: bool>(
        qx:                Tensor,
        mb_weight:         Option<Tensor>,
        mb_bias:           Option<Tensor>,
        mean:              Tensor,
        var:               Tensor,
        eps:               f64,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(mb_weight.has_value(), "Weight must be provided");
      TORCH_CHECK(mb_bias.has_value(), "Bias must be provided");
      const auto& weight = *mb_weight;
      const auto& bias = *mb_bias;

      if (qx.numel() == 0) {
        auto out = qx.clone();
        return out;
      }
      i64 ndim = qx.dim();
      TORCH_CHECK(ndim == 2 || ndim == 3, "Expecting the input tensor of rank 2 or 3.");
      const i64 N = qx.size(0);
      const i64 C = qx.size(1);
      const i64 H = ndim == 3 ? qx.size(2) : 1;

      TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
      TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

      const float* weight_data = weight.template data_ptr<float>();
      const float* bias_data = bias.template data_ptr<float>();

      TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
      TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

      Tensor alpha = empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor beta = empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      float* alpha_data = alpha.data_ptr<float>();
      float* beta_data = beta.data_ptr<float>();

      const float* mean_data = mean.template data_ptr<float>();
      const float* var_data = var.template data_ptr<float>();

      if (ndim == 2) {
        // create a fake H and W dimension so we can use NHWC
        qx = qx.unsqueeze(-1).unsqueeze(-1);
      } else {
        // create a fake W dimension so we can use NHWC
        qx = qx.unsqueeze(-1);
      }

      auto oSizes = qx.sizes();
      auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);
      Tensor qy = _empty_affine_quantized(
          oSizes,
          device(kCPU)
            .dtype(qx_nhwc.scalar_type())
            .memory_format(MemoryFormat::ChannelsLast),
          output_scale,
          output_zero_point,
          nullopt);

      compute_fused_params(
          C,
          weight_data,
          bias_data,
          mean_data,
          var_data,
          eps,
          qx.q_scale(),
          output_scale,
          alpha_data,
          beta_data);
      if (ReluFused) {
        qbatch_norm_relu_stub(
            qx.device().type(),
            N,
            C,
            H,
            qx.q_zero_point(),
            output_zero_point,
            qx_nhwc,
            alpha,
            beta,
            qy);
      } else {
        qbatch_norm_stub(
            qx.device().type(),
            N,
            C,
            H,
            qx.q_zero_point(),
            output_zero_point,
            qx_nhwc,
            alpha,
            beta,
            qy);
      }
      // Remove the fake dimension, and go back to contiguous format
      // (since there is no 4th channel). Note, this has a performance
      // cost.
      Tensor result = qy.contiguous(MemoryFormat::Contiguous).squeeze(-1);
      if (ndim == 2) {
        result = result.squeeze(-1);
      }
      return result;
        */
}

pub fn q_batch_norm2d_impl<const ReluFused: bool>(
        qx:                Tensor,
        mb_weight:         Option<Tensor>,
        mb_bias:           Option<Tensor>,
        mean:              Tensor,
        var:               Tensor,
        eps:               f64,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(mb_weight.has_value(), "Weight must be provided");
      TORCH_CHECK(mb_bias.has_value(), "Bias must be provided");
      const auto& weight = *mb_weight;
      const auto& bias = *mb_bias;

      if (qx.numel() == 0) {
        auto out = qx.clone();
        return out;
      }
      i64 ndim = qx.dim();
      TORCH_CHECK(ndim == 4, "Expecting the input tensor of rank 4.");
      const i64 N = qx.size(0);
      const i64 C = qx.size(1);
      const i64 H = qx.size(2);
      const i64 W = qx.size(3);

      TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
      TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

      const float* weight_data = weight.template data_ptr<float>();
      const float* bias_data = bias.template data_ptr<float>();

      TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
      TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

      Tensor alpha = empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor beta = empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      float* alpha_data = alpha.data_ptr<float>();
      float* beta_data = beta.data_ptr<float>();

      const float* mean_data = mean.template data_ptr<float>();
      const float* var_data = var.template data_ptr<float>();

      auto oSizes = qx.sizes();
      auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast);
      Tensor qy = _empty_affine_quantized(
          oSizes,
          device(kCPU)
            .dtype(qx_nhwc.scalar_type())
            .memory_format(MemoryFormat::ChannelsLast),
          output_scale,
          output_zero_point,
          nullopt);

      compute_fused_params(
          C,
          weight_data,
          bias_data,
          mean_data,
          var_data,
          eps,
          qx.q_scale(),
          output_scale,
          alpha_data,
          beta_data);
      if (ReluFused) {
        qbatch_norm_relu_stub(
            qx.device().type(),
            N,
            C,
            H * W,
            qx.q_zero_point(),
            output_zero_point,
            qx_nhwc,
            alpha,
            beta,
            qy);
      } else {
        qbatch_norm_stub(
            qx.device().type(),
            N,
            C,
            H * W,
            qx.q_zero_point(),
            output_zero_point,
            qx_nhwc,
            alpha,
            beta,
            qy);
      }
      return qy;
        */
}

pub fn q_batch_norm3d_impl<const ReluFused: bool>(
        qx:                Tensor,
        mb_weight:         Option<Tensor>,
        mb_bias:           Option<Tensor>,
        mean:              Tensor,
        var:               Tensor,
        eps:               f64,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(mb_weight.has_value(), "Weight must be provided")
      TORCH_CHECK(mb_bias.has_value(), "Bias must be provided")

      const auto& weight = *mb_weight;
      const auto& bias = *mb_bias;

      if (qx.numel() == 0) {
        auto out = qx.clone();
        return out;
      }
      i64 ndim = qx.dim();
      TORCH_CHECK(ndim == 5, "Expecting the input tensor of rank 5.");
      const i64 N = qx.size(0);
      const i64 C = qx.size(1);
      const i64 D = qx.size(2);
      const i64 H = qx.size(3);
      const i64 W = qx.size(4);

      TORCH_CHECK(weight.numel() == C, "Expect weight size to match C");
      TORCH_CHECK(bias.numel() == C, "Expect weight size to match C");

      const float* weight_data = weight.template data_ptr<float>();
      const float* bias_data = bias.template data_ptr<float>();

      TORCH_CHECK(mean.numel() == C, "Mean size must match channel dimension");
      TORCH_CHECK(var.numel() == C, "Variance size must match channel dimension");

      Tensor alpha = empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      Tensor beta = empty_like(mean, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      float* alpha_data = alpha.data_ptr<float>();
      float* beta_data = beta.data_ptr<float>();

      const float* mean_data = mean.template data_ptr<float>();
      const float* var_data = var.template data_ptr<float>();

      auto oSizes = qx.sizes();
      auto qx_nhwc = qx.contiguous(MemoryFormat::ChannelsLast3d);
      Tensor qy = _empty_affine_quantized(
          oSizes,
          device(kCPU)
            .dtype(qx_nhwc.scalar_type())
            .memory_format(MemoryFormat::ChannelsLast3d),
          output_scale,
          output_zero_point,
          nullopt);

      compute_fused_params(
          C,
          weight_data,
          bias_data,
          mean_data,
          var_data,
          eps,
          qx.q_scale(),
          output_scale,
          alpha_data,
          beta_data);

      if (ReluFused) {
        qbatch_norm_relu_stub(
            qx.device().type(),
            N,
            C,
            D * H * W,
            qx.q_zero_point(),
            output_zero_point,
            qx_nhwc,
            alpha,
            beta,
            qy);
      } else {
        qbatch_norm_stub(
            qx.device().type(),
            N,
            C,
            D * H * W,
            qx.q_zero_point(),
            output_zero_point,
            qx_nhwc,
            alpha,
            beta,
            qy);
      }
      return qy;
        */
}

pub fn q_batch_norm_impl<const ReluFused: bool>(
        qx:                Tensor,
        mb_weight:         Option<Tensor>,
        mb_bias:           Option<Tensor>,
        mean:              Tensor,
        var:               Tensor,
        eps:               f64,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {

    todo!();
        /*
            Tensor qy;
      i64 dim = qx.dim();
      if (dim == 2 || dim == 3) {
        qy = q_batch_norm1d_impl<ReluFused>(
            qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
      } else if (dim == 4) {
        qy = q_batch_norm2d_impl<ReluFused>(
            qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
      } else if (dim == 5) {
        qy = q_batch_norm3d_impl<ReluFused>(
            qx, mb_weight, mb_bias, mean, var, eps, output_scale, output_zero_point);
      } else {
        TORCH_CHECK(false, "quantized::batch_norm only support 2d, 3d, 4d or 5d inputs.");
      }
      return qy;
        */
}


pub fn quantized_batch_norm(
        qx:                &Tensor,
        weight_opt:        &Option<Tensor>,
        bias_opt:          &Option<Tensor>,
        mean:              &Tensor,
        var:               &Tensor,
        eps:               f64,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;
      const Tensor& bias = value_or_else(bias_opt, [] {return Tensor();});

      Tensor qy;
      // TODO: this should arguably support 3d as well
      qy = q_batch_norm2d_impl<false>(
          qx,
          weight.defined() ? make_optional(weight) : nullopt,
          bias.defined() ? make_optional(bias) : nullopt,
          mean, var, eps, output_scale, output_zero_point);
      return qy;
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm"),        TORCH_FN(q_batch_norm_impl<false>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm_relu"),   TORCH_FN(q_batch_norm_impl<true>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm1d"),      TORCH_FN(q_batch_norm1d_impl<false>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm1d_relu"), TORCH_FN(q_batch_norm1d_impl<true>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm2d"),      TORCH_FN(q_batch_norm2d_impl<false>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm2d_relu"), TORCH_FN(q_batch_norm2d_impl<true>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm3d"),      TORCH_FN(q_batch_norm3d_impl<false>));
      m.impl(TORCH_SELECTIVE_NAME("quantized::batch_norm3d_relu"), TORCH_FN(q_batch_norm3d_impl<true>));
    }
    */
}
