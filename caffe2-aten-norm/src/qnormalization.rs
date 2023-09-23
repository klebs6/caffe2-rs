crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnormalization.cpp]

define_dispatch!{quantized_normalize_stub}

pub fn quantized_layer_norm_impl(
        input:             &Tensor,
        normalized_shape:  &[i32],
        weight:            &Tensor,
        bias:              &Tensor,
        eps:               f64,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
    todo!();
        /*
            auto M_N = _check_layer_norm_inputs(input, normalized_shape, weight, bias);
      auto M = M_N.first;
      auto N = M_N.second;
      auto X = input.expect_contiguous();
      auto gamma = weight.expect_contiguous();
      auto beta = bias.expect_contiguous();

      Tensor Y = _empty_affine_quantized(
        X->sizes(),
        X->scalar_type(),
        output_scale,
        output_zero_point,
        X->suggest_memory_format());

      if (M > 0) {
        bool affine_per_channel = false;
        int num_channels = 1; // not relevant for LayerNorm
        int num_groups = 1; // not relevant for LayerNorm
        quantized_normalize_stub(kCPU, *X, *gamma, *beta, affine_per_channel,
            num_channels, num_groups, M, N, eps, &Y);
      }
      return Y;
        */
}



pub fn quantized_group_norm_impl(
        qx:                &Tensor,
        num_groups:        i64,
        weight:            &Tensor,
        bias:              &Tensor,
        eps:               f64,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
    todo!();
        /*
            const auto& qx_contig = qx.contiguous();
      const auto& weight_contig = weight.contiguous();
      const auto& bias_contig = bias.contiguous();

      const auto input_ndim = qx_contig.dim();
      TORCH_CHECK(
          input_ndim >= 3,
          "Expected normalized_shape to be at least 3-dimensional");
      TORCH_CHECK(num_groups > 0, "Expected num_groups to be positive");

      const auto input_shape = qx_contig.sizes();
      TORCH_CHECK(input_shape[1] % num_groups == 0,
          "Expected channels to be divisible by groups");

      const i64 batches = input_shape[0];
      const i64 num_channels = input_shape[1];
      const i64 elements_per_batch =
          multiply_integers(input_shape.cbegin() + 1, input_shape.cend());

      const i64 M = batches * num_groups;
      const i64 N = elements_per_batch / num_groups;

      Tensor Y = _empty_affine_quantized(
        qx_contig.sizes(),
        qx_contig.scalar_type(),
        output_scale,
        output_zero_point,
        qx_contig.suggest_memory_format());

      if (M > 0) {
        bool affine_per_channel = true;
        quantized_normalize_stub(kCPU, qx_contig, weight_contig, bias_contig,
            affine_per_channel, num_channels, num_groups, M, N, eps, &Y);
      }
      return Y;
        */
}



pub fn quantized_instance_norm_impl(
        qx:                &Tensor,
        weight:            &Tensor,
        bias:              &Tensor,
        eps:               f64,
        output_scale:      f64,
        output_zero_point: i64) -> Tensor {
    
    todo!();
        /*
            const auto input_ndim = qx.dim();
      TORCH_CHECK(
          input_ndim >= 3,
          "Expected normalized_shape to be at least 3-dimensional");
      const auto input_shape = qx.sizes();

      // IN is GN with num_groups == num_channels
      const auto num_channels = input_shape[1];
      TORCH_CHECK(num_channels > 0, "Expected 2nd dimension to be positive");

      return quantized_group_norm_impl(
          qx, num_channels, weight, bias, eps, output_scale, output_zero_point);
        */
}

lazy_static!{
    /*
    TORCH_LIBRARY_IMPL(quantized, QuantizedCPU, m) {
      // TODO: this is kind of... blegh
      m.impl(TORCH_SELECTIVE_NAME("quantized::layer_norm"), [](
        Tensor input,
        vector<i64> normalized_shape,  // because IntArrayRef doesn't work
        optional<Tensor> weight,
        optional<Tensor> bias,
        double eps,
        double output_scale,
        i64 output_zero_point) {
          return quantized_layer_norm_impl(
              input, normalized_shape,
              weight.has_value() ? *weight : Tensor(),
              bias.has_value() ? *bias : Tensor(),
              eps, output_scale, output_zero_point);
      });
      m.impl(TORCH_SELECTIVE_NAME("quantized::group_norm"), [](
          Tensor qx,
          i64 num_groups,
          optional<Tensor> weight,
          optional<Tensor> bias,
          double eps,
          double output_scale,
          i64 output_zero_point) {
        return quantized_group_norm_impl(
            qx, num_groups,
            weight.has_value() ? *weight : Tensor(),
            bias.has_value() ? *bias : Tensor(),
            eps, output_scale, output_zero_point);
      });
      m.impl(TORCH_SELECTIVE_NAME("quantized::instance_norm"), [](
          Tensor qx,
          optional<Tensor> weight,
          optional<Tensor> bias,
          double eps,
          double output_scale,
          i64 output_zero_point) {
        return quantized_instance_norm_impl(
            qx,
            weight.has_value() ? *weight : Tensor(),
            bias.has_value() ? *bias : Tensor(),
            eps, output_scale, output_zero_point);
      });
    }
    */
}

