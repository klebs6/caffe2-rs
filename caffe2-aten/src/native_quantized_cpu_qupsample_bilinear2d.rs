// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qupsample_bilinear2d.cpp]

/**
  | native functions for the native_functions.yaml
  |
  */
pub fn upsample_bilinear2d_out_frame<Scalar>(
    output:        &mut Tensor,
    input:         &Tensor,
    input_height:  i64,
    input_width:   i64,
    output_height: i64,
    output_width:  i64,
    nbatch:        i64,
    channels:      i64,
    align_corners: bool,
    scales_h:      Option<f64>,
    scales_w:      Option<f64>)  {

    todo!();
        /*
            auto* idata = static_cast<Scalar*>(input.data_ptr());
      auto* odata = static_cast<Scalar*>(output.data_ptr());

      channels = channels * nbatch;
      auto* i_p = reinterpret_cast<typename Scalar::underlying*>(idata);
      auto* o_p = reinterpret_cast<typename Scalar::underlying*>(odata);

      // special case: just copy
      if (input_height == output_height && input_width == output_width) {
        memcpy(
            o_p,
            i_p,
            channels * input_height * input_width *
                sizeof(typename Scalar::underlying));
        return;
      }

      const auto rheight = area_pixel_compute_scale<float>(
          input_height, output_height, align_corners, scales_h);

      const auto rwidth =
          area_pixel_compute_scale<float>(input_width, output_width, align_corners, scales_w);
      float output_scale = output.q_scale() / input.q_scale();

      for (i64 h2 = 0; h2 < output_height; ++h2) {
        const auto h1r = area_pixel_compute_source_index<float>(
            rheight, h2, align_corners, /*cubic=*/false);

        const i64 h1 = h1r;
        const i64 h1p = (h1 < input_height - 1) ? 1 : 0;

        const float h1lambda = h1r - h1;
        const float h0lambda = static_cast<float>(1.) - h1lambda;

        for (i64 w2 = 0; w2 < output_width; ++w2) {
          const auto w1r = area_pixel_compute_source_index<float>(
              rwidth, w2, align_corners, /*cubic=*/false);

          const i64 w1 = w1r;
          const i64 w1p = (w1 < input_width - 1) ? 1 : 0;

          const float w1lambda = w1r - w1;
          const float w0lambda = static_cast<float>(1.) - w1lambda;
          const typename Scalar::underlying* pos1 = i_p + h1 * input_width + w1;
          typename Scalar::underlying* pos2 = o_p + h2 * output_width + w2;

          for (i64 c = 0; c < channels; ++c) {
            float result = h0lambda * (w0lambda * pos1[0] + w1lambda * pos1[w1p]) +
                h1lambda *
                    (w0lambda * pos1[h1p * input_width] +
                     w1lambda * pos1[h1p * input_width + w1p]) - input.q_zero_point();
            // requantization
            pos2[0] = native::quantize_val<Scalar>(
                          output_scale, output.q_zero_point(), result)
                          .val_;
            pos1 += input_width * input_height;
            pos2 += output_width * output_height;
          }
        }
      }
        */
}

pub fn upsample_bilinear2d_quantized_cpu_with_scales(
    input:         &Tensor,
    output_size:   &[i32],
    align_corners: bool,
    scales_h:      Option<f64>,
    scales_w:      Option<f64>) -> Tensor {

    todo!();
        /*
            TORCH_CHECK(
          output_size.size() == 2,
          "It is expected output_size equals to 2, but got size ",
          output_size.size());

      TORCH_CHECK(
          input.dim() == 4,
          "Non-empty 4D data tensor expected but got a tensor with sizes ",
          input.sizes());

      i64 output_height = output_size[0];
      i64 output_width = output_size[1];

      i64 nbatch = input.size(0);
      i64 channels = input.size(1);
      i64 input_height = input.size(2);
      i64 input_width = input.size(3);
      AT_ASSERT(input_width > 0 && output_width > 0);

      if (input.is_contiguous(MemoryFormat::ChannelsLast)) {
        Tensor output = _empty_affine_quantized(
            {nbatch, channels, output_height, output_width},
            input.options().memory_format(input.suggest_memory_format()),
            input.q_scale(),
            input.q_zero_point(),
            nullopt);

        qupsample_bilinear2d_nhwc_stub(
            input.device().type(),
            output,
            input,
            input_height,
            input_width,
            output_height,
            output_width,
            nbatch,
            channels,
            align_corners,
            scales_h,
            scales_w);
        return output;
      } else {
        Tensor output = _empty_affine_quantized(
            {nbatch, channels, output_height, output_width},
            input.options(),
            input.q_scale(),
            input.q_zero_point());

        auto input_contig = input.contiguous();
        AT_DISPATCH_QINT_TYPES(
            input_contig.scalar_type(), "upsample_bilinear2d", [&] {
              upsample_bilinear2d_out_frame<Scalar>(
                  output,
                  input_contig,
                  input_height,
                  input_width,
                  output_height,
                  output_width,
                  nbatch,
                  channels,
                  align_corners,
                  scales_h,
                  scales_w);
            });
        return output;
      }
        */
}

pub fn upsample_bilinear2d_quantized_cpu(
    input:         &Tensor,
    output_size:   Option<&[i32]>,
    align_corners: bool,
    scale_factors: Option<&[f64]>) -> Tensor {
    
    todo!();
        /*
            auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
      auto scale_h = get_scale_value(scale_factors, 0);
      auto scale_w = get_scale_value(scale_factors, 1);
      return upsample_bilinear2d_quantized_cpu(input, osize, align_corners, scale_h, scale_w);
        */
}

define_dispatch!{qupsample_bilinear2d_nhwc_stub}
