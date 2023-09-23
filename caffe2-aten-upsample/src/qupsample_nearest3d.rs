// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qupsample_nearest3d.cpp]

/**
  | native functions for the native_functions.yaml
  |
  */
pub fn upsample_nearest3d_out_frame<Scalar>(
    odata:         *mut Scalar,
    idata:         *mut Scalar,
    input_depth:   i64,
    input_height:  i64,
    input_width:   i64,
    output_depth:  i64,
    output_height: i64,
    output_width:  i64,
    nbatch:        i64,
    channels:      i64,
    scales_d:      Option<f64>,
    scales_h:      Option<f64>,
    scales_w:      Option<f64>)  {

    todo!();
    /*
       float depth_scale = compute_scales_value<float>(scales_d, input_depth, output_depth);
      float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
      float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

      channels = channels * nbatch;
      auto* i_p = reinterpret_cast<typename Scalar::underlying*>(idata);
      auto* o_p = reinterpret_cast<typename Scalar::underlying*>(odata);

      // special case: just copy
      if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
        memcpy(o_p, i_p, channels * input_depth * input_height * input_width * sizeof(typename Scalar::underlying));
        return;
      }

      for (i64 d2 = 0; d2 < output_depth; ++d2) {
        const i64 d1 =
              nearest_neighbor_compute_source_index(depth_scale, d2, input_depth);

        for (i64 h2 = 0; h2 < output_height; ++h2) {
          const i64 h1 =
              nearest_neighbor_compute_source_index(height_scale, h2, input_height);

          for (i64 w2 = 0; w2 < output_width; ++w2) {
            const i64 w1 =
                nearest_neighbor_compute_source_index(width_scale, w2, input_width);

            const auto* pos1 = &i_p[d1 * input_height * input_width + h1 * input_width + w1];
            auto* pos2 = &o_p[d2 * output_height * output_width + h2 * output_width + w2];

            for (i64 c = 0; c < channels; ++c) {
              pos2[0] = pos1[0];
              pos1 += input_depth * input_height * input_width;
              pos2 += output_depth * output_height * output_width;
            }
          }
        }
      }
        */
}

pub fn upsample_nearest3d_out_frame_nhwc<Scalar>(
    odata:         *mut Scalar,
    idata:         *mut Scalar,
    input_depth:   i64,
    input_height:  i64,
    input_width:   i64,
    output_depth:  i64,
    output_height: i64,
    output_width:  i64,
    nbatch:        i64,
    channels:      i64,
    scales_d:      Option<f64>,
    scales_h:      Option<f64>,
    scales_w:      Option<f64>)  {

    todo!();
        /*
            float depth_scale = compute_scales_value<float>(scales_d, input_depth, output_depth);
      float height_scale = compute_scales_value<float>(scales_h, input_height, output_height);
      float width_scale = compute_scales_value<float>(scales_w, input_width, output_width);

      for (const auto b : irange(nbatch)) {
        auto* i_p = reinterpret_cast<typename Scalar::underlying*>(idata + b * input_depth * input_height * input_width * channels);
        auto* o_p = reinterpret_cast<typename Scalar::underlying*>(odata + b * output_depth * output_height * output_width * channels);
        // special case: just copy
        if (input_depth == output_depth && input_height == output_height && input_width == output_width) {
          memcpy(o_p, i_p, channels * input_depth * input_height * input_width * sizeof(typename Scalar::underlying));
          return;
        }

        for (i64 d2 = 0; d2 < output_depth; ++d2) {
          const i64 d1 =
              nearest_neighbor_compute_source_index(depth_scale, d2, input_depth);
          for (i64 h2 = 0; h2 < output_height; ++h2) {
            const i64 h1 =
                nearest_neighbor_compute_source_index(height_scale, h2, input_height);

            for (i64 w2 = 0; w2 < output_width; ++w2) {
              const i64 w1 =
                  nearest_neighbor_compute_source_index(width_scale, w2, input_width);

              const auto* pos1 = &i_p[(d1 * input_height * input_width + h1 * input_width + w1)*channels];
              auto* pos2 = &o_p[(d2 * output_height * output_width + h2 * output_width + w2)*channels];
              memcpy(pos2, pos1, channels * sizeof(typename Scalar::underlying));
            }
          }
        }
      }
        */
}

pub fn upsample_nearest3d_quantized_cpu_with_scales(
    input:       &Tensor,
    output_size: &[i32],
    scales_d:    Option<f64>,
    scales_h:    Option<f64>,
    scales_w:    Option<f64>) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(
          output_size.size() == 3,
          "It is expected output_size equals to 3, but got size ",
          output_size.size());

      TORCH_CHECK(
          input.numel() != 0 && input.dim() == 5,
          "Non-empty 5D data tensor expected but got a tensor with sizes ",
          input.sizes());

      i64 output_depth = output_size[0];
      i64 output_height = output_size[1];
      i64 output_width = output_size[2];

      i64 nbatch = input.size(0);
      i64 channels = input.size(1);
      i64 input_depth = input.size(2);
      i64 input_height = input.size(3);
      i64 input_width = input.size(4);
      AT_ASSERT(input_width > 0 && output_width > 0);
      if (input.is_contiguous(MemoryFormat::ChannelsLast3d)) {
        Tensor output = _empty_affine_quantized(
            {nbatch, channels, output_depth, output_height, output_width},
            input.options().memory_format(input.suggest_memory_format()),
            input.q_scale(),
            input.q_zero_point(),
            nullopt);

        AT_DISPATCH_QINT_TYPES(input.scalar_type(), "upsample_nearest3d", [&] {
          auto* idata = static_cast<Scalar*>(input.data_ptr());
          auto* odata = static_cast<Scalar*>(output.data_ptr());
          upsample_nearest3d_out_frame_nhwc<Scalar>(
              odata,
              idata,
              input_depth,
              input_height,
              input_width,
              output_depth,
              output_height,
              output_width,
              nbatch,
              channels,
              scales_d,
              scales_h,
              scales_w);
        });
        return output;
      } else {
        Tensor output = _empty_affine_quantized(
            {nbatch, channels, output_depth, output_height, output_width},
            input.options(),
            input.q_scale(),
            input.q_zero_point());

        auto input_contig = input.contiguous();

        AT_DISPATCH_QINT_TYPES(input_contig.scalar_type(), "upsample_nearest3d", [&] {
          auto* idata = static_cast<Scalar*>(input_contig.data_ptr());
          auto* odata = static_cast<Scalar*>(output.data_ptr());
          upsample_nearest3d_out_frame<Scalar>(
              odata,
              idata,
              input_depth,
              input_height,
              input_width,
              output_depth,
              output_height,
              output_width,
              nbatch,
              channels,
              scales_d,
              scales_h,
              scales_w);
        });
        return output;
      }
        */
}

pub fn upsample_nearest3d_quantized_cpu(
    input:         &Tensor,
    output_size:   Option<&[i32]>,
    scale_factors: Option<&[f64]>) -> Tensor {
    
    todo!();
        /*
            auto osize = compute_output_size(input.sizes(), output_size, scale_factors);
      auto scale_d = get_scale_value(scale_factors, 0);
      auto scale_h = get_scale_value(scale_factors, 1);
      auto scale_w = get_scale_value(scale_factors, 2);
      return upsample_nearest3d_quantized_cpu(input, osize, scale_d, scale_h, scale_w);
        */
}
