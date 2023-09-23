crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/UpSampleMoreKernel.cpp]

pub type scale_t = Vec<Option<f64>>;

pub fn cpu_upsample_linear_backward<Scalar, scale_type>(
        grad_input:    &Tensor,
        grad_output:   &Tensor,
        align_corners: bool,
        scales:        &ScaleType)  {

    todo!();
        /*
            TORCH_CHECK(grad_input_.dtype() == grad_output_.dtype(), "expected dtype ", grad_output_.dtype(),
                  " for `grad_input` but got dtype ", grad_input_.dtype());

      auto grad_output = grad_output_.contiguous();
      auto grad_input = grad_input_.contiguous();

      auto grad_output_data = grad_output.data_ptr<Scalar>();
      auto grad_input_data = grad_input.data_ptr<Scalar>();
      auto input_sizes = grad_input.sizes().vec();
      auto output_sizes = grad_output.sizes().vec();
      auto ndim = input_sizes.size();

      // treat nbatch and channels as one dimension
      i64 channels = input_sizes[0] * input_sizes[1];
      i64 input_depth = (ndim == 5) ? input_sizes[2] : 1;
      i64 output_depth = (ndim == 5) ? output_sizes[2] : 1;
      i64 input_height = (ndim >= 4) ? input_sizes[ndim - 2] : 1;
      i64 output_height = (ndim >= 4) ? output_sizes[ndim - 2] : 1;
      i64 input_width = input_sizes[ndim - 1];
      i64 output_width = output_sizes[ndim - 1];

      i64 output_slice_size = output_depth * output_height * output_width;

      auto loop1d = [&](i64 begin, i64 end) {
        const Scalar width_scale = area_pixel_compute_scale<Scalar>(
            input_width, output_width, align_corners, scales[0]);

        auto input_indexr = [=](i64 c, i64 w) {
          return grad_input_data + c * input_width + w;
        };

        i64 iw0, iw1;
        Scalar w0lambda, w1lambda;
        for (i64 c = begin; c < end; c++){
          for (i64 ow = 0; ow < output_width; ow++) {
            compute_source_index_and_lambda(
                iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
            Scalar grad_output_value = grad_output_data[c * output_slice_size + ow];
            *input_indexr(c, iw0) += w0lambda * grad_output_value; /* i0 */
            *input_indexr(c, iw1) += w1lambda * grad_output_value; /* i1*/
          }
        }
      };

      auto loop2d = [&](i64 begin, i64 end) {
        const Scalar height_scale = area_pixel_compute_scale<Scalar>(
            input_height, output_height, align_corners, scales[0]);
        const Scalar width_scale = area_pixel_compute_scale<Scalar>(
            input_width, output_width, align_corners, scales[1]);

        auto input_indexr = [=](i64 c, i64 h, i64 w){
          return grad_input_data + c * input_height * input_width + h * input_width + w;
        };

        i64 ih0, ih1, iw0, iw1;
        Scalar h0lambda, h1lambda, w0lambda, w1lambda;
        for (i64 c = begin; c < end; c++) {
          for (i64 oh = 0; oh < output_height; oh++) {
            compute_source_index_and_lambda(
                ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
            for (i64 ow = 0; ow < output_width; ow++) {
              compute_source_index_and_lambda(
                  iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
              Scalar grad_output_value = grad_output_data[c * output_slice_size + oh * output_width + ow];
              *input_indexr(c, ih0, iw0) += h0lambda * w0lambda * grad_output_value; /* i00 */
              *input_indexr(c, ih0, iw1) += h0lambda * w1lambda * grad_output_value; /* i01 */
              *input_indexr(c, ih1, iw0) += h1lambda * w0lambda * grad_output_value; /* i10 */
              *input_indexr(c, ih1, iw1) += h1lambda * w1lambda * grad_output_value; /* i11 */
            }
          }
        }
      };

      auto loop3d = [&](i64 begin, i64 end) {
        const Scalar depth_scale = area_pixel_compute_scale<Scalar>(
            input_depth, output_depth, align_corners, scales[0]);
        const Scalar height_scale = area_pixel_compute_scale<Scalar>(
            input_height, output_height, align_corners, scales[1]);
        const Scalar width_scale = area_pixel_compute_scale<Scalar>(
            input_width, output_width, align_corners, scales[2]);

        auto input_indexr = [=](i64 c, i64 d, i64 h, i64 w) {
          return grad_input_data + c * input_depth * input_height * input_width +
              d * input_height * input_width + h * input_width + w;
        };

        i64 id0, id1, ih0, ih1, iw0, iw1;
        Scalar d0lambda, d1lambda, h0lambda, h1lambda, w0lambda, w1lambda;
        for (i64 c = begin; c < end; c++) {
          for (i64 od = 0; od < output_depth; od++) {
            compute_source_index_and_lambda(
                id0, id1, d0lambda, d1lambda, depth_scale, od, input_depth, output_depth, align_corners);
            for (i64 oh = 0; oh < output_height; oh++) {
              compute_source_index_and_lambda(
                  ih0, ih1, h0lambda, h1lambda, height_scale, oh, input_height, output_height, align_corners);
              for (i64 ow = 0; ow < output_width; ow++) {
                compute_source_index_and_lambda(
                    iw0, iw1, w0lambda, w1lambda, width_scale, ow, input_width, output_width, align_corners);
                Scalar grad_output_value = grad_output_data[c * output_slice_size +
                    od *  output_height * output_width + oh * output_width + ow];
                *input_indexr(c, id0, ih0, iw0) += d0lambda * h0lambda * w0lambda * grad_output_value; /* i000 */
                *input_indexr(c, id0, ih0, iw1) += d0lambda * h0lambda * w1lambda * grad_output_value; /* i001 */
                *input_indexr(c, id0, ih1, iw0) += d0lambda * h1lambda * w0lambda * grad_output_value; /* i010 */
                *input_indexr(c, id0, ih1, iw1) += d0lambda * h1lambda * w1lambda * grad_output_value; /* i011 */
                *input_indexr(c, id1, ih0, iw0) += d1lambda * h0lambda * w0lambda * grad_output_value; /* i100 */
                *input_indexr(c, id1, ih0, iw1) += d1lambda * h0lambda * w1lambda * grad_output_value; /* i101 */
                *input_indexr(c, id1, ih1, iw0) += d1lambda * h1lambda * w0lambda * grad_output_value; /* i110 */
                *input_indexr(c, id1, ih1, iw1) += d1lambda * h1lambda * w1lambda * grad_output_value; /* i111 */
              }
            }
          }
        }
      };

      if (ndim == 3) {
        // upsample linear 1d
        parallel_for(0, channels, internal::GRAIN_SIZE / output_slice_size / 2, loop1d);
      } else if (ndim == 4) {
        // upsample bilinear 2d
        parallel_for(0, channels, internal::GRAIN_SIZE / output_slice_size / 4, loop2d);
      } else {
        // upsample trilinear 3d
        TORCH_INTERNAL_ASSERT(ndim == 5);
        parallel_for(0, channels, internal::GRAIN_SIZE / output_slice_size / 8, loop3d);
      }

      if (!grad_input_.is_contiguous()) {
        grad_input_.copy_(grad_input);
      }
        */
}

pub fn upsample_linear1d_backward_kernel_impl(
        grad_input:    &Tensor,
        grad_output:   &Tensor,
        align_corners: bool,
        scales_w:      Option<f64>)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_linear1d_backward", [&] {
        cpu_upsample_linear_backward<Scalar, scale_t>(grad_input, grad_output, align_corners, {scales_w});
      });
        */
}

pub fn upsample_bilinear2d_backward_kernel_impl(
        grad_input:    &Tensor,
        grad_output:   &Tensor,
        align_corners: bool,
        scales_h:      Option<f64>,
        scales_w:      Option<f64>)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_bilinear2d_backward", [&] {
        cpu_upsample_linear_backward<Scalar, scale_t>(grad_input, grad_output, align_corners, {scales_h, scales_w});
      });
        */
}

pub fn upsample_trilinear3d_backward_kernel_impl(
        grad_input:    &Tensor,
        grad_output:   &Tensor,
        align_corners: bool,
        scales_d:      Option<f64>,
        scales_h:      Option<f64>,
        scales_w:      Option<f64>)  {
    
    todo!();
        /*
            AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "upsample_trilinear3d_backward", [&] {
        cpu_upsample_linear_backward<Scalar, scale_t>(grad_input, grad_output, align_corners, {scales_d, scales_h, scales_w});
      });
        */
}

register_dispatch!{upsample_linear1d_backward_kernel    , &upsample_linear1d_backward_kernel_impl}
register_dispatch!{upsample_bilinear2d_backward_kernel  , &upsample_bilinear2d_backward_kernel_impl}
register_dispatch!{upsample_trilinear3d_backward_kernel , &upsample_trilinear3d_backward_kernel_impl}
