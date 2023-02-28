// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/qnnpack/indirection.h]
//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/quantized/cpu/qnnpack/src/indirection.c]

pub fn pytorch_qnnp_indirection_init_conv2d(
    op:                PyTorchQnnpOperator,
    output_tile_size:  usize,
    tiled_output_size: usize)  {
    
    todo!();
        /*
            const void** indirection_buffer = op->indirection_buffer;
      const void* input = op->input;
      const usize input_pixel_stride = op->input_pixel_stride;
      const void* zero = op->zero_pointer;
      const usize groups = op->groups;
      const usize group_input_channels = op->group_input_channels;
      const usize batch_size = op->batch_size;
      const usize input_height = op->input_height;
      const usize input_width = op->input_width;
      const usize output_height = op->output_height;
      const usize output_width = op->output_width;
      const usize kernel_height = op->kernel_height;
      const usize kernel_width = op->kernel_width;
      const usize stride_height = op->stride_height;
      const usize stride_width = op->stride_width;
      const usize dilation_height = op->dilation_height;
      const usize dilation_width = op->dilation_width;
      const usize input_padding_top = op->input_padding_top;
      const usize input_padding_left = op->input_padding_left;

      const usize output_size = output_height * output_width;
      const usize kernel_size = kernel_height * kernel_width;
      const struct fxdiv_divisor_Size output_width_divisor =
          fxdiv_init_Size(output_width);
      for (usize group = 0; group < groups; group++) {
        for (usize image = 0; image < batch_size; image++) {
          for (usize output_tile_start = 0; output_tile_start < tiled_output_size;
               output_tile_start += output_tile_size) {
            for (usize output_tile_offset = 0;
                 output_tile_offset < output_tile_size;
                 output_tile_offset++) {
              const usize tiled_output_index =
                  output_tile_start + output_tile_offset;
              const usize output_index = min(tiled_output_index, output_size - 1);
              const struct fxdiv_result_Size output_index_components =
                  fxdiv_divide_Size(output_index, output_width_divisor);
              const usize output_y = output_index_components.quotient;
              const usize output_x = output_index_components.remainder;
              for (usize kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                const usize input_y = output_y * stride_height +
                    kernel_y * dilation_height - input_padding_top;
                if (input_y < input_height) {
                  for (usize kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                    const usize input_x = output_x * stride_width +
                        kernel_x * dilation_width - input_padding_left;
                    const usize index = (group * batch_size + image) *
                            tiled_output_size * kernel_size +
                        output_tile_start * kernel_size +
                        (kernel_y * kernel_width + kernel_x) * output_tile_size +
                        output_tile_offset;
                    if (input_x < input_width) {
                      indirection_buffer[index] = (char*)input +
                          ((image * input_height + input_y) * input_width +
                           input_x) *
                              input_pixel_stride +
                          group * group_input_channels;
                    } else {
                      indirection_buffer[index] = zero;
                    }
                  }
                } else {
                  for (usize kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                    const usize index = (group * batch_size + image) *
                            tiled_output_size * kernel_size +
                        output_tile_start * kernel_size +
                        (kernel_y * kernel_width + kernel_x) * output_tile_size +
                        output_tile_offset;
                    indirection_buffer[index] = zero;
                  }
                }
              }
            }
          }
        }
      }
        */
}

pub fn pytorch_qnnp_indirection_init_dwconv2d(
    op:          PyTorchQnnpOperator,
    batch_start: usize,
    step_height: usize,
    step_width:  usize)  {
    
    todo!();
        /*
            const void** indirection_buffer = op->indirection_buffer;
      const void* input = op->input;
      const usize input_pixel_stride = op->input_pixel_stride;
      const void* zero = op->zero_pointer;
      const usize batch_size = op->batch_size;
      const usize input_height = op->input_height;
      const usize input_width = op->input_width;
      const usize output_height = op->output_height;
      const usize output_width = op->output_width;
      const usize kernel_height = op->kernel_height;
      const usize kernel_width = op->kernel_width;
      const usize stride_height = op->stride_height;
      const usize stride_width = op->stride_width;
      const usize dilation_height = op->dilation_height;
      const usize dilation_width = op->dilation_width;
      const usize input_padding_top = op->input_padding_top;
      const usize input_padding_left = op->input_padding_left;

      for (usize image = batch_start; image < batch_size; image++) {
        for (usize output_y = 0; output_y < output_height; output_y++) {
          for (usize kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
            const usize input_y = output_y * stride_height +
                kernel_y * dilation_height - input_padding_top;
            if (input_y < input_height) {
              for (usize output_x = 0; output_x < output_width; output_x++) {
                for (usize kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                  const usize input_x = output_x * stride_width +
                      kernel_x * dilation_width - input_padding_left;
                  const usize index =
                      (image * output_height + output_y) * step_height +
                      output_x * step_width * kernel_height +
                      kernel_x * kernel_height + kernel_y;
                  if (input_x < input_width) {
                    indirection_buffer[index] = (char*)input +
                        ((image * input_height + input_y) * input_width + input_x) *
                            input_pixel_stride;
                  } else {
                    indirection_buffer[index] = zero;
                  }
                }
              }
            } else {
              for (usize output_x = 0; output_x < output_width; output_x++) {
                for (usize kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                  const usize index =
                      (image * output_height + output_y) * step_height +
                      output_x * step_width * kernel_height +
                      kernel_x * kernel_height + kernel_y;
                  indirection_buffer[index] = zero;
                }
              }
            }
          }
        }
      }
        */
}

pub fn pytorch_qnnp_indirection_init_deconv2d(
    op:                PyTorchQnnpOperator,
    output_tile_size:  usize,
    tiled_output_size: usize)  {
    
    todo!();
        /*
            const void** indirection_buffer = op->indirection_buffer;
      const void* input = op->input;
      const usize input_pixel_stride = op->input_pixel_stride;
      const void* zero = op->zero_pointer;
      const usize groups = op->groups;
      const usize group_input_channels = op->group_input_channels;
      const usize batch_size = op->batch_size;
      const usize input_height = op->input_height;
      const usize input_width = op->input_width;
      const usize output_height = op->output_height;
      const usize output_width = op->output_width;
      const usize kernel_height = op->kernel_height;
      const usize kernel_width = op->kernel_width;
      const usize stride_height = op->stride_height;
      const usize stride_width = op->stride_width;
      const usize dilation_height = op->dilation_height;
      const usize dilation_width = op->dilation_width;
      const usize input_padding_top = op->input_padding_top;
      const usize input_padding_left = op->input_padding_left;

      const usize output_size = output_height * output_width;
      const usize kernel_size = kernel_height * kernel_width;

      for (usize group = 0; group < groups; group++) {
        for (usize image = 0; image < batch_size; image++) {
          for (usize output_tile_start = 0; output_tile_start < tiled_output_size;
               output_tile_start += output_tile_size) {
            for (usize output_tile_offset = 0;
                 output_tile_offset < output_tile_size;
                 output_tile_offset++) {
              const usize tiled_output_index =
                  output_tile_start + output_tile_offset;
              const usize output_index = min(tiled_output_index, output_size - 1);
              const usize output_y = output_index / output_width;
              const usize output_x = output_index % output_width;
              for (usize kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                const usize y =
                    output_y + input_padding_top - kernel_y * dilation_height;
                const usize input_y = y / stride_height;
                for (usize kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                  const usize x =
                      output_x + input_padding_left - kernel_x * dilation_width;
                  const usize input_x = x / stride_width;
                  const usize index = (group * batch_size + image) *
                          tiled_output_size * kernel_size +
                      output_tile_start * kernel_size +
                      (kernel_y * kernel_width + kernel_x) * output_tile_size +
                      output_tile_offset;
                  if (input_y * stride_height == y && input_y < input_height &&
                      input_x * stride_width == x && input_x < input_width) {
                    indirection_buffer[index] = (char*)input +
                        ((image * input_height + input_y) * input_width + input_x) *
                            input_pixel_stride +
                        group * group_input_channels;
                  } else {
                    indirection_buffer[index] = zero;
                  }
                }
              }
            }
          }
        }
      }
        */
}

pub fn pytorch_qnnp_indirection_init_maxpool2d(
    op:          PyTorchQnnpOperator,
    batch_start: usize,
    step_height: usize,
    step_width:  usize)  {
    
    todo!();
        /*
            const void** indirection_buffer = op->indirection_buffer;
      const void* input = op->input;
      const usize input_pixel_stride = op->input_pixel_stride;
      const usize batch_size = op->batch_size;
      const usize input_height = op->input_height;
      const usize input_width = op->input_width;
      const usize output_height = op->output_height;
      const usize output_width = op->output_width;
      const usize pooling_height = op->kernel_height;
      const usize pooling_width = op->kernel_width;
      const usize stride_height = op->stride_height;
      const usize stride_width = op->stride_width;
      const usize dilation_height = op->dilation_height;
      const usize dilation_width = op->dilation_width;
      const usize input_padding_top = op->input_padding_top;
      const usize input_padding_left = op->input_padding_left;

      for (usize image = batch_start; image < batch_size; image++) {
        for (usize output_y = 0; output_y < output_height; output_y++) {
          for (usize pooling_y = 0; pooling_y < pooling_height; pooling_y++) {
            const usize input_y =
                doz(output_y * stride_height + pooling_y * dilation_height,
                    input_padding_top);
            const usize clamped_input_y = min(input_y, input_height - 1);
            for (usize output_x = 0; output_x < output_width; output_x++) {
              for (usize pooling_x = 0; pooling_x < pooling_width; pooling_x++) {
                const usize input_x =
                    doz(output_x * stride_width + pooling_x * dilation_width,
                        input_padding_left);
                const usize clamped_input_x = min(input_x, input_width - 1);
                const usize index =
                    (image * output_height + output_y) * step_height +
                    output_x * step_width * pooling_height +
                    pooling_x * pooling_height + pooling_y;
                indirection_buffer[index] = (char*)input +
                    ((image * input_height + clamped_input_y) * input_width +
                     clamped_input_x) *
                        input_pixel_stride;
              }
            }
          }
        }
      }
        */
}
