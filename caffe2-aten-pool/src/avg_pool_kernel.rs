crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/AvgPoolKernel.cpp]

pub fn cpu_avg_pool<scalar_t>(
        output:            &Tensor,
        input:             &Tensor,
        kw:                i32,
        kh:                i32,
        dw:                i32,
        dh:                i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {

    todo!();
        /*
            auto input = input_.contiguous();
      auto output = output_.contiguous();

      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();

      i64 numel = output.numel();
      i64 ndim = input.ndimension();
      // treat batch size and channels as one dimension
      i64 channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
      i64 input_height = input.size(-2);
      i64 input_width = input.size(-1);
      i64 output_height = output.size(-2);
      i64 output_width = output.size(-1);

      // parallel on dim N, C, H, W
      parallel_for(0, numel, 0, [&](i64 begin, i64 end) {
        i64 c = 0;
        i64 oh = 0;
        i64 ow = 0;
        data_index_init(begin, c, channels, oh, output_height, ow, output_width);

        for (i64 i = begin; i < end; i++) {
          output_data[i] = static_cast<scalar_t>(0);

          // local pointers
          scalar_t* input_ptr = input_data + c * input_height * input_width;

          // compute the mean of the input image...
          i64 ih0 = oh * dH - padH;
          i64 iw0 = ow * dW - padW;
          i64 ih1 = min(ih0 + kH, input_height + padH);
          i64 iw1 = min(iw0 + kW, input_width + padW);
          i64 pool_size = (ih1 - ih0) * (iw1 - iw0);
          ih0 = max(ih0, (i64) 0);
          iw0 = max(iw0, (i64) 0);
          ih1 = min(ih1, input_height);
          iw1 = min(iw1, input_width);

          if (ih0 >= ih1 || iw0 >= iw1) {
            // move on to next output index
            data_index_step(c, channels, oh, output_height, ow, output_width);
            continue;
          }

          scalar_t sum = 0;

          i64 divide_factor;
          if (divisor_override.has_value()) {
            divide_factor = divisor_override.value();
          } else {
            if(count_include_pad) {
              divide_factor = pool_size;
            } else {
              divide_factor = (ih1 - ih0) * (iw1 - iw0);
            }
          }

          for (i64 ih = ih0; ih < ih1; ih++) {
            for (i64 iw = iw0; iw < iw1; iw++) {
              sum += input_ptr[ih * input_width + iw];
            }
          }
          output_data[i] += sum / divide_factor;

          // move on to next output index
          data_index_step(c, channels, oh, output_height, ow, output_width);
        }
      });

      if (!output_.is_contiguous()) {
        output_.copy_(output);
      }
        */
}

pub fn cpu_avg_pool_channels_last<scalar_t>(
        output:            &Tensor,
        input:             &Tensor,
        kw:                i32,
        kh:                i32,
        dw:                i32,
        dh:                i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {

    todo!();
        /*
            TORCH_CHECK(input_.ndimension() == 4,
                  "average pooling with channels last format supports tensors with 4 dims");
      auto memory_format = MemoryFormat::ChannelsLast;
      auto input = input_.contiguous(memory_format);
      auto output = output_.contiguous(memory_format);

      auto input_data = input.data_ptr<scalar_t>();
      auto output_data = output.data_ptr<scalar_t>();

      i64 nbatch = input.size(0);
      i64 channels = input.size(1);
      i64 input_height = input.size(2);
      i64 input_width = input.size(3);
      i64 output_height = output.size(2);
      i64 output_width = output.size(3);

      using Vec = vec::Vectorized<scalar_t>;
      // parallel on dim N, H, W
      parallel_for(0, nbatch * output_height * output_width, 0, [&](i64 begin, i64 end) {
        i64 n = 0;
        i64 oh = 0;
        i64 ow = 0;
        data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

        i64 size = channels;
        i64 len = size - (size % Vec::size());
        for (i64 i = begin; i < end; i++) {
          // compute the mean of the input image...
          i64 ih0 = oh * dH - padH;
          i64 iw0 = ow * dW - padW;
          i64 ih1 = min(ih0 + kH, input_height + padH);
          i64 iw1 = min(iw0 + kW, input_width + padW);
          i64 pool_size = (ih1 - ih0) * (iw1 - iw0);
          ih0 = max(ih0, (i64) 0);
          iw0 = max(iw0, (i64) 0);
          ih1 = min(ih1, input_height);
          iw1 = min(iw1, input_width);

          i64 divide_factor;
          if (divisor_override.has_value()) {
            divide_factor = divisor_override.value();
          } else {
            if(count_include_pad) {
              divide_factor = pool_size;
            } else {
              divide_factor = (ih1 - ih0) * (iw1 - iw0);
            }
          }

          scalar_t* out = output_data + i * channels;

          // Pass I: zero the out lane
          i64 d1 = 0;
          for (; d1 < len; d1 += Vec::size()) {
            Vec out_vec = Vec(scalar_t(0));
            out_vec.store(out + d1);
          }
          for (; d1 < size; d1++) {
            out[d1] = scalar_t(0);
          }

          if (ih0 >= ih1 || iw0 >= iw1) {
            // move on to next output index
            data_index_step(n, nbatch, oh, output_height, ow, output_width);
            continue;
          }

          // Pass II: compute local sum
          for (i64 ih = ih0; ih < ih1; ih++) {
            for (i64 iw = iw0; iw < iw1; iw++) {
              scalar_t* in = input_data + n * input_height * input_width * channels +
                  ih * input_width * channels + iw * channels;

              i64 d2 = 0;
              for (; d2 < len; d2 += Vec::size()) {
                Vec out_vec = Vec::loadu(out + d2) + Vec::loadu(in + d2);
                out_vec.store(out + d2);
              }
              for (; d2 < size; d2++) {
                out[d2] += in[d2];
              }
            }
          }

          // Pass III: compute local average
          i64 d3 = 0;
          for (; d3 < len; d3 += Vec::size()) {
            Vec out_vec = Vec::loadu(out + d3) / Vec(scalar_t(divide_factor));
            out_vec.store(out + d3);
          }
          for (; d3 < size; d3++) {
            out[d3] = out[d3] / divide_factor;
          }

          // move on to next output index
          data_index_step(n, nbatch, oh, output_height, ow, output_width);
        }
      });

      if (!output_.is_contiguous(memory_format)) {
        output_.copy_(output);
      }
        */
}

pub fn cpu_avg_pool_backward<scalar_t>(
        grad_input:        &Tensor,
        grad_output:       &Tensor,
        kw:                i32,
        kh:                i32,
        dw:                i32,
        dh:                i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {

    todo!();
        /*
            auto grad_output = grad_output_.contiguous();
      auto grad_input = grad_input_.contiguous();

      auto grad_output_data = grad_output.data_ptr<scalar_t>();
      auto grad_input_data = grad_input.data_ptr<scalar_t>();

      i64 ndim = grad_output.ndimension();
      // treat batch size and channels as one dimension
      i64 channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
      i64 input_height = grad_input.size(-2);
      i64 input_width = grad_input.size(-1);
      i64 output_height = grad_output.size(-2);
      i64 output_width = grad_output.size(-1);

      // parallel on dim of N, C
      parallel_for(0, channels, 0, [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          scalar_t* grad_input_ptr = grad_input_data + c * input_height * input_width;
          scalar_t* grad_output_ptr = grad_output_data + c * output_height * output_width;

          for (i64 oh = 0; oh < output_height; oh++) {
            for (i64 ow = 0; ow < output_width; ow++) {
              i64 ih0 = oh * dH - padH;
              i64 iw0 = ow * dW - padW;
              i64 ih1 = min(ih0 + kH, input_height + padH);
              i64 iw1 = min(iw0 + kW, input_width + padW);
              i64 pool_size = (ih1 - ih0) * (iw1 - iw0);
              ih0 = max(ih0, (i64) 0);
              iw0 = max(iw0, (i64) 0);
              ih1 = min(ih1, input_height);
              iw1 = min(iw1, input_width);

              i64 divide_factor;
              if (divisor_override.has_value()) {
                divide_factor = divisor_override.value();
              } else {
                if(count_include_pad) {
                  divide_factor = pool_size;
                } else {
                  divide_factor = (ih1 - ih0) * (iw1 - iw0);
                }
              }

              scalar_t grad_delta = grad_output_ptr[oh * output_width + ow] / divide_factor;
              for (i64 ih = ih0; ih < ih1; ih++) {
                for (i64 iw = iw0; iw < iw1; iw++) {
                  grad_input_ptr[ih * input_width + iw] += grad_delta;
                }
              }
            }
          }
        }
      });

      if (!grad_input_.is_contiguous()) {
        grad_input_.copy_(grad_input);
      }
        */
}

pub fn cpu_avg_pool_backward_channels_last<scalar_t>(
        grad_input:        &Tensor,
        grad_output:       &Tensor,
        kw:                i32,
        kh:                i32,
        dw:                i32,
        dh:                i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {

    todo!();
        /*
            auto memory_format = MemoryFormat::ChannelsLast;
      auto grad_input = grad_input_.contiguous(memory_format);
      auto grad_output = grad_output_.contiguous(memory_format);

      auto grad_input_data = grad_input.data_ptr<scalar_t>();
      auto grad_output_data = grad_output.data_ptr<scalar_t>();

      i64 nbatch = grad_input.size(0);
      i64 channels = grad_input.size(1);
      i64 input_height = grad_input.size(2);
      i64 input_width = grad_input.size(3);
      i64 output_height = grad_output.size(2);
      i64 output_width = grad_output.size(3);

      using Vec = vec::Vectorized<scalar_t>;
      // parallel on dim N
      parallel_for(0, nbatch, 0, [&](i64 begin, i64 end) {
        for (i64 n = begin; n < end; n++) {
          scalar_t* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;
          scalar_t* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;

          for (i64 oh = 0; oh < output_height; oh++) {
            for (i64 ow = 0; ow < output_width; ow++) {
              i64 ih0 = oh * dH - padH;
              i64 iw0 = ow * dW - padW;
              i64 ih1 = min(ih0 + kH, input_height + padH);
              i64 iw1 = min(iw0 + kW, input_width + padW);
              i64 pool_size = (ih1 - ih0) * (iw1 - iw0);
              ih0 = max(ih0, (i64) 0);
              iw0 = max(iw0, (i64) 0);
              ih1 = min(ih1, input_height);
              iw1 = min(iw1, input_width);

              i64 divide_factor;
              if (divisor_override.has_value()) {
                divide_factor = divisor_override.value();
              } else {
                if(count_include_pad) {
                  divide_factor = pool_size;
                } else {
                   divide_factor = (ih1 - ih0) * (iw1 - iw0);
                }
              }

              scalar_t* gout = grad_output_ptr + oh * output_width * channels + ow * channels;
              i64 size = channels;
              i64 len = size - (size % Vec::size());
              for (i64 ih = ih0; ih < ih1; ih++) {
                for (i64 iw = iw0; iw < iw1; iw++) {
                  scalar_t* gin = grad_input_ptr + ih * input_width * channels + iw * channels;

                  i64 d = 0;
                  for (; d < len; d += Vec::size()) {
                    Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(scalar_t(divide_factor));
                    gin_vec.store(gin + d);
                  }
                  for (; d < size; d++) {
                    gin[d] += gout[d] / divide_factor;
                  }
                }
              }
            }
          }
        }
      });

      if (!grad_input_.is_contiguous(memory_format)) {
        grad_input_.copy_(grad_input);
      }
        */
}

pub fn avg_pool2d_kernel_impl(
        output:            &Tensor,
        input:             &Tensor,
        kw:                i32,
        kh:                i32,
        dw:                i32,
        dh:                i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {
    
    todo!();
        /*
            switch (input.suggest_memory_format()) {
        case MemoryFormat::Contiguous: {
          AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, input.scalar_type(), "avg_pool2d", [&] {
            cpu_avg_pool<scalar_t>(output, input, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
          });
          break;
        }
        case MemoryFormat::ChannelsLast: {
          AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, input.scalar_type(), "avg_pool2d_channels_last", [&] {
            cpu_avg_pool_channels_last<scalar_t>(output, input, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
          });
          break;
        }
        default:
          TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
        */
}


pub fn avg_pool2d_backward_kernel_impl(
        grad_input:        &Tensor,
        grad_output:       &Tensor,
        kw:                i32,
        kh:                i32,
        dw:                i32,
        dh:                i32,
        padw:              i32,
        padh:              i32,
        count_include_pad: bool,
        divisor_override:  Option<i64>)  {
    
    todo!();
        /*
            switch (grad_output.suggest_memory_format()) {
        case MemoryFormat::Contiguous: {
          AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, grad_output.scalar_type(), "avg_pool2d_backward", [&] {
            cpu_avg_pool_backward<scalar_t>(grad_input, grad_output, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
          });
          break;
        }
        case MemoryFormat::ChannelsLast: {
          AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::Long, grad_output.scalar_type(), "avg_pool2d_backward_channels_last", [&] {
            cpu_avg_pool_backward_channels_last<scalar_t>(grad_input, grad_output, kW, kH, dW, dH, padW, padH, count_include_pad, divisor_override);
          });
          break;
        }
        default:
          TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
        */
}

register_dispatch!{avg_pool2d_kernel          , &avg_pool2d_kernel_impl}
register_dispatch!{avg_pool2d_backward_kernel , &avg_pool2d_backward_kernel_impl}
