crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/AdaptiveAvgPoolKernel.cpp]

pub fn cpu_adaptive_avg_pool<Scalar>(
    output:      &mut Tensor,
    input:       &Tensor,
    output_size: &[i32])  {

    todo!();
        /*
            auto input = input_.contiguous();
      auto output = output_.contiguous();

      auto input_data = input.data_ptr<Scalar>();
      auto output_data = output.data_ptr<Scalar>();

      i64 ndim = input.ndimension();
      // treat batch size and channels as one dimension
      i64 channels = ndim == 3 ? input.size(0) : input.size(0) * input.size(1);
      i64 input_height = input.size(-2);
      i64 input_width = input.size(-1);
      i64 output_height = output_size[0];
      i64 output_width = output_size[1];

      // parallel on dim of N, C
      at::parallel_for(0, channels, 0, [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          Scalar* input_ptr = input_data + c * input_height * input_width;
          Scalar* output_ptr = output_data + c * output_height * output_width;

          for (i64 oh = 0; oh < output_height; oh++) {
            i64 ih0 = start_index(oh, output_height, input_height);
            i64 ih1 = end_index(oh, output_height, input_height);
            i64 kh = ih1 - ih0;

            for (i64 ow = 0; ow < output_width; ow++) {
              i64 iw0 = start_index(ow, output_width, input_width);
              i64 iw1 = end_index(ow, output_width, input_width);
              i64 kw = iw1 - iw0;

              // compute local average
              Scalar sum = 0;
              for (i64 ih = ih0; ih < ih1; ih++) {
                for (i64 iw = iw0; iw < iw1; iw++) {
                  sum += input_ptr[ih * input_width + iw];
                }
              }
              output_ptr[oh * output_width + ow] = sum / kh / kw;
            }
          }
        }
      });

      if (!output_.is_contiguous()) {
        output_.copy_(output);
      }
        */
}

pub fn cpu_adaptive_avg_pool_channels_last<Scalar>(
    output:      &mut Tensor,
    input:       &Tensor,
    output_size: &[i32])  {

    todo!();
        /*
            auto memory_format = at::MemoryFormat::ChannelsLast;
      auto input = input_.contiguous(memory_format);
      auto output = output_.contiguous(memory_format);

      auto input_data = input.data_ptr<Scalar>();
      auto output_data = output.data_ptr<Scalar>();

      i64 nbatch = input.size(0);
      i64 channels = input.size(1);
      i64 input_height = input.size(2);
      i64 input_width = input.size(3);
      i64 output_height = output_size[0];
      i64 output_width = output_size[1];

      using Vec = vec::Vectorized<Scalar>;
      // parallel on dim N, H, W
      at::parallel_for(0, nbatch * output_height * output_width, 0, [&](i64 begin, i64 end) {
        i64 n = 0;
        i64 oh = 0;
        i64 ow = 0;
        data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

        for (i64 i = begin; i < end; i++) {
          i64 ih0 = start_index(oh, output_height, input_height);
          i64 ih1 = end_index(oh, output_height, input_height);
          i64 kh = ih1 - ih0;

          i64 iw0 = start_index(ow, output_width, input_width);
          i64 iw1 = end_index(ow, output_width, input_width);
          i64 kw = iw1 - iw0;

          Scalar* out = output_data + i * channels;
          i64 size = channels;

          // Note: For oridinary usage scenario, each out lane should
          //   fit in L1 cache; otherwise consider block dim C.
          // Pass I: zero the out lane
          i64 d1 = 0;
          for (; d1 < size - (size % Vec::size()); d1 += Vec::size()) {
            Vec out_vec = Vec(Scalar(0));
            out_vec.store(out + d1);
          }
          for (; d1 < size; d1++) {
            out[d1] = Scalar(0);
          }
          // Pass II: compute local sum
          for (i64 ih = ih0; ih < ih1; ih++) {
            for (i64 iw = iw0; iw < iw1; iw++) {
              Scalar* in = input_data + n * input_height * input_width * channels +
                  ih * input_width * channels + iw * channels;

              i64 d2 = 0;
              for (; d2 < size - (size % Vec::size()); d2 += Vec::size()) {
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
          for (; d3 < size - (size % Vec::size()); d3 += Vec::size()) {
            Vec out_vec = Vec::loadu(out + d3) / Vec(Scalar(kh * kw));
            out_vec.store(out + d3);
          }
          for (; d3 < size; d3++) {
            out[d3] = out[d3] / kh / kw;
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

pub fn cpu_adaptive_avg_pool_backward<Scalar>(
    grad_input:  &mut Tensor,
    grad_output: &Tensor)  {

    todo!();
        /*
            auto grad_output = grad_output_.contiguous();
      auto grad_input = grad_input_.contiguous();

      auto grad_output_data = grad_output.data_ptr<Scalar>();
      auto grad_input_data = grad_input.data_ptr<Scalar>();

      i64 ndim = grad_output.ndimension();
      // treat batch size and channels as one dimension
      i64 channels = ndim == 3 ? grad_output.size(0) : grad_output.size(0) * grad_output.size(1);
      i64 input_height = grad_input.size(-2);
      i64 input_width = grad_input.size(-1);
      i64 output_height = grad_output.size(-2);
      i64 output_width = grad_output.size(-1);

      // parallel on dim of N, C
      at::parallel_for(0, channels, 0, [&](i64 begin, i64 end) {
        for (i64 c = begin; c < end; c++) {
          Scalar* grad_input_ptr = grad_input_data + c * input_height * input_width;
          Scalar* grad_output_ptr = grad_output_data + c * output_height * output_width;

          for (i64 oh = 0; oh < output_height; oh++) {
            i64 ih0 = start_index(oh, output_height, input_height);
            i64 ih1 = end_index(oh, output_height, input_height);
            i64 kh = ih1 - ih0;

            for (i64 ow = 0; ow < output_width; ow++) {
              i64 iw0 = start_index(ow, output_width, input_width);
              i64 iw1 = end_index(ow, output_width, input_width);
              i64 kw = iw1 - iw0;

              Scalar grad_delta = grad_output_ptr[oh * output_width + ow] / kh / kw;
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

pub fn cpu_adaptive_avg_pool_backward_channels_last<Scalar>(
    grad_input:  &mut Tensor,
    grad_output: &Tensor)  {

    todo!();
        /*
            auto memory_format = at::MemoryFormat::ChannelsLast;
      auto grad_input = grad_input_.contiguous(memory_format);
      auto grad_output = grad_output_.contiguous(memory_format);

      auto grad_input_data = grad_input.data_ptr<Scalar>();
      auto grad_output_data = grad_output.data_ptr<Scalar>();

      i64 nbatch = grad_input.size(0);
      i64 channels = grad_input.size(1);
      i64 input_height = grad_input.size(2);
      i64 input_width = grad_input.size(3);
      i64 output_height = grad_output.size(2);
      i64 output_width = grad_output.size(3);

      using Vec = vec::Vectorized<Scalar>;
      // parallel on dim N
      at::parallel_for(0, nbatch, 0, [&](i64 begin, i64 end) {
        for (i64 n = begin; n < end; n++) {
          Scalar* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;
          Scalar* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;

          for (i64 oh = 0; oh < output_height; oh++) {
            i64 ih0 = start_index(oh, output_height, input_height);
            i64 ih1 = end_index(oh, output_height, input_height);
            i64 kh = ih1 - ih0;

            for (i64 ow = 0; ow < output_width; ow++) {
              i64 iw0 = start_index(ow, output_width, input_width);
              i64 iw1 = end_index(ow, output_width, input_width);
              i64 kw = iw1 - iw0;

              Scalar* gout = grad_output_ptr + oh * output_width * channels + ow * channels;
              i64 size = channels;
              for (i64 ih = ih0; ih < ih1; ih++) {
                for (i64 iw = iw0; iw < iw1; iw++) {
                  Scalar* gin = grad_input_ptr + ih * input_width * channels + iw * channels;

                  i64 d = 0;
                  for (; d < size - (size % Vec::size()); d += Vec::size()) {
                    Vec gin_vec = Vec::loadu(gin + d) + Vec::loadu(gout + d) / Vec(Scalar(kh * kw));
                    gin_vec.store(gin + d);
                  }
                  for (; d < size; d++) {
                    gin[d] += gout[d] / kw / kw;
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

pub fn adaptive_avg_pool2d_kernel_impl(
    output:      &mut Tensor,
    input:       &Tensor,
    output_size: &[i32])  {
    
    todo!();
        /*
            switch (input.suggest_memory_format()) {
        case at::MemoryFormat::Contiguous: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_avg_pool2d", [&] {
            cpu_adaptive_avg_pool<Scalar>(output, input, output_size);
          });
          break;
        }
        case at::MemoryFormat::ChannelsLast: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "adaptive_avg_pool2d_channels_last", [&]{
            cpu_adaptive_avg_pool_channels_last<Scalar>(output, input, output_size);
          });
          break;
        }
        default:
          TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
        */
}

pub fn adapative_avg_pool2d_backward_kernel_impl(
    grad_input:  &mut Tensor,
    grad_output: &Tensor)  {
    
    todo!();
        /*
            switch (grad_output.suggest_memory_format()) {
        case at::MemoryFormat::Contiguous: {
          AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "adaptive_avg_pool2d_backward", [&] {
            cpu_adaptive_avg_pool_backward<Scalar>(grad_input, grad_output);
          });
          break;
        }
        case at::MemoryFormat::ChannelsLast: {
          AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "adaptive_avg_pool2d_backward_channels_last", [&]{
            cpu_adaptive_avg_pool_backward_channels_last<Scalar>(grad_input, grad_output);
          });
          break;
        }
        default:
          TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
        */
}

register_dispatch!{
    adaptive_avg_pool2d_kernel, 
    &adaptive_avg_pool2d_kernel_impl
}

register_dispatch!{
    adaptive_avg_pool2d_backward_kernel, 
    &adapative_avg_pool2d_backward_kernel_impl
}
