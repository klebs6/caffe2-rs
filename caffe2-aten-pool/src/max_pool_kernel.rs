crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/cpu/MaxPoolKernel.cpp]

pub fn cpu_max_pool<Scalar>(
        output:    &Tensor,
        indices:   &Tensor,
        input:     &Tensor,
        kw:        i32,
        kh:        i32,
        dw:        i32,
        dh:        i32,
        padw:      i32,
        padh:      i32,
        dilationw: i32,
        dilationh: i32)  {

    todo!();
        /*
            auto input = input_.contiguous();
      auto output = output_.contiguous();
      auto indices = indices_.contiguous();

      auto input_data = input.data_ptr<Scalar>();
      auto output_data = output.data_ptr<Scalar>();
      auto indices_data = indices.data_ptr<i64>();

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
          i64 ih0 = oh * dH - padH;
          i64 iw0 = ow * dW - padW;
          i64 ih1 = min(ih0 + (kH - 1) * dilationH + 1, input_height);
          i64 iw1 = min(iw0 + (kW - 1) * dilationW + 1, input_width);
          while(ih0 < 0) { ih0 += dilationH; }
          while(iw0 < 0) { iw0 += dilationW; }

          // local pointers
          Scalar* input_ptr = input_data + c * input_height * input_width;

          // compute local max
          i64 maxindex = ih0 * input_width + iw0;
          Scalar maxval = -numeric_limits<Scalar>::infinity();
          for (i64 ih = ih0; ih < ih1; ih += dilationH) {
            for (i64 iw = iw0; iw < iw1; iw += dilationW) {
              i64 index = ih * input_width + iw;
              Scalar val = input_ptr[index];
              if ((val > maxval) || isnan(val)) {
                maxval = val;
                maxindex = index;
              }
            }
          }

          // set output to local max and store location of max
          output_data[i] = maxval;
          indices_data[i] = maxindex;

          // move on to next output index
          data_index_step(c, channels, oh, output_height, ow, output_width);
        }
      });

      if (!output_.is_contiguous()) {
        output_.copy_(output);
      }
      if (!indices_.is_contiguous()) {
        indices_.copy_(indices);
      }
        */
}

pub fn cpu_max_pool_channels_last<Scalar>(
        output:    &Tensor,
        indices:   &Tensor,
        input:     &Tensor,
        kw:        i32,
        kh:        i32,
        dw:        i32,
        dh:        i32,
        padw:      i32,
        padh:      i32,
        dilationw: i32,
        dilationh: i32)  {

    todo!();
        /*
            TORCH_CHECK(input_.ndimension() == 4,
                  "max pooling with channels last format supports tensors with 4 dims");
      auto memory_format = MemoryFormat::ChannelsLast;
      auto input = input_.contiguous(memory_format);
      auto output = output_.contiguous(memory_format);
      auto indices = indices_.contiguous(memory_format);

      auto input_data = input.data_ptr<Scalar>();
      auto output_data = output.data_ptr<Scalar>();
      auto indices_data = indices.data_ptr<i64>();

      i64 nbatch = input.size(0);
      i64 channels = input.size(1);
      i64 input_height = input.size(2);
      i64 input_width = input.size(3);
      i64 output_height = output.size(2);
      i64 output_width = output.size(3);

      using Vec = vec::Vectorized<Scalar>;
      using integer_t = vec::int_same_size_t<Scalar>;
      using iVec = vec::Vectorized<integer_t>;
      // for the convience of vectorization, use integer of the same size of Scalar,
      //   e.g. i32 for float, i64 for double
      // need to make sure doesn't overflow
      TORCH_CHECK(input_height <= ceil((double)integer_t::max / (double)input_width));

      // parallel on dim N, H, W
      parallel_for(0, nbatch * output_height * output_width, 0, [&](i64 begin, i64 end) {
        i64 n = 0;
        i64 oh = 0;
        i64 ow = 0;
        data_index_init(begin, n, nbatch, oh, output_height, ow, output_width);

        i64 size = channels;
        i64 len = size - (size % Vec::size());
        // temp buffer holding index with integer_t
        unique_ptr<integer_t []> index_buffer(new integer_t[len]);

        for (i64 i = begin; i < end; i++) {
          i64 ih0 = oh * dH - padH;
          i64 iw0 = ow * dW - padW;
          i64 ih1 = min(ih0 + (kH - 1) * dilationH + 1, input_height);
          i64 iw1 = min(iw0 + (kW - 1) * dilationW + 1, input_width);
          while(ih0 < 0) { ih0 += dilationH; }
          while(iw0 < 0) { iw0 += dilationW; }

          Scalar* out = output_data + i * channels;
          i64* ind = indices_data + i * channels;

          // Pass I: init out lane
          iVec index0_vec = iVec(ih0 * input_width + iw0);
          Vec out_vec = Vec(-numeric_limits<Scalar>::infinity());
          i64 d1 = 0;
          for (; d1 < len; d1 += Vec::size()) {
            index0_vec.store(index_buffer.get() + d1);
            out_vec.store(out + d1);
          }
          for (; d1 < size; d1++) {
            ind[d1] = ih0 * input_width + iw0;
            out[d1] = -numeric_limits<Scalar>::infinity();
          }
          // Pass II: compute local max
          for (i64 ih = ih0; ih < ih1; ih += dilationH) {
            for (i64 iw = iw0; iw < iw1; iw += dilationW) {
              Scalar* in = input_data + n * input_height * input_width * channels +
                  ih * input_width * channels + iw * channels;

              i64 d2 = 0;
              for (; d2 < len; d2 += Vec::size()) {
                iVec index_vec = iVec(ih * input_width + iw);
                Vec val_vec = Vec::loadu(in + d2);
                iVec maxindex_vec = iVec::loadu(index_buffer.get() + d2);
                Vec maxval_vec = Vec::loadu(out + d2);

                // true = all ones, false = all zeros
                Vec mask = (val_vec > maxval_vec) | val_vec.isnan();
                iVec imask = vec::cast<integer_t>(mask);
                Vec out_vec = Vec::blendv(maxval_vec, val_vec, mask);
                iVec ind_vec = iVec::blendv(maxindex_vec, index_vec, imask);

                out_vec.store(out + d2);
                ind_vec.store(index_buffer.get() + d2);
              }
              for (; d2 < size; d2++) {
                i64 index = ih * input_width + iw;
                Scalar val = in[d2];
                i64 maxindex = ind[d2];
                Scalar maxval = out[d2];

                bool mask = (val > maxval) || isnan(val);
                out[d2] = mask ? val : maxval;
                ind[d2] = mask ? index : maxindex;
              }
            }
          }
          // convert indice data type
          vec::convert<integer_t, i64>(index_buffer.get(), ind, len);

          // move on to next output index
          data_index_step(n, nbatch, oh, output_height, ow, output_width);
        }
      });

      if (!output_.is_contiguous(memory_format)) {
        output_.copy_(output);
      }
      if (!indices_.is_contiguous(memory_format)) {
        indices_.copy_(indices);
      }
        */
}

pub fn cpu_max_pool_backward<Scalar>(
        grad_input:  &Tensor,
        grad_output: &Tensor,
        indices:     &Tensor)  {

    todo!();
        /*
            auto grad_output = grad_output_.contiguous();
      auto indices = indices_.contiguous();
      auto grad_input = grad_input_.contiguous();

      auto grad_output_data = grad_output.data_ptr<Scalar>();
      auto indices_data = indices.data_ptr<i64>();
      auto grad_input_data = grad_input.data_ptr<Scalar>();

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
          Scalar* grad_input_ptr = grad_input_data + c * input_height * input_width;
          Scalar* grad_output_ptr = grad_output_data + c * output_height * output_width;
          i64 * indices_ptr = indices_data + c * output_height * output_width;

          for (i64 oh = 0; oh < output_height; oh++) {
            for (i64 ow = 0; ow < output_width; ow++) {
              // retrieve position of max
              i64 index = oh * output_width + ow;
              i64 maxindex = indices_ptr[index];
              if (maxindex != -1) {
                // update gradient
                grad_input_ptr[maxindex] += grad_output_ptr[index];
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

pub fn cpu_max_pool_backward_channels_last<Scalar>(
        grad_input:  &Tensor,
        grad_output: &Tensor,
        indices:     &Tensor)  {

    todo!();
        /*
            TORCH_CHECK(grad_output_.ndimension() == 4,
                  "max pooling backward with channels last format supports tensors with 4 dims.");
      auto memory_format = MemoryFormat::ChannelsLast;
      auto grad_input = grad_input_.contiguous(memory_format);
      auto grad_output = grad_output_.contiguous(memory_format);
      auto indices = indices_.contiguous(memory_format);

      auto grad_input_data = grad_input.data_ptr<Scalar>();
      auto grad_output_data = grad_output.data_ptr<Scalar>();
      auto indices_data = indices.data_ptr<i64>();

      i64 nbatch = grad_input.size(0);
      i64 channels = grad_input.size(1);
      i64 input_height = grad_input.size(2);
      i64 input_width = grad_input.size(3);
      i64 output_height = grad_output.size(2);
      i64 output_width = grad_output.size(3);

      // parallel on dim N
      parallel_for(0, nbatch, 0, [&](i64 begin, i64 end) {
        for (i64 n = begin; n < end; n++) {
          Scalar* grad_input_ptr = grad_input_data + n * input_height * input_width * channels;
          Scalar* grad_output_ptr = grad_output_data + n * output_height * output_width * channels;
          i64* indices_ptr = indices_data + n * output_height * output_width * channels;

          for (i64 oh = 0; oh < output_height; oh++) {
            for (i64 ow = 0; ow < output_width; ow++) {
              Scalar* gout = grad_output_ptr + oh * output_width * channels + ow * channels;
              i64* ind = indices_ptr + oh * output_width * channels + ow * channels;
              // TODO: gcc vectorization
              for (i64 c = 0; c < channels; c++) {
                i64 maxindex = ind[c];
                if (maxindex != -1) {
                  grad_input_ptr[maxindex * channels + c] += gout[c];
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

pub fn max_pool2d_kernel_impl(
        output:    &Tensor,
        indices:   &Tensor,
        input:     &Tensor,
        kw:        i32,
        kh:        i32,
        dw:        i32,
        dh:        i32,
        padw:      i32,
        padh:      i32,
        dilationw: i32,
        dilationh: i32)  {
    
    todo!();
        /*
            switch (input.suggest_memory_format()) {
        case MemoryFormat::Contiguous: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d", [&] {
            cpu_max_pool<Scalar>(output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
          });
          break;
        }
        case MemoryFormat::ChannelsLast: {
          AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "max_pool2d_channels_last", [&] {
            cpu_max_pool_channels_last<Scalar>(output, indices, input, kW, kH, dW, dH, padW, padH, dilationW, dilationH);
          });
          break;
        }
        default:
          TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
        */
}

pub fn max_pool2d_backward_kernel_impl(
        grad_input:  &Tensor,
        grad_output: &Tensor,
        indices:     &Tensor)  {
    
    todo!();
        /*
            switch (grad_output.suggest_memory_format()) {
        case MemoryFormat::Contiguous: {
          AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "max_pool2d_backward", [&] {
            cpu_max_pool_backward<Scalar>(grad_input, grad_output, indices);
          });
          break;
        }
        case MemoryFormat::ChannelsLast: {
          AT_DISPATCH_FLOATING_TYPES(grad_output.scalar_type(), "max_pool2d_backward_channels_last", [&] {
            cpu_max_pool_backward_channels_last<Scalar>(grad_input, grad_output, indices);
          });
          break;
        }
        default:
          TORCH_CHECK(false, "Unsupported memory format. Supports only ChannelsLast, Contiguous");
      }
        */
}

register_dispatch!{max_pool2d_kernel          , &max_pool2d_kernel_impl}
register_dispatch!{max_pool2d_backward_kernel , &max_pool2d_backward_kernel_impl}
