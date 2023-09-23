crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/AdaptiveAveragePooling.cpp]

pub fn adaptive_avg_pool2d_out_cpu_template(
        output:      &mut Tensor,
        input:       &Tensor,
        output_size: &[i32])  {
    
    todo!();
        /*
            TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");
        i64 ndim = input.ndimension();
        for (i64 i = 0; i < ndim; i++) {
          TORCH_CHECK(input.size(i) > 0,
            "adaptive_avg_pooling2d(): expected input to have non-empty spatial dimensions, "
            "but input has sizes ", input.sizes(), " with dimension ", i, " being "
            "empty");
        }

        TORCH_CHECK((ndim == 3 || ndim == 4),
          "non-empty 3D or 4D (batch mode) tensor expected for input");
        TORCH_CHECK(input.dtype() == output.dtype(),
          "expected dtype ", input.dtype(), " for `output` but got dtype ", output.dtype());

        i64 channels  = input.size(-3);
        i64 output_height = output_size[0];
        i64 output_width = output_size[1];

        if (ndim == 3) {
          output.resize_({channels, output_height, output_width});
        } else {
          i64 nbatch = input.size(0);
          output.resize_({nbatch, channels, output_height, output_width}, input.suggest_memory_format());
        }

        adaptive_avg_pool2d_kernel(kCPU, output, input, output_size);
        */
}

pub fn adaptive_avg_pool2d_backward_out_cpu_template<'a>(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        input:       &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            i64 ndim = grad_output.ndimension();
        for (i64 i = 0; i < ndim; i++) {
          TORCH_CHECK(grad_output.size(i) > 0,
            "adaptive_avg_pooling2d_backward(): expected grad_output to have non-empty spatial dimensions, "
            "but grad_output has sizes ", grad_output.sizes(), " with dimension ", i, " being "
            "empty");
        }

        TORCH_CHECK((ndim == 3 || ndim == 4),
          "non-empty 3D or 4D (batch mode) tensor expected for grad_output");
        TORCH_CHECK(input.dtype() == grad_output.dtype(),
          "expected dtype ", input.dtype(), " for `grad_output` but got dtype ", grad_output.dtype());
        TORCH_CHECK(input.dtype() == grad_input.dtype(),
          "expected dtype ", input.dtype(), " for `grad_input` but got dtype ", grad_input.dtype());

        grad_input.resize_(input.sizes(), input.suggest_memory_format());
        grad_input.zero_();

        adaptive_avg_pool2d_backward_kernel(kCPU, grad_input, grad_output);
        return grad_input;
        */
}

pub fn adaptive_avg_pool2d_out_cpu<'a>(
        input:       &Tensor,
        output_size: &[i32],
        output:      &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            adaptive_avg_pool2d_out_cpu_template(
          output, input, output_size);
        return output;
        */
}

pub fn adaptive_avg_pool2d_cpu(
        input:       &Tensor,
        output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            auto output = empty({0}, input.options());
        adaptive_avg_pool2d_out_cpu_template(
          output, input, output_size);
        return output;
        */
}

pub fn adaptive_avg_pool2d(
        input:       &Tensor,
        output_size: &[i32]) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(output_size.size() == 2, "adaptive_avg_pool2d: output_size must be 2");

        if (input.is_mkldnn()) {
          return mkldnn_adaptive_avg_pool2d(input, output_size);
        }

        if (!input.is_quantized() && output_size[0] == 1 && output_size[1] == 1) {
          // in this case, adaptive pooling is just computing mean over hw
          // dimensions, which can be done more efficiently
          #if defined(C10_MOBILE) && defined(USE_XNNPACK)
          if (xnnpack::use_global_average_pool(input)) {
            return xnnpack::global_average_pool(input);
          }
          #endif

          Tensor out = input.mean({-1, -2}, /* keepdim = */ true);
          if (input.suggest_memory_format() == MemoryFormat::ChannelsLast) {
            // assert ndim == 4, since ndim = 3 doesn't give channels_last
            const int n = input.size(0);
            const int c = input.size(1);
            out.as_strided_({n, c, 1, 1}, {c, 1, c, c});
          }
          return out;
        } else {
          return _adaptive_avg_pool2d(input, output_size);
        }
        */
}

pub fn adaptive_avg_pool2d_backward_out_cpu<'a>(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        input:       &Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            adaptive_avg_pool2d_backward_out_cpu_template(
          grad_input, grad_output, input);
        return grad_input;
        */
}

pub fn adaptive_avg_pool2d_backward_cpu(
        grad_output: &Tensor,
        input:       &Tensor) -> Tensor {
    
    todo!();
        /*
            auto grad_input = empty({0}, input.options());
        adaptive_avg_pool2d_backward_out_cpu_template(
          grad_input, grad_output, input);
        return grad_input;
        */
}

define_dispatch!{adaptive_avg_pool2d_kernel}
define_dispatch!{adaptive_avg_pool2d_backward_kernel}
