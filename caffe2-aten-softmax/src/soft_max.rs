// # vim: ft=none
crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/SoftMax.cpp]

pub fn host_softmax<Scalar, const LogSoftMax: bool>(
    output: Tensor,
    input:  &Tensor,
    dim:    i64)  {

    todo!();
        /*
            i64 outer_size = 1;
      i64 dim_size = input.size(dim);
      i64 inner_size = 1;
      for (i64 i = 0; i < dim; ++i)
        outer_size *= input.size(i);
      for (i64 i = dim + 1; i < input.dim(); ++i)
        inner_size *= input.size(i);
      i64 dim_stride = inner_size;
      i64 outer_stride = dim_size * dim_stride;
      Scalar* input_data_base = input.data_ptr<Scalar>();
      Scalar* output_data_base = output.data_ptr<Scalar>();
      i64 grain_size = min(internal::GRAIN_SIZE / dim_size, (i64)1);
      parallel_for(
          0, outer_size * inner_size, grain_size,
          [&](i64 begin, i64 end) {
            for (i64 i = begin; i < end; i++) {
              i64 outer_idx = i / inner_size;
              i64 inner_idx = i % inner_size;
              Scalar* input_data =
                  input_data_base + outer_idx * outer_stride + inner_idx;
              Scalar* output_data =
                  output_data_base + outer_idx * outer_stride + inner_idx;
              Scalar max_input = input_data[0];
              for (i64 d = 1; d < dim_size; d++)
                max_input = max(max_input, input_data[d * dim_stride]);

              acc_type<Scalar, false> tmpsum = 0;
              for (i64 d = 0; d < dim_size; d++) {
                Scalar z = exp(input_data[d * dim_stride] - max_input);
                if (!LogSoftMax) {
                  output_data[d * dim_stride] = z;
                }
                tmpsum += z;
              }

              if (LogSoftMax)
                tmpsum = log(tmpsum);
              else
                tmpsum = 1 / tmpsum;

              for (i64 d = 0; d < dim_size; d++)
                if (LogSoftMax)
                  output_data[d * dim_stride] =
                      input_data[d * dim_stride] - max_input - tmpsum;
                else
                  output_data[d * dim_stride] *= tmpsum;
            }
          });
        */
}

pub fn host_softmax_backward<Scalar, const LogSoftMax: bool>(
    gi:     &mut Tensor,
    grad:   &Tensor,
    output: &Tensor,
    dim:    i64)  {

    todo!();
        /*
            i64 outer_size = 1;
      i64 dim_size = grad.size(dim);
      i64 inner_size = 1;
      for (i64 i = 0; i < dim; ++i)
        outer_size *= grad.size(i);
      for (i64 i = dim + 1; i < grad.dim(); ++i)
        inner_size *= grad.size(i);
      i64 dim_stride = inner_size;
      i64 outer_stride = dim_size * dim_stride;
      Scalar* gradInput_data_base = gI.data_ptr<Scalar>();
      Scalar* output_data_base = output.data_ptr<Scalar>();
      Scalar* gradOutput_data_base = grad.data_ptr<Scalar>();
      i64 grain_size = min(internal::GRAIN_SIZE / dim_size, (i64)1);
      parallel_for(
          0, outer_size * inner_size, grain_size, [&](i64 begin, i64 end) {
            for (i64 i = begin; i < end; i++) {
              i64 outer_idx = i / inner_size;
              i64 inner_idx = i % inner_size;
              Scalar* gradInput_data =
                  gradInput_data_base + outer_idx * outer_stride + inner_idx;
              Scalar* output_data =
                  output_data_base + outer_idx * outer_stride + inner_idx;
              const Scalar* gradOutput_data =
                  gradOutput_data_base + outer_idx * outer_stride + inner_idx;

              acc_type<Scalar, false> sum = 0;
              for (i64 d = 0; d < dim_size; d++)
                if (LogSoftMax)
                  sum += gradOutput_data[d * dim_stride];
                else
                  sum +=
                      gradOutput_data[d * dim_stride] * output_data[d * dim_stride];

              for (i64 d = 0; d < dim_size; d++) {
                if (LogSoftMax) {
                  gradInput_data[d * dim_stride] = gradOutput_data[d * dim_stride] -
                      exp(output_data[d * dim_stride]) * sum;
                } else {
                  gradInput_data[d * dim_stride] = output_data[d * dim_stride] *
                      (gradOutput_data[d * dim_stride] - sum);
                }
              }
            }
          });
        */
}

pub fn softmax_cpu(
    input:         &Tensor,
    dim:           i64,
    half_to_float: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on CPU");
      auto input = input_.contiguous();
      Tensor output = native::empty_like(
          input,
          nullopt /* dtype */,
          nullopt /* layout */,
          nullopt /* device */,
          nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      i64 dim = maybe_wrap_dim(dim_, input.dim());

      if (input.numel() == 0) {
        return output;
      }
      if (input.dim() == 0)
        input = input.view(1);
      TORCH_CHECK(
          dim >= 0 && dim < input.dim(),
          "dim must be non-negative and less than input dimensions");
      if (input.ndimension() > 0 && dim == input.ndimension() - 1) {
        softmax_lastdim_kernel(kCPU, output, input);
      } else {
        softmax_kernel(kCPU, output, input, dim);
      }
      return output;
        */
}

pub fn log_softmax_cpu(
    input:         &Tensor,
    dim:           i64,
    half_to_float: bool) -> Tensor {
    
    todo!();
        /*
            TORCH_CHECK(!half_to_float, "softmax with half to float conversion is not supported on CPU");
      auto input = input_.contiguous();
      Tensor output = native::empty_like(
          input,
          nullopt /* dtype */,
          nullopt /* layout */,
          nullopt /* device */,
          nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      i64 dim = maybe_wrap_dim(dim_, input.dim());

      if (input.numel() == 0) {
        return output;
      }
      if (input.dim() == 0)
        input = input.view(1);
      TORCH_CHECK(
          dim >= 0 && dim < input.dim(),
          "dim must be non-negative and less than input dimensions");
      if (input.ndimension() > 0 && dim == input.ndimension() - 1) {
        log_softmax_lastdim_kernel(kCPU, output, input);
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND(
            ScalarType::BFloat16, input.scalar_type(), "log_softmax",
            [&] { host_softmax<Scalar, true>(output, input, dim); });
      }
      return output;
        */
}

pub fn softmax_backward_cpu(
    grad:   &Tensor,
    output: &Tensor,
    dim:    i64,
    input:  &Tensor) -> Tensor {
    
    todo!();
        /*
            TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
      checkSameSize("softmax_backward", grad_arg, output_arg);
      i64 dim = maybe_wrap_dim(dim_, grad_.dim());
      auto grad = grad_.contiguous();
      auto output = output_.contiguous();
      Tensor grad_input = native::empty_like(
          grad,
          nullopt /* dtype */,
          nullopt /* layout */,
          nullopt /* device */,
          nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      if (output.numel() == 0) {
        return grad_input;
      }
      if (grad.dim() == 0)
        grad = grad.view(1);
      if (output.dim() == 0)
        output = output.view(1);
      TORCH_CHECK(
          dim >= 0 && dim < grad.dim(),
          "dim must be non-negative and less than input dimensions");
      if (grad.ndimension() > 0 && dim == grad.ndimension() - 1) {
        softmax_backward_lastdim_kernel(kCPU, grad_input, grad, output);
      } else {
        AT_DISPATCH_FLOATING_TYPES(grad.scalar_type(), "softmax_backward", [&] {
          host_softmax_backward<Scalar, false>(grad_input, grad, output, dim);
        });
      }
      return grad_input;
        */
}

pub fn log_softmax_backward_cpu(
    grad:   &Tensor,
    output: &Tensor,
    dim:    i64,
    input:  &Tensor) -> Tensor {

    todo!();
        /*
            TensorArg grad_arg{grad_, "grad", 1}, output_arg{output_, "output", 2};
      checkSameSize("log_softmax_backward", grad_arg, output_arg);
      i64 dim = maybe_wrap_dim(dim_, grad_.dim());
      auto grad = grad_.contiguous();
      auto output = output_.contiguous();
      Tensor grad_input = native::empty_like(
          grad,
          nullopt /* dtype */,
          nullopt /* layout */,
          nullopt /* device */,
          nullopt /* pin_memory */,
          LEGACY_CONTIGUOUS_MEMORY_FORMAT);

      if (output.numel() == 0) {
        return grad_input;
      }
      if (grad.dim() == 0)
        grad = grad.view(1);
      if (output.dim() == 0)
        output = output.view(1);
      TORCH_CHECK(
          dim >= 0 && dim < grad.dim(),
          "dim must be non-negative and less than input dimensions");
      if (grad.ndimension() > 0 && dim == grad.ndimension() - 1) {
        log_softmax_backward_lastdim_kernel(kCPU, grad_input, grad, output);
      } else {
        AT_DISPATCH_FLOATING_TYPES_AND(ScalarType::BFloat16, grad.scalar_type(),
                                       "log_softmax_backward", [&] {
                                         host_softmax_backward<Scalar, true>(
                                             grad_input, grad, output, dim);
                                       });
      }
      return grad_input;
        */
}


pub fn softmax_a(
        input: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return _softmax(input_, dim_, false);
      }();
      namedinference::propagate_names(result, input_);
      return result;
        */
}


pub fn softmax_b(
        input: &Tensor,
        dim:   i64,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
            return _softmax(input_, dim_, true);
        } else {
            Tensor converted = dtype.has_value() ? input_.toType(dtype.value()) : input_;
            return _softmax(converted, dim_, false);
        }
      }();
      namedinference::propagate_names(result, input_);
      return result;
        */
}


pub fn log_softmax_a(
        input: &Tensor,
        dim:   i64) -> Tensor {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        return _log_softmax(input_, dim_, false);
      }();
      namedinference::propagate_names(result, input_);
      return result;
        */
}


pub fn log_softmax_b(
        input: &Tensor,
        dim:   i64,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            auto result = [&]() {
        NoNamesGuard guard;
        if (input_.is_cuda() && input_.scalar_type() == ScalarType::Half && dtype == ScalarType::Float){
            return _log_softmax(input_, dim_, true);
        } else {
            Tensor converted = dtype.has_value()? input_.toType(dtype.value()) : input_;
            return _log_softmax(converted, dim_, false);
        }
      }();
      namedinference::propagate_names(result, input_);
      return result;
        */
}

define_dispatch!{softmax_lastdim_kernel}
define_dispatch!{log_softmax_lastdim_kernel}
define_dispatch!{softmax_backward_lastdim_kernel}
define_dispatch!{log_softmax_backward_lastdim_kernel}
define_dispatch!{softmax_kernel}

pub fn softmax_c(
        self_: &Tensor,
        dim:   Dimname,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return softmax(self, dimname_to_position(self, dim), dtype);
        */
}

pub fn log_softmax_c(
        self_: &Tensor,
        dim:   Dimname,
        dtype: Option<ScalarType>) -> Tensor {
    
    todo!();
        /*
            return log_softmax(self, dimname_to_position(self, dim), dtype);
        */
}
