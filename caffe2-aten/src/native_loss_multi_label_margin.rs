crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LossMultiLabelMargin.cpp]

#[inline] pub fn multilabel_margin_loss_forward_inner_sum_cpu<Scalar>(
        input_data:     *mut Scalar,
        target_data:    *mut i64,
        is_target_data: *mut Scalar,
        dim:            i64) -> Scalar {

    todo!();
        /*
            using accscalar_t = acc_type<Scalar, false>;
      accscalar_t sum = 0;
      for (i64 ddt = 0; ddt < dim; ddt++) {
        i64 target_idx = target_data[ddt];
        if (target_idx < 0) {
          break;
        }
        is_target_data[target_idx] = 1;
      }
      for (i64 dt = 0; dt < dim; dt++) {
        i64 target_idx = target_data[dt];
        if (target_idx < 0) {
          break;
        }

        Scalar input_target = input_data[target_idx];
        for (i64 d = 0; d < dim; d++) {
          if (!is_target_data[d]) {
            Scalar z = 1 - input_target + input_data[d];
            if (z > 0) {
              sum += z;
            }
          }
        }
      }

      return sum;
        */
}

pub fn multilabel_margin_loss_forward_out_frame<Scalar>(
        input_contiguous:  &Tensor,
        target_contiguous: &Tensor,
        output:            &mut Tensor,
        is_target:         &mut Tensor,
        reduction:         i64,
        nframe:            i64,
        dim:               i64)  {

    todo!();
        /*
            using accscalar_t = acc_type<Scalar, false>;
      Scalar* input_data = input_contiguous.data_ptr<Scalar>();
      i64* target_data = target_contiguous.data_ptr<i64>();
      Scalar* is_target_data = is_target.data_ptr<Scalar>();

      if (reduction != Reduction::None || output.dim() == 0) {
        Scalar* output_data = output.data_ptr<Scalar>();

        accscalar_t sum = 0;

        for (i64 t = 0; t < nframe; t++) {
          sum += multilabel_margin_loss_forward_inner_sum_cpu(
              input_data, target_data, is_target_data, dim);

          input_data += dim;
          target_data += dim;
          is_target_data += dim;
        }

        sum /= dim;
        if (reduction == Reduction::Mean) {
          sum /= nframe;
        }

        *output_data = sum; // write scalar output value
      } else {
        auto output_acc = output.accessor<Scalar, 1>();

        for (i64 t = 0; t < nframe; t++) {
          Scalar sum = multilabel_margin_loss_forward_inner_sum_cpu(
              input_data, target_data, is_target_data, dim);

          sum /= dim;
          output_acc[t] = sum;

          input_data += dim;
          target_data += dim;
          is_target_data += dim;
        }
      }
        */
}

pub fn multilabel_margin_loss_forward_out_cpu_template(
        input:     &Tensor,
        target:    &Tensor,
        output:    &mut Tensor,
        is_target: &mut Tensor,
        reduction: i64)  {
    
    todo!();
        /*
            auto target_arg = TensorArg(target, "target", 2);
      i64 nframe, dim;
      const i64 ndims = input.dim();
      if (ndims <= 1) {
        nframe = 1;
        dim = ndims == 0 ? 1 : input.size(0);
      }
      else {
        nframe = input.size(0);
        dim = input.size(1);
      }
      multilabel_margin_loss_shape_check(nframe, dim, ndims, target_arg, input, target);

      // special case target.dim() <= 1: produce scalar output for scalar inputs
      // even if reduction == Reduction::None
      if (reduction != Reduction::None || target.dim() <= 1) {
        output.resize_({});
      } else {
        output.resize_({nframe});
      }

      is_target.resize_as_(target);
      TORCH_CHECK(is_target.is_contiguous(), "is_target must be contiguous");
      is_target.zero_();

      if (input.numel() == 0) {
        return;
      }

      TORCH_CHECK(
          target.min().item<i64>() >= -1, target_arg, " is out of range");
      TORCH_CHECK(
          target.max().item<i64>() < dim, target_arg, " is out of range");

      auto input_contiguous = input.contiguous();
      auto target_contiguous = target.contiguous();

      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "multilabel_margin_loss_forward_out_frame", [&] {
            multilabel_margin_loss_forward_out_frame<Scalar>(
                input_contiguous, target_contiguous, output, is_target, reduction, nframe, dim);
          });
        */
}

pub fn multilabel_margin_loss_backward_out_frame<Scalar>(
        grad_input:           &mut Tensor,
        grad_output:          &Tensor,
        input_contiguous:     &Tensor,
        target_contiguous:    &Tensor,
        reduction:            i64,
        is_target_contiguous: &Tensor,
        nframe:               i64,
        dim:                  i64)  {

    todo!();
        /*
            auto is_target_arg = TensorArg(is_target_contiguous, "is_target", 5);

      TORCH_CHECK(
          is_target_contiguous.min().item<Scalar>() >= 0, is_target_arg, " is out of range");
      TORCH_CHECK(
          is_target_contiguous.max().item<Scalar>() <= 1, is_target_arg, " is out of range");

      Scalar* input_data = input_contiguous.data_ptr<Scalar>();
      i64* target_data = target_contiguous.data_ptr<i64>();
      Scalar* is_target_data = is_target_contiguous.data_ptr<Scalar>();
      Scalar g = static_cast<Scalar>(
          reduction == Reduction::Mean ? 1. / (nframe * dim) : 1. / dim);

      Scalar* grad_input_row_data = grad_input.data_ptr<Scalar>();
      for (i64 t = 0; t < nframe; t++) {
        for (i64 dt = 0; dt < dim; dt++) {
          i64 target_idx = target_data[dt];
          if (target_idx < 0) {
            break;
          }

          Scalar input_target = input_data[target_idx];
          for (i64 d = 0; d < dim; d++) {
            if (!is_target_data[d]) {
              Scalar z = 1 - input_target + input_data[d];
              if (z > 0) {
                grad_input_row_data[target_idx] -= g;
                grad_input_row_data[d] += g;
              }
            }
          }
        }
        input_data += dim;
        target_data += dim;
        is_target_data += dim;
        grad_input_row_data += dim;
      }

      Scalar* grad_input_data = grad_input.data_ptr<Scalar>();
      if (reduction != Reduction::None || grad_output.dim() == 0) {
        assert(
            reduction != Reduction::None || grad_output.dim() > 0 || nframe == 1);
        const auto d = *grad_output.data_ptr<Scalar>();
        for (i64 t = 0; t < nframe * dim; t++) {
          grad_input_data[t] *= d;
        }
      } else {
        check_dim_size(grad_output, 1, 0, nframe);
        auto grad_output_acc = grad_output.accessor<Scalar, 1>();
        for (i64 t = 0; t < nframe; t++) {
          for (i64 d = 0; d < dim; d++) {
            grad_input_data[t * dim + d] *= grad_output_acc[t];
          }
        }
      }
        */
}

pub fn multilabel_margin_loss_backward_out_cpu_template(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        is_target:   &Tensor)  {
    
    todo!();
        /*
      i64 nframe, dim;
      CheckedFrom c = "multilabel_margin_loss_backward_cpu_template";
      auto target_arg = TensorArg(target, "target", 3);
      auto is_target_arg = TensorArg(is_target, "is_target", 5);
      const i64 ndims = input.dim();

      multilabel_margin_loss_shape_check(nframe, dim, ndims, target_arg, input, target);
      checkSameSize(c, target_arg, is_target_arg);

      grad_input.resize_as_(input);
      if (grad_input.numel() == 0) {
        return;
      }

      TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");
      grad_input.zero_();

      TORCH_CHECK(
          target.min().item<i64>() >= -1, target_arg, " is out of range");
      TORCH_CHECK(
          target.max().item<i64>() < dim, target_arg, " is out of range");

      auto input_contiguous = input.contiguous();
      auto target_contiguous = target.contiguous();
      auto is_target_contiguous = is_target.contiguous();

      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "multilabel_margin_loss_backward_out_frame", [&] {
            multilabel_margin_loss_backward_out_frame<Scalar>(
                grad_input,
                grad_output,
                input_contiguous,
                target_contiguous,
                reduction,
                is_target_contiguous,
                nframe,
                dim);
          });
        */
}

pub fn multilabel_margin_loss_forward_out_cpu(
        self_:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        output:    &mut Tensor,
        is_target: &mut Tensor) -> (&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            multilabel_margin_loss_forward_out_cpu_template(
          self, target, output, is_target, reduction);
      return tuple<Tensor&, Tensor&>(output, is_target);
        */
}

pub fn multilabel_margin_loss_forward_cpu(
        self_:     &Tensor,
        target:    &Tensor,
        reduction: i64) -> (Tensor,Tensor) {
    
    todo!();
        /*
            auto output = empty({0}, self.options());
      auto is_target = empty({0}, self.options());
      native::multilabel_margin_loss_forward_out_cpu(
          self, target, reduction, output, is_target);
      return make_tuple(output, is_target);
        */
}

pub fn multilabel_margin_loss_backward_cpu_out(
        grad_output: &Tensor,
        self_:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        is_target:   &Tensor,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            multilabel_margin_loss_backward_out_cpu_template(
          grad_input, grad_output, self, target, reduction, is_target);
      return grad_input;
        */
}

pub fn multilabel_margin_loss_backward_cpu(
        grad_output: &Tensor,
        self_:       &Tensor,
        target:      &Tensor,
        reduction:   i64,
        is_target:   &Tensor) -> Tensor {
    
    todo!();
        /*
            auto grad_input = zeros_like(self, LEGACY_CONTIGUOUS_MEMORY_FORMAT);
      native::multilabel_margin_loss_backward_cpu_out(
          grad_output, self, target, reduction, is_target, grad_input);
      return grad_input;
        */
}

pub fn multilabel_margin_loss_out(
        self_:     &Tensor,
        target:    &Tensor,
        reduction: i64,
        output:    &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            Tensor is_target = empty({0}, self.options());
      return get<0>(multilabel_margin_loss_forward_out(output, is_target, self, target, reduction));
        */
}

pub fn multilabel_margin_loss(
        self_:     &Tensor,
        target:    &Tensor,
        reduction: i64) -> Tensor {
    
    todo!();
        /*
            return get<0>(multilabel_margin_loss_forward(self, target, reduction));
        */
}
