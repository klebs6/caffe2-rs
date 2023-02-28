crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/LossMultiMargin.cpp]

#[inline] pub fn multi_margin_inner_sum_cpu<Scalar>(
        input_data:  *const Scalar,
        weight_data: *const Scalar,
        p:           i32,
        margin:      Scalar,
        dim:         i64,
        target_idx:  i64) -> Scalar {

    todo!();
        /*
            const Scalar input_target = input_data[target_idx];
      Scalar sum = 0;
      for (i64 d = 0; d < dim; d++) {
        if (d == target_idx) {
          continue;
        }

        const Scalar z = margin - input_target + input_data[d];
        if (z > 0) {
          Scalar h = (p == 1) ? z : z * z;
          if (weight_data != nullptr) {
            h *= weight_data[target_idx];
          }
          sum += h;
        }
      }

      sum /= dim;
      return sum;
        */
}

#[inline] pub fn target_index_checked(
        target_data: *const i64,
        index:       i64,
        dim:         i64) -> i64 {
    
    todo!();
        /*
            const i64 idx = target_data[index];
      TORCH_CHECK(idx >= 0 && idx < dim, "target out of range");
      return idx;
        */
}


#[inline] pub fn multi_margin_loss_cpu_kernel<Scalar>(
        output:      &mut Tensor,
        input_data:  *mut Scalar,
        target_data: *mut i64,
        p:           i32,
        margin:      Scalar,
        weight_data: *mut Scalar,
        nframe:      i64,
        dim:         i64,
        reduction:   i64)  {

    todo!();
        /*
            using accscalar_t = acc_type<Scalar, false>;

      // dim() != 0 check is for 1d input which produces a scalar output (that
      // cannot be handled by TensorAccessor)
      if (reduction == Reduction::None && output.dim() > 0) {
        auto output_acc = output.accessor<Scalar, 1>();
        for (i64 t = 0; t < nframe; t++) {
          const auto idx = target_index_checked(target_data, t, dim);
          auto sum = multi_margin_inner_sum_cpu(
              input_data, weight_data, p, margin, dim, idx);
          output_acc[t] = sum;
          input_data += dim;
        }
      } else {
        accscalar_t sum = 0;
        auto output_acc = output.data_ptr<Scalar>();
        for (i64 t = 0; t < nframe; t++) {
          const auto idx = target_index_checked(target_data, t, dim);
          sum += multi_margin_inner_sum_cpu(
              input_data, weight_data, p, margin, dim, idx);
          input_data += dim;
        }
        if (reduction == Reduction::Mean) {
          sum /= nframe;
        }
        output_acc[0] = sum;
      }
        */
}

pub fn multi_margin_loss_out_cpu_template(
        output:    &mut Tensor,
        input:     &Tensor,
        target:    &Tensor,
        p:         i32,
        margin:    &Scalar,
        weight:    &Tensor,
        reduction: i64)  {
    
    todo!();
        /*
      i64 nframe, dim;
      const auto ndims = input.dim();
      auto target_arg = TensorArg(target, "target", 2);

      TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");

      multi_margin_loss_shape_check(nframe, dim, ndims, target_arg, input, target);

      // produce a scalar output for 1d input
      if (reduction == Reduction::None && target.dim() > 0) {
        output.resize_({nframe});
      } else {
        output.resize_({});
      }
      if (input.numel() == 0) {
        return;
      }

      auto input_contiguous = input.contiguous();
      auto target_contiguous = target.contiguous();

      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "multi_margin_loss_cpu_kernel", [&] {
            auto input_data = input_contiguous.data_ptr<Scalar>();
            auto target_data = target_contiguous.data_ptr<i64>();
            auto weight_data =
                weight.defined() ? weight.data_ptr<Scalar>() : nullptr;
            multi_margin_loss_cpu_kernel<Scalar>(
                output,
                input_data,
                target_data,
                p,
                margin.to<Scalar>(),
                weight_data,
                nframe,
                dim,
                reduction);
          });
        */
}

pub fn multi_margin_loss_backward_cpu_kernel<Scalar>(
        grad_input_data: *mut Scalar,
        grad_output:     &Tensor,
        input_data:      *mut Scalar,
        target_data:     *mut i64,
        p:               i32,
        margin:          Scalar,
        g:               Scalar,
        weight_data:     *mut Scalar,
        nframe:          i64,
        dim:             i64,
        reduction:       i64)  {

    todo!();
        /*
            Scalar* grad_input_row_data = grad_input_data;
      for (i64 t = 0; t < nframe; t++) {
        i64 target_idx = target_index_checked(target_data, t, dim);
        Scalar input_target = input_data[target_idx];
        Scalar grad_input_target = 0;
        for (i64 d = 0; d < dim; d++) {
          Scalar z = margin - input_target + input_data[d];
          if (d == target_idx) {
            continue;
          }

          if (z > 0) {
            Scalar h = (p == 1) ? g : 2 * g * z;
            if (weight_data != nullptr) {
              h *= weight_data[target_idx];
            }
            grad_input_target -= h;
            grad_input_row_data[d] = h;
          } else {
            grad_input_row_data[d] = 0;
          }
        }
        grad_input_row_data[target_idx] = grad_input_target;

        input_data += dim;
        grad_input_row_data += dim;
      }

      if (reduction != Reduction::None || grad_output.dim() == 0) {
        assert(
            reduction != Reduction::None || grad_output.dim() > 0 ||
            nframe == 1); // check 1d scalar fallback-case
        const auto d = *grad_output.data_ptr<Scalar>();
        for (i64 t = 0; t < nframe * dim; t++) {
          grad_input_data[t] *= d;
        }
      } else {
        auto grad_output_acc = grad_output.accessor<Scalar, 1>();
        for (i64 t = 0; t < nframe; t++) {
          for (i64 d = 0; d < dim; d++) {
            grad_input_data[t * dim + d] *= grad_output_acc[t];
          }
        }
      }
        */
}

pub fn multi_margin_loss_backward_out_cpu_template(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        p:           i32,
        margin:      &Scalar,
        weight:      &Tensor,
        reduction:   i64)  {
    
    todo!();
        /*
      i64 nframe, dim;
      auto target_arg = TensorArg(target, "target", 2);
      const auto ndims = input.dim();

      TORCH_CHECK(p == 1 || p == 2, "only p == 1 and p == 2 supported");

      multi_margin_loss_shape_check(nframe, dim, ndims, target_arg, input, target);
      grad_input.resize_as_(input);
      TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous");

      if (input.numel() == 0) {
        return;
      }

      auto input_contiguous = input.contiguous();
      auto target_contiguous = target.contiguous();
      auto weight_contiguous = weight.contiguous();
      AT_DISPATCH_FLOATING_TYPES(
          input.scalar_type(), "multi_margin_loss_backward_cpu_kernel", [&] {
            auto grad_input_data = grad_input.data_ptr<Scalar>();
            auto input_data = input_contiguous.data_ptr<Scalar>();
            auto target_data = target_contiguous.data_ptr<i64>();
            auto weight_data = weight_contiguous.defined()
                ? weight_contiguous.data_ptr<Scalar>()
                : nullptr;
            Scalar g = reduction == Reduction::Mean
                ? static_cast<Scalar>(1. / (nframe * dim))
                : static_cast<Scalar>(1. / dim);
            multi_margin_loss_backward_cpu_kernel<Scalar>(
                grad_input_data,
                grad_output,
                input_data,
                target_data,
                p,
                margin.to<Scalar>(),
                g,
                weight_data,
                nframe,
                dim,
                reduction);
          });
        */
}

pub fn multi_margin_loss_cpu(
        input:      &Tensor,
        target:     &Tensor,
        p:          &Scalar,
        margin:     &Scalar,
        weight_opt: &Option<Tensor>,
        reduction:  i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      auto output = empty({0}, input.options());
      multi_margin_loss_out_cpu_template(
          output, input, target, p.toInt(), margin, weight, reduction);
      return output;
        */
}

pub fn multi_margin_loss_cpu_out(
        input:      &Tensor,
        target:     &Tensor,
        p:          &Scalar,
        margin:     &Scalar,
        weight_opt: &Option<Tensor>,
        reduction:  i64,
        output:     &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      multi_margin_loss_out_cpu_template(
          output, input, target, p.toInt(), margin, weight, reduction);
      return output;
        */
}

pub fn multi_margin_loss_cpu_backward(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        p:           &Scalar,
        margin:      &Scalar,
        weight_opt:  &Option<Tensor>,
        reduction:   i64) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      auto grad_input = empty({0}, input.options());
      multi_margin_loss_backward_out_cpu_template(
          grad_input,
          grad_output,
          input,
          target,
          p.toInt(),
          margin,
          weight,
          reduction);
      return grad_input;
        */
}

pub fn multi_margin_loss_cpu_backward_out(
        grad_output: &Tensor,
        input:       &Tensor,
        target:      &Tensor,
        p:           &Scalar,
        margin:      &Scalar,
        weight_opt:  &Option<Tensor>,
        reduction:   i64,
        grad_input:  &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> weight_maybe_owned = borrow_from_optional_tensor(weight_opt);
      const Tensor& weight = *weight_maybe_owned;

      multi_margin_loss_backward_out_cpu_template(
          grad_input,
          grad_output,
          input,
          target,
          p.toInt(),
          margin,
          weight,
          reduction);
      return grad_input;
        */
}
