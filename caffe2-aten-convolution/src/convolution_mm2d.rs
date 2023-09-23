crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ConvolutionMM2d.cpp]

#[inline] pub fn slow_conv2d_shape_check(
        input:           &Tensor,
        grad_output:     &Tensor,
        weight:          &Tensor,
        bias:            &Tensor,
        kernel_height:   i64,
        kernel_width:    i64,
        stride_height:   i64,
        stride_width:    i64,
        pad_height:      i64,
        pad_width:       i64,
        weight_optional: bool)  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_width > 0 && kernel_height > 0,
          "kernel size should be greater than zero, but got kernel_height: ",
          kernel_height,
          " kernel_width: ",
          kernel_width);
      TORCH_CHECK(
          stride_width > 0 && stride_height > 0,
          "stride should be greater than zero, but got stride_height: ",
          stride_height,
          " stride_width: ",
          stride_width);

      if (weight.defined()) {
        TORCH_CHECK(
            weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 4),
            "non-empty 2D or 4D weight tensor expected, but got: ",
            weight.sizes());
        if (bias.defined()) {
          check_dim_size(bias, 1, 0, weight.size(0));
        }
      } else {
        TORCH_CHECK(weight_optional, "weight tensor is undefined");
      }

      const i64 ndim = input.dim();
      const i64 dim_batch = 0;
      const i64 dim_planes = 1;
      const i64 dim_height = 2;
      const i64 dim_width = 3;

      // Allow for empty batch size but not other dimensions
      bool valid_empty = ndim == 4 && input.size(dim_batch) == 0 &&
          input.size(dim_planes) != 0 && input.size(dim_height) != 0 &&
          input.size(dim_width) != 0;

      TORCH_CHECK(
          (input.numel() > 0 || valid_empty) && ndim == 4,
          "non-empty 4D input tensor expected but got: ",
          input.sizes());

      const i64 input_height = input.size(dim_height);
      const i64 input_width = input.size(dim_width);

      const i64 exact_input_height = input_height + 2 * pad_height;
      const i64 exact_input_width = input_width + 2 * pad_width;

      TORCH_CHECK(
          exact_input_height >= kernel_height && exact_input_width >= kernel_width,
          "Calculated padded input size per channel: (",
          exact_input_height,
          " x ",
          exact_input_width,
          "). ",
          "Kernel size: (",
          kernel_height,
          " x ",
          kernel_width,
          "). Kernel size can't be greater than actual input size");

      const i64 output_height =
          div_rtn<i64>(exact_input_height - kernel_height, stride_height) + 1;
      const i64 output_width =
          div_rtn<i64>(exact_input_width - kernel_width, stride_width) + 1;

      TORCH_CHECK(
          output_width >= 1 && output_height >= 1,
          "Given input size per channel: (",
          input_height,
          " x ",
          input_width,
          "). "
          "Calculated output size per channel: (",
          output_height,
          " x ",
          output_width,
          "). Output size is too small");

      if (weight.defined()) {
        i64 n_input_plane = weight.size(1);
        if (weight.dim() == 2) {
          n_input_plane /= (kernel_height * kernel_width);
        }
        check_dim_size(input, ndim, dim_planes, n_input_plane);
      }

      if (grad_output.defined()) {
        if (weight.defined()) {
          i64 n_output_plane = weight.size(0);
          check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
        } else if (bias.defined()) {
          TORCH_CHECK(bias.numel() > 0, "non-empty bias tensor expected");
          const i64 n_output_plane = bias.dim() == 0 ? 1 : bias.size(0);
          check_dim_size(grad_output, ndim, dim_planes, n_output_plane);
        }
        check_dim_size(grad_output, ndim, dim_height, output_height);
        check_dim_size(grad_output, ndim, dim_width, output_width);
      }
        */
}

pub fn view_weight_2d(weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor weight = weight_.contiguous();
      if (weight.dim() == 4) {
        const i64 s1 = weight.size(0);
        const i64 s2 = weight.size(1) * weight.size(2) * weight.size(3);
        return weight.view({s1, s2});
      } else {
        return weight;
      }
        */
}

pub fn slow_conv2d_update_output_frame(
        input:          &mut Tensor,
        output:         &mut Tensor,
        weight:         &Tensor,
        bias:           &Tensor,
        finput:         &mut Tensor,
        kernel_height:  i64,
        kernel_width:   i64,
        stride_height:  i64,
        stride_width:   i64,
        pad_height:     i64,
        pad_width:      i64,
        n_input_plane:  i64,
        input_height:   i64,
        input_width:    i64,
        n_output_plane: i64,
        output_height:  i64,
        output_width:   i64)  {
    
    todo!();
        /*
            // Note: this is a no_group conv2d
      if ((input.ndimension() == 4) && (kernel_height == 1) && (stride_height == 1) && (pad_height == 0) &&
          (kernel_width == 1) && (stride_width == 1) && (pad_width == 0)) {
        auto output2d =
            output.reshape({n_output_plane, output_height * output_width});
        auto weight_new =
            weight.view({n_output_plane, n_input_plane});
        auto input_new =
            input.view({n_input_plane, output_height * output_width});

        if (bias.defined()) {
          output.copy_(bias.unsqueeze(-1).unsqueeze(-1));
          output2d.addmm_(weight_new, input_new, 1, 1);
        } else {
          mm_out(output2d, weight_new, input_new);
        }
        return;
      }
      unfolded2d_copy_stub(
          kCPU,
          finput,
          input,
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width,
          n_input_plane,
          input_height,
          input_width,
          output_height,
          output_width);

      auto output2d =
          output.reshape({n_output_plane, output_height * output_width});
      if (bias.defined()) {
        output.copy_(bias.unsqueeze(-1).unsqueeze(-1));
        output2d.addmm_(weight, finput, 1, 1);
      } else {
        output2d.addmm_(weight, finput, 0, 1);
      }
        */
}

pub fn slow_conv2d_backward_update_grad_input_frame(
        grad_input:    &mut Tensor,
        grad_output:   &Tensor,
        weight:        &Tensor,
        fgrad_input:   &mut Tensor,
        kernel_height: i64,
        kernel_width:  i64,
        stride_height: i64,
        stride_width:  i64,
        pad_height:    i64,
        pad_width:     i64)  {
    
    todo!();
        /*
            auto grad_output_2d = grad_output.reshape(
          {grad_output.size(0), grad_output.size(1) * grad_output.size(2)});
      fgrad_input.addmm_(weight, grad_output_2d, 0, 1);

      grad_input.zero_();
      unfolded2d_acc_stub(
          kCPU,
          fgrad_input,
          grad_input,
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width,
          grad_input.size(0),
          grad_input.size(1),
          grad_input.size(2),
          grad_output.size(1),
          grad_output.size(2));
        */
}

pub fn slow_conv2d_backward_out_cpu_template(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        input:       &Tensor,
        weight:      &Tensor,
        finput:      &Tensor,
        fgrad_input: &mut Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32])  {
    
    todo!();
        /*
            const i64 kernel_height = kernel_size[0];
      const i64 kernel_width = kernel_size[1];
      const i64 pad_height = padding[0];
      const i64 pad_width = padding[1];
      const i64 stride_height = stride[0];
      const i64 stride_width = stride[1];

      const Tensor weight = view_weight_2d(weight_);
      slow_conv2d_shape_check(
          input_,
          grad_output_,
          weight,
          Tensor(),
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width,
          false);

      const Tensor input = input_.contiguous();
      const Tensor grad_output = grad_output_.contiguous();
      grad_input.resize_as_(input);
      fgrad_input.resize_as_(finput);
      fgrad_input.zero_();
      const Tensor tweight = weight.transpose(0, 1);
      const i64 batch_size = input.size(0);
      parallel_for(0, batch_size, 0, [&](i64 start, i64 end) {
        NoGradGuard no_grad;
        AutoDispatchBelowADInplaceOrView non_variable_type_mode;
        for (i64 t = start; t < end; t++) {
          Tensor grad_input_t = grad_input[t];
          Tensor grad_output_t = grad_output[t];
          Tensor fgrad_input_t = fgrad_input[t];
          slow_conv2d_backward_update_grad_input_frame(
              grad_input_t,
              grad_output_t,
              tweight,
              fgrad_input_t,
              kernel_height,
              kernel_width,
              stride_height,
              stride_width,
              pad_height,
              pad_width);
        }
      });
        */
}

pub fn slow_conv2d_backward_parameters_frame(
        grad_weight: &mut Tensor,
        grad_bias:   &mut Tensor,
        grad_output: &mut Tensor,
        finput:      &Tensor)  {
    
    todo!();
        /*
            auto grad_output_2d = grad_output.view(
          {grad_output.size(0), grad_output.size(1) * grad_output.size(2)});
      if (grad_weight.defined()) {
        const Tensor tfinput = finput.transpose(0, 1);
        grad_weight.addmm_(grad_output_2d, tfinput);
      }

      if (grad_bias.defined()) {
        AT_DISPATCH_FLOATING_TYPES_AND(
            ScalarType::BFloat16,
            grad_output.scalar_type(),
            "slow_conv2d_backward_parameters",
            [&] {
              auto grad_output_2d_acc = grad_output_2d.accessor<Scalar, 2>();
              auto grad_bias_acc = grad_bias.accessor<Scalar, 1>();
              const auto sz = grad_output_2d.size(1);
              for (i64 i = 0; i < grad_bias.size(0); i++) {
                Scalar sum = 0;
                for (i64 k = 0; k < sz; k++) {
                  sum += grad_output_2d_acc[i][k];
                }
                grad_bias_acc[i] += sum;
              }
            });
      }
        */
}

pub fn slow_conv2d_backward_parameters_out_cpu_template(
        grad_weight: &mut Tensor,
        grad_bias:   &mut Tensor,
        input:       &Tensor,
        grad_output: &Tensor,
        finput:      &Tensor,
        fgrad_input: Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32])  {
    
    todo!();
        /*
            CheckedFrom c = "slow_conv2d_backward_parameters_cpu";
      auto grad_weight_arg = TensorArg(grad_weight, "grad_weight_arg", 0);
      auto grad_bias_arg = TensorArg(grad_bias, "grad_bias_arg", 0);

      const i64 kernel_height = kernel_size[0];
      const i64 kernel_width = kernel_size[1];
      const i64 pad_height = padding[0];
      const i64 pad_width = padding[1];
      const i64 stride_height = stride[0];
      const i64 stride_width = stride[1];

      Tensor grad_weight_2d;
      if (grad_weight.defined()) {
        checkContiguous(c, grad_weight_arg);
        grad_weight_2d = view_weight_2d(grad_weight);
      }

      if (grad_bias.defined()) {
        checkContiguous(c, grad_bias_arg);
      }

      slow_conv2d_shape_check(
          input_,
          grad_output_,
          grad_weight_2d,
          grad_bias,
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width,
          true);

      auto input = input_.contiguous();
      auto grad_output = grad_output_.contiguous();

      const i64 batch_size = input.size(0);
      for (i64 t = 0; t < batch_size; t++) {
        Tensor grad_output_t = grad_output[t];
        Tensor finput_t;
        if (grad_weight_2d.defined()) {
          finput_t = finput[t];
        }

        slow_conv2d_backward_parameters_frame(
            grad_weight_2d, grad_bias, grad_output_t, finput_t);
      }
        */
}

pub fn slow_conv2d_forward_out_cpu<'a>(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        output:      &mut Tensor,
        finput:      &mut Tensor,
        fgrad_input: &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      const i64 kernel_height = kernel_size[0];
      const i64 kernel_width = kernel_size[1];
      const i64 pad_height = padding[0];
      const i64 pad_width = padding[1];
      const i64 stride_height = stride[0];
      const i64 stride_width = stride[1];

      const Tensor weight_2d = view_weight_2d(weight_);

      slow_conv2d_shape_check(
          self,
          Tensor(),
          weight_2d,
          bias,
          kernel_height,
          kernel_width,
          stride_height,
          stride_width,
          pad_height,
          pad_width,
          false);

      const Tensor input = self.contiguous();
      const i64 dim_planes = 1;
      const i64 dim_height = 2;
      const i64 dim_width = 3;

      const i64 n_input_plane = input.size(dim_planes);
      const i64 input_height = input.size(dim_height);
      const i64 input_width = input.size(dim_width);
      const i64 n_output_plane = weight_2d.size(0);
      const i64 output_height =
          (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
      const i64 output_width =
          (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

      const i64 batch_size = input.size(0);

      if ((input.ndimension() == 4) && (kernel_height == 1) && (stride_height == 1) && (pad_height == 0) &&
          (kernel_width == 1) && (stride_width == 1) && (pad_width == 0)) {
        finput =
            input.view({batch_size, n_input_plane, output_height * output_width})
                .detach();
      } else {
         finput.resize_({batch_size,
                      n_input_plane * kernel_height * kernel_width,
                      output_height * output_width});
      }
      output.resize_({batch_size, n_output_plane, output_height, output_width});

      parallel_for(0, batch_size, 0, [&](i64 start, i64 end) {
        NoGradGuard no_grad;
        AutoDispatchBelowADInplaceOrView non_variable_type_mode;
        for (i64 t = start; t < end; t++) {
          Tensor input_t = input[t].unsqueeze(0);
          Tensor output_t = output[t];
          Tensor finput_t = finput[t];
          slow_conv2d_update_output_frame(
              input_t,
              output_t,
              weight_2d,
              bias,
              finput_t,
              kernel_height,
              kernel_width,
              stride_height,
              stride_width,
              pad_height,
              pad_width,
              n_input_plane,
              input_height,
              input_width,
              n_output_plane,
              output_height,
              output_width);
        }
      });

      return tuple<Tensor&, Tensor&, Tensor&>(output, finput, fgrad_input);
        */
}

pub fn slow_conv2d_forward_cpu(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      auto output = empty({0}, self.options());
      auto finput = empty({0}, self.options());
      auto fgrad_input = empty({0}, self.options());
      native::slow_conv2d_forward_out_cpu(
          self,
          weight,
          kernel_size,
          bias,
          stride,
          padding,
          output,
          finput,
          fgrad_input);
      return make_tuple(output, finput, fgrad_input);
        */
}

pub fn slow_conv2d_backward_out_cpu<'a>(
        grad_output: &Tensor,
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        finput:      &Tensor,
        fgrad_input: &Tensor,
        grad_input:  &mut Tensor,
        grad_weight: &mut Tensor,
        grad_bias:   &mut Tensor) -> (&'a mut Tensor,&'a mut Tensor,&'a mut Tensor) {
    
    todo!();
        /*
            if (grad_input.defined()) {
        slow_conv2d_backward_out_cpu_template(
            grad_input,
            grad_output,
            self,
            weight,
            finput,
            const_cast<Tensor&>(fgrad_input),   // cast away auto-generated const of buffer
            kernel_size,
            stride,
            padding);
      }

      if (grad_weight.defined()) {
        grad_weight.resize_(weight.sizes());
        grad_weight.zero_();
      }

      if (grad_bias.defined()) {
        grad_bias.resize_({grad_output.size(1)});
        grad_bias.zero_();
      }

      if (grad_weight.defined() || grad_bias.defined()) {
        slow_conv2d_backward_parameters_out_cpu_template(
            grad_weight,
            grad_bias,
            self,
            grad_output,
            finput,
            fgrad_input,
            kernel_size,
            stride,
            padding);
      }

      return tuple<Tensor&, Tensor&, Tensor&>(
          grad_input, grad_weight, grad_bias);
        */
}

pub fn slow_conv2d_backward_cpu(
        grad_output: &Tensor,
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        finput:      &Tensor,
        fgrad_input: &Tensor,
        output_mask: [bool; 3]) -> (Tensor,Tensor,Tensor) {
    
    todo!();
        /*
            Tensor grad_input;
      Tensor grad_weight;
      Tensor grad_bias;

      if (output_mask[0]) {
        grad_input = empty({0}, grad_output.options());
      }

      if (output_mask[1]) {
        grad_weight = empty({0}, grad_output.options());
      }

      if (output_mask[2]) {
        grad_bias = empty({0}, grad_output.options());
      }

      native::slow_conv2d_backward_out_cpu(
          grad_output,
          self,
          weight,
          kernel_size,
          stride,
          padding,
          finput,
          fgrad_input,
          grad_input,
          grad_weight,
          grad_bias);

      return make_tuple(grad_input, grad_weight, grad_bias);
        */
}

pub fn thnn_conv2d_out<'a>(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        output:      &mut Tensor) -> &'a mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      Tensor finput = empty({0}, self.options());
      Tensor fgrad_input = empty({0}, self.options());
      return get<0>(thnn_conv2d_forward_out(output, finput, fgrad_input, self, weight, kernel_size, bias, stride, padding));
        */
}

pub fn thnn_conv2d(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32]) -> Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      return get<0>(thnn_conv2d_forward(self, weight, kernel_size, bias, stride, padding));
        */
}
