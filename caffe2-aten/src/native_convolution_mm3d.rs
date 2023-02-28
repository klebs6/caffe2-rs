crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/native/ConvolutionMM3d.cpp]

pub const CONV3D_GRAIN_SALT: i64 = 20;

#[inline] pub fn slow_conv3d_shape_check(
        input:           &Tensor,
        grad_output:     &Tensor,
        weight:          &Tensor,
        bias:            &Tensor,
        kernel_depth:    i64,
        kernel_height:   i64,
        kernel_width:    i64,
        stride_depth:    i64,
        stride_height:   i64,
        stride_width:    i64,
        pad_depth:       i64,
        pad_height:      i64,
        pad_width:       i64,
        groups:          i64,
        weight_optional: bool)  {
    
    todo!();
        /*
            TORCH_CHECK(
          kernel_width > 0 && kernel_height > 0 && kernel_depth > 0,
          "kernel size should be greater than zero, but got: ",
          kernel_depth,
          " x ",
          kernel_height,
          " x ",
          kernel_width,
          " (TxHxW)");
      TORCH_CHECK(
          stride_width > 0 && stride_height > 0 && stride_depth > 0,
          "stride should be greater than zero, but got: ",
          stride_depth,
          " x ",
          stride_height,
          " x ",
          stride_width,
          " (TxHxW)");
      if (weight.defined()) {
        TORCH_CHECK(
            weight.numel() > 0 && (weight.dim() == 2 || weight.dim() == 5),
            "non-empty 2D or 5D weight tensor expected, but got: ",
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
      const i64 dim_depth = 2;
      const i64 dim_height = 3;
      const i64 dim_width = 4;

      // Allow for empty batch size but not other dimensions
      bool valid_empty = ndim == 5 && input.size(dim_batch) == 0 &&
          input.size(dim_planes) != 0 && input.size(dim_depth) != 0 &&
          input.size(dim_height) != 0 && input.size(dim_width) != 0;

      TORCH_CHECK(
          (input.numel() > 0 || valid_empty) && ndim == 5,
          "non-empty 5D input tensor expected but got: ",
          input.sizes());

      const i64 input_depth = input.size(dim_depth);
      const i64 input_height = input.size(dim_height);
      const i64 input_width = input.size(dim_width);

      const i64 exact_input_depth = input_depth + 2 * pad_depth;
      const i64 exact_input_height = input_height + 2 * pad_height;
      const i64 exact_input_width = input_width + 2 * pad_width;

      TORCH_CHECK(
          exact_input_depth >= kernel_depth &&
              exact_input_height >= kernel_height &&
              exact_input_width >= kernel_width,
          "Calculated padded input size per channel: (",
          exact_input_depth,
          " x ",
          exact_input_height,
          " x ",
          exact_input_width,
          "). ",
          "Kernel size: (",
          kernel_depth,
          " x ",
          kernel_height,
          " x ",
          kernel_width,
          "). Kernel size can't be greater than actual input size");

      const i64 output_depth =
          div_rtn<i64>(exact_input_depth - kernel_depth, stride_depth) + 1;
      const i64 output_height =
          div_rtn<i64>(exact_input_height - kernel_height, stride_height) + 1;
      const i64 output_width =
          div_rtn<i64>(exact_input_width - kernel_width, stride_width) + 1;

      TORCH_CHECK(
          output_depth >= 1 && output_width >= 1 && output_height >= 1,
          "Given input size per channel: (",
          input_depth,
          " x ",
          input_height,
          " x ",
          input_width,
          "). "
          "Calculated output size per channel: (",
          output_depth,
          " x ",
          output_height,
          " x ",
          output_width,
          "). Output size is too small");

      if (weight.defined()) {
        i64 n_input_plane = weight.size(1);
        if (weight.dim() == 2) {
          n_input_plane /= (kernel_height * kernel_width);
        }
        // to support grouped conv we need to check if input.size(dim_planes)
        // is multiple of weight.size(dim_planes)
        TORCH_CHECK(groups > 0, "none zero group size expected");
        check_dim_size(input, ndim, dim_planes, n_input_plane * groups);
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
        check_dim_size(grad_output, ndim, dim_depth, output_depth);
        check_dim_size(grad_output, ndim, dim_height, output_height);
        check_dim_size(grad_output, ndim, dim_width, output_width);
      }
        */
}

pub fn view_weight_2d(weight: &Tensor) -> Tensor {
    
    todo!();
        /*
            Tensor weight = weight_.contiguous();
      if (weight.dim() == 5) {
        const i64 s1 = weight.size(0);
        const i64 s2 =
            weight.size(1) * weight.size(2) * weight.size(3) * weight.size(4);
        return weight.view({s1, s2});
      } else {
        return weight;
      }
        */
}

pub fn slow_conv3d_update_output_frame(
        input:          &mut Tensor,
        output:         &mut Tensor,
        weight:         &Tensor,
        bias:           &Tensor,
        finput:         &mut Tensor,
        kernel_depth:   i64,
        kernel_height:  i64,
        kernel_width:   i64,
        stride_depth:   i64,
        stride_height:  i64,
        stride_width:   i64,
        pad_depth:      i64,
        pad_height:     i64,
        pad_width:      i64,
        n_input_plane:  i64,
        groups:         i64,
        input_depth:    i64,
        input_height:   i64,
        input_width:    i64,
        n_output_plane: i64,
        output_depth:   i64,
        output_height:  i64,
        output_width:   i64)  {
    
    todo!();
        /*
            if ((kernel_depth == 1) && (kernel_height == 1) && (kernel_width == 1) &&
          (pad_depth == 0) && (pad_height == 0) && (pad_width == 0) &&
          (stride_depth == 1) && (stride_height == 1) && (stride_width == 1) && (groups == 1)) {
        auto output2d = output.reshape(
            {n_output_plane, output_depth * output_height * output_width});
        auto weight_new = weight.reshape({n_output_plane, n_input_plane});
        auto input_new = input.reshape({n_input_plane, output_depth * output_height * output_width});
        if (bias.defined()) {
          output.copy_(bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1));
          output2d.addmm_(weight_new, input_new, 1, 1);
        } else {
          mm_out(output2d, weight_new, input_new);
        }
        return;
      }
      Unfold3dCopyCPU(
          input,
          n_input_plane,
          input_depth,
          input_height,
          input_width,
          output_depth,
          output_height,
          output_width,
          kernel_depth,
          kernel_height,
          kernel_width,
          stride_depth,
          stride_height,
          stride_width,
          pad_depth,
          pad_height,
          pad_width,
          &finput);

      if (groups > 1) {
        auto output2d =
            output.reshape({groups,
                            n_output_plane / groups,
                            output_depth * output_height * output_width});
        auto weight_g = weight.reshape(
            {groups,
             n_output_plane / groups,
             n_input_plane / groups * kernel_depth * kernel_height * kernel_width});
        auto finput_g = finput.reshape(
            {groups,
             n_input_plane / groups * kernel_depth * kernel_width * kernel_height,
             output_depth * output_height * output_width});

        if (bias.defined()) {
          output.copy_(bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1));
          output2d.baddbmm_(weight_g, finput_g, 1, 1);
        } else {
          bmm_out(output2d, weight_g, finput_g);
        }
      } else {
        auto output2d = output.reshape(
            {n_output_plane, output_depth * output_height * output_width});
        if (bias.defined()) {
          output.copy_(bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1));
          output2d.addmm_(weight, finput, 1, 1);
        } else {
          mm_out(output2d, weight, finput);
        }
      }
        */
}

pub fn slow_conv3d_backward_update_grad_input_frame(
        grad_input:    &mut Tensor,
        grad_output:   &Tensor,
        weight:        &Tensor,
        fgrad_input:   &mut Tensor,
        kernel_depth:  i64,
        kernel_height: i64,
        kernel_width:  i64,
        stride_depth:  i64,
        stride_height: i64,
        stride_width:  i64,
        pad_depth:     i64,
        pad_height:    i64,
        pad_width:     i64,
        groups:        i64)  {
    
    todo!();
        /*
            if (groups > 1) {
        auto n = grad_output.size(0);
        auto d = grad_output.size(1);
        auto h = grad_output.size(2);
        auto w = grad_output.size(3);
        auto grad_output_2d = grad_output.reshape({groups, n / groups, d * h * w});
        auto weight_g =
            weight.reshape({groups, weight.size(0), weight.size(1) / groups});
        auto fgrad_input_g = fgrad_input.reshape(
            {groups, fgrad_input.size(0) / groups, fgrad_input.size(1)});

        bmm_out(fgrad_input_g, weight_g, grad_output_2d);
      } else {
        auto grad_output_2d = grad_output.reshape(
            {grad_output.size(0),
             grad_output.size(1) * grad_output.size(2) * grad_output.size(3)});
        mm_out(fgrad_input, weight, grad_output_2d);
      }
      Unfold3dAccCPU(
          fgrad_input,
          grad_input.size(0),
          grad_input.size(1),
          grad_input.size(2),
          grad_input.size(3),
          grad_output.size(1),
          grad_output.size(2),
          grad_output.size(3),
          kernel_depth,
          kernel_height,
          kernel_width,
          stride_depth,
          stride_height,
          stride_width,
          pad_depth,
          pad_height,
          pad_width,
          &grad_input);
        */
}

pub fn slow_conv3d_backward_out_cpu_template(
        grad_input:  &mut Tensor,
        grad_output: &Tensor,
        input:       &Tensor,
        weight:      &Tensor,
        finput:      &Tensor,
        fgrad_input: &mut Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        groups:      i64)  {
    
    todo!();
        /*
            const i64 kernel_depth = kernel_size[0];
      const i64 kernel_height = kernel_size[1];
      const i64 kernel_width = kernel_size[2];
      const i64 pad_depth = padding[0];
      const i64 pad_height = padding[1];
      const i64 pad_width = padding[2];
      const i64 stride_depth = stride[0];
      const i64 stride_height = stride[1];
      const i64 stride_width = stride[2];

      slow_conv3d_shape_check(
          input,
          grad_output,
          weight,
          Tensor(),
          kernel_depth,
          kernel_height,
          kernel_width,
          stride_depth,
          stride_height,
          stride_width,
          pad_depth,
          pad_height,
          pad_width,
          groups,
          false);

      const Tensor weight2d = view_weight_2d(weight);
      const Tensor grad_output_contiguous = grad_output.contiguous();
      grad_input.resize_as_(input);
      TORCH_CHECK(grad_input.is_contiguous(), "grad_input must be contiguous")
      fgrad_input.resize_as_(finput);
      TORCH_CHECK(fgrad_input.is_contiguous(), "fgrad_input must be contiguous")
      fgrad_input.zero_();

      // if the weight is grouped, we need to transpose for each individual
      // group instead of the entire weight2d
      Tensor tweight2d;
      if (groups > 1) {
        tweight2d =
            weight2d.reshape({groups, weight2d.size(0) / groups, weight2d.size(1)})
                .permute({0, 2, 1})
                .reshape({weight2d.size(1), weight2d.size(0)});
      } else {
        tweight2d = weight2d.transpose(0, 1);
      }
      const i64 batch_size = input.size(0);
      parallel_for(
          0, batch_size, CONV3D_GRAIN_SALT, [&](i64 start, i64 end) {
            AutoDispatchBelowADInplaceOrView non_variable_type_mode;
            for (i64 t = start; t < end; t++) {
              Tensor grad_input_t = grad_input[t];
              Tensor grad_output_t = grad_output_contiguous[t];
              Tensor fgrad_input_t = fgrad_input[t];
              slow_conv3d_backward_update_grad_input_frame(
                  grad_input_t,
                  grad_output_t,
                  tweight2d,
                  fgrad_input_t,
                  kernel_depth,
                  kernel_height,
                  kernel_width,
                  stride_depth,
                  stride_height,
                  stride_width,
                  pad_depth,
                  pad_height,
                  pad_width,
                  groups);
            }
          });
        */
}

pub fn slow_conv3d_backward_parameters_frame(
        grad_weight: &mut Tensor,
        grad_bias:   &mut Tensor,
        grad_output: &mut Tensor,
        finput:      &Tensor,
        groups:      i64)  {
    
    todo!();
        /*
            auto grad_output_2d = groups > 1
          ? grad_output.view(
                {groups,
                 grad_output.size(0) / groups,
                 grad_output.size(1) * grad_output.size(2) * grad_output.size(3)})
          : grad_output.view(
                {grad_output.size(0),
                 grad_output.size(1) * grad_output.size(2) * grad_output.size(3)});

      if (grad_weight.defined()) {
        if (groups > 1) {
          auto grad_weight_g = grad_weight.reshape(
              {groups, grad_weight.size(0) / groups, grad_weight.size(1)});
          Tensor tfinput =
              finput.reshape({groups, finput.size(0) / groups, finput.size(1)})
                  .permute({0, 2, 1})
                  .contiguous();
          grad_weight_g.baddbmm_(grad_output_2d, tfinput);
        } else {
          const Tensor tfinput = finput.transpose(0, 1);
          grad_weight.addmm_(grad_output_2d, tfinput);
        }
      }

      if (grad_bias.defined()) {
        AT_DISPATCH_FLOATING_TYPES_AND(
            ScalarType::BFloat16,
            grad_output.scalar_type(),
            "slow_conv3d_backward_parameters",
            [&] {
              auto grad_bias_acc = grad_bias.accessor<Scalar, 1>();
              if (groups > 1) {
                grad_output_2d = grad_output_2d.reshape(
                    {grad_output.size(0),
                     grad_output.size(1) * grad_output.size(2) *
                         grad_output.size(3)});
              }
              auto grad_output_2d_acc = grad_output_2d.accessor<Scalar, 2>();
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

pub fn slow_conv3d_backward_parameters_out_cpu_template(
        grad_weight: &mut Tensor,
        grad_bias:   &mut Tensor,
        input:       &Tensor,
        grad_output: &Tensor,
        finput:      &Tensor,
        fgrad_input: &Tensor,
        kernel_size: &[i32],
        stride:      &[i32],
        padding:     &[i32],
        groups:      i64)  {
    
    todo!();
        /*
            CheckedFrom c = "slow_conv3d_backward_parameters_cpu";
      auto grad_weight_arg = TensorArg(grad_weight, "grad_weight_arg", 0);
      auto grad_bias_arg = TensorArg(grad_bias, "grad_bias_arg", 0);

      const i64 kernel_depth = kernel_size[0];
      const i64 kernel_height = kernel_size[1];
      const i64 kernel_width = kernel_size[2];
      const i64 pad_depth = padding[0];
      const i64 pad_height = padding[1];
      const i64 pad_width = padding[2];
      const i64 stride_depth = stride[0];
      const i64 stride_height = stride[1];
      const i64 stride_width = stride[2];

      slow_conv3d_shape_check(
          input,
          grad_output,
          grad_weight,
          grad_bias,
          kernel_depth,
          kernel_height,
          kernel_width,
          stride_depth,
          stride_height,
          stride_width,
          pad_depth,
          pad_height,
          pad_width,
          groups,
          true);

      Tensor grad_weight_2d;
      if (grad_weight.defined()) {
        checkContiguous(c, grad_weight_arg);
        grad_weight_2d = view_weight_2d(grad_weight);
      }

      if (grad_bias.defined()) {
        checkContiguous(c, grad_bias_arg);
      }

      auto grad_output_contiguous = grad_output.contiguous();

      const i64 batch_size = input.size(0);
      for (i64 t = 0; t < batch_size; t++) {
        Tensor grad_output_t = grad_output_contiguous[t];
        Tensor finput_t;
        if (grad_weight_2d.defined()) {
          finput_t = finput[t];
        }
        slow_conv3d_backward_parameters_frame(
            grad_weight_2d, grad_bias, grad_output_t, finput_t, groups);
      }
        */
}

pub fn slow_conv3d_forward_out_cpu(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        output:      &mut Tensor,
        finput:      &mut Tensor,
        fgrad_input: &mut Tensor) -> (&mut Tensor,&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      const i64 kernel_depth = kernel_size[0];
      const i64 kernel_height = kernel_size[1];
      const i64 kernel_width = kernel_size[2];
      const i64 pad_depth = padding[0];
      const i64 pad_height = padding[1];
      const i64 pad_width = padding[2];
      const i64 stride_depth = stride[0];
      const i64 stride_height = stride[1];
      const i64 stride_width = stride[2];

      // TODO: hacky way of deciding the groups
      // Assuming the group size is checked in upstream functions
      const i64 groups = self.size(1) / weight.size(1);

      slow_conv3d_shape_check(
          self,
          Tensor(),
          weight,
          bias,
          kernel_depth,
          kernel_height,
          kernel_width,
          stride_depth,
          stride_height,
          stride_width,
          pad_depth,
          pad_height,
          pad_width,
          groups,
          false);

      const Tensor input = self.contiguous();
      const Tensor weight_2d = view_weight_2d(weight);

      const i64 dim_planes = 1;
      const i64 dim_depth = 2;
      const i64 dim_height = 3;
      const i64 dim_width = 4;

      const i64 n_input_plane = input.size(dim_planes);
      const i64 input_depth = input.size(dim_depth);
      const i64 input_height = input.size(dim_height);
      const i64 input_width = input.size(dim_width);
      const i64 n_output_plane = weight_2d.size(0);
      const i64 output_depth =
          (input_depth + 2 * pad_depth - kernel_depth) / stride_depth + 1;
      const i64 output_height =
          (input_height + 2 * pad_height - kernel_height) / stride_height + 1;
      const i64 output_width =
          (input_width + 2 * pad_width - kernel_width) / stride_width + 1;

      const i64 batch_size = input.size(0);
      if ((kernel_depth == 1) && (kernel_height == 1) && (kernel_width == 1) &&
          (pad_depth == 0) && (pad_height == 0) && (pad_width == 0) &&
          (stride_depth == 1) && (stride_height == 1) && (stride_width == 1) && (groups == 1)) {
        finput = input.view({batch_size, n_input_plane, output_height * output_width * output_depth}).detach();
      } else {
        finput.resize_({batch_size,
                        n_input_plane * kernel_depth * kernel_height * kernel_width,
                        output_depth * output_height * output_width});
      }
      output.resize_(
          {batch_size, n_output_plane, output_depth, output_height, output_width});

      parallel_for(
          0, batch_size, CONV3D_GRAIN_SALT, [&](i64 start, i64 end) {
            AutoDispatchBelowADInplaceOrView non_variable_type_mode;
            for (i64 t = start; t < end; t++) {
              Tensor input_t = input[t];
              Tensor output_t = output[t];
              Tensor finput_t = finput[t];
              slow_conv3d_update_output_frame(
                  input_t,
                  output_t,
                  weight_2d,
                  bias,
                  finput_t,
                  kernel_depth,
                  kernel_height,
                  kernel_width,
                  stride_depth,
                  stride_height,
                  stride_width,
                  pad_depth,
                  pad_height,
                  pad_width,
                  n_input_plane,
                  groups,
                  input_depth,
                  input_height,
                  input_width,
                  n_output_plane,
                  output_depth,
                  output_height,
                  output_width);
            }
          });

      return tuple<Tensor&, Tensor&, Tensor&>(output, finput, fgrad_input);
        */
}

pub fn slow_conv3d_forward_cpu(
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
      native::slow_conv3d_forward_out_cpu(
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

pub fn slow_conv3d_backward_out_cpu(
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
        grad_bias:   &mut Tensor) -> (&mut Tensor,&mut Tensor,&mut Tensor) {
    
    todo!();
        /*
            // TODO: hacky way of determine the group size
      i64 groups = self.size(1) / weight.size(1);
      if (grad_input.defined()) {
        slow_conv3d_backward_out_cpu_template(
            grad_input,
            grad_output,
            self,
            weight,
            finput,
            const_cast<Tensor&>(
                fgrad_input), // cast away auto-generated const of buffer
            kernel_size,
            stride,
            padding,
            groups);
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
        slow_conv3d_backward_parameters_out_cpu_template(
            grad_weight,
            grad_bias,
            self,
            grad_output,
            finput,
            fgrad_input,
            kernel_size,
            stride,
            padding,
            groups);
      }

      return tuple<Tensor&, Tensor&, Tensor&>(
          grad_input, grad_weight, grad_bias);
        */
}

pub fn slow_conv3d_backward_cpu(
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

      native::slow_conv3d_backward_out_cpu(
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

pub fn slow_conv3d_out(
        self_:       &Tensor,
        weight:      &Tensor,
        kernel_size: &[i32],
        bias_opt:    &Option<Tensor>,
        stride:      &[i32],
        padding:     &[i32],
        output:      &mut Tensor) -> &mut Tensor {
    
    todo!();
        /*
            // See [Note: hacky wrapper removal for optional tensor]
      MaybeOwned<Tensor> bias_maybe_owned = borrow_from_optional_tensor(bias_opt);
      const Tensor& bias = *bias_maybe_owned;

      Tensor finput = empty({0}, self.options());
      Tensor fgrad_input = empty({0}, self.options());
      return get<0>(slow_conv3d_forward_out(
          output,
          finput,
          fgrad_input,
          self,
          weight,
          kernel_size,
          bias,
          stride,
          padding));
        */
}

pub fn slow_conv3d(
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

      return get<0>(slow_conv3d_forward(
          self, weight, kernel_size, bias, stride, padding));
        */
}
